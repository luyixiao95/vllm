# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the packed-arena KV cache allocation used by multi-group
uniform-type models (e.g. DeepSeek V4 hybrids).

Each group packs its layers densely into a per-block byte window; groups
overlay each other in one arena (a block ID is owned by one group at a
time), so bytes per block is the max of the group window sizes — parity
with the pre-refactor packed allocator.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.core.kv_cache_utils import (
    _get_packed_kv_cache_arena,
    _pool_bytes_per_block,
    _use_packed_kv_cache_arena,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheLayout,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import reshape_packed_arena_kv_cache


def _make_mla_spec(head_size: int, block_size: int = 64) -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
    )


def _make_groups(n_big=3, n_small=3, n_tiny=2, n_swa=5):
    big_specs = {}
    for i in range(n_big):
        big_specs[f"mla.{i}"] = _make_mla_spec(512)
    for i in range(n_small):
        big_specs[f"idx.{i}"] = _make_mla_spec(128)
    for i in range(n_tiny):
        big_specs[f"tiny.{i}"] = _make_mla_spec(64)

    mla_group = KVCacheGroupSpec(
        layer_names=list(big_specs),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=big_specs),
    )

    swa_specs = {f"swa.{i}": _make_mla_spec(512) for i in range(n_swa)}
    swa_group = KVCacheGroupSpec(
        layer_names=list(swa_specs),
        kv_cache_spec=UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=swa_specs),
    )
    return [mla_group, swa_group]


def _mock_vllm_config():
    config = MagicMock()
    config.cache_config.num_gpu_blocks_override = None
    return config


def _page_sizes_by_layer(groups):
    return {
        layer_name: group.kv_cache_spec.kv_cache_specs[layer_name].page_size_bytes
        for group in groups
        for layer_name in group.layer_names
    }


def _group_window_bytes(groups):
    pages = _page_sizes_by_layer(groups)
    return [sum(pages[n] for n in g.layer_names) for g in groups]


class TestPackedArena:
    def test_arena_gate(self):
        groups = _make_groups()
        assert _use_packed_kv_cache_arena(groups)
        assert not _use_packed_kv_cache_arena(groups[:1])
        full = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
        )
        mixed = [groups[0], KVCacheGroupSpec(["full.0"], full)]
        assert not _use_packed_kv_cache_arena(mixed)

    def test_block_stride_is_max_group_window(self):
        groups = _make_groups()
        block_stride, offsets, shared_by = _get_packed_kv_cache_arena(groups)
        assert block_stride == max(_group_window_bytes(groups))
        pages = _page_sizes_by_layer(groups)
        # Each group's layers are dense and fit within the window.
        for group in groups:
            expected_offset = 0
            layer_offsets = {
                name: off for off, names in zip(offsets, shared_by) for name in names
            }
            for layer_name in group.layer_names:
                assert layer_offsets[layer_name] == expected_offset
                expected_offset += pages[layer_name]
            assert expected_offset <= block_stride

    def test_groups_overlay_at_shared_offsets(self):
        groups = _make_groups()
        _, offsets, shared_by = _get_packed_kv_cache_arena(groups)
        # Both groups start at offset 0: the SWA group's window aliases the
        # MLA group's bytes.
        assert set(shared_by[offsets.index(0)]) == {"mla.0", "swa.0"}

    def test_single_arena_tensor_and_accounting(self):
        groups = _make_groups()
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(), groups, 100 * 1024 * 1024
        )
        (tensor,) = config.kv_cache_tensors
        block_stride = max(_group_window_bytes(groups))
        assert tensor.block_stride == block_stride
        assert config.num_blocks == 100 * 1024 * 1024 // block_stride
        assert tensor.size == block_stride * config.num_blocks
        assert _pool_bytes_per_block(groups) == block_stride
        all_names = {n for slot in tensor.shared_by for n in slot}
        assert all_names == set(_page_sizes_by_layer(groups))

    def test_group_owned_blocks_do_not_alias(self):
        groups = _make_groups()
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(), groups, 8 * 1024 * 1024
        )
        (tensor,) = config.kv_cache_tensors
        buf = torch.zeros(tensor.size, dtype=torch.int8)
        views = reshape_packed_arena_kv_cache(
            buf, tensor, config.kv_cache_groups, None, KVCacheLayout.LBNHC
        )
        g1, g2 = (g.layer_names for g in groups)
        # Blocks 0 and 2 owned by group 1; blocks 1 and 3 by group 2.
        for i, name in enumerate(g1):
            views[name][0].fill_(i + 1)
            views[name][2].fill_(i + 1)
        for i, name in enumerate(g2):
            views[name][1].fill_(100 + i)
            views[name][3].fill_(100 + i)
        for i, name in enumerate(g1):
            assert (views[name][0].to(torch.int32) == i + 1).all()
            assert (views[name][2].to(torch.int32) == i + 1).all()
        for i, name in enumerate(g2):
            assert (views[name][1].to(torch.int32) == 100 + i).all()
            assert (views[name][3].to(torch.int32) == 100 + i).all()
        # Layers within a group stay disjoint.
        views[g1[0]][0].fill_(77)
        for i, name in enumerate(g1[1:], start=1):
            assert (views[name][0].to(torch.int32) == i + 1).all()

    def test_plane_layouts_rejected(self):
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        groups = _make_groups()
        set_kv_cache_layout("LHBNC")
        try:
            with pytest.raises(ValueError, match="packed KV cache arena"):
                get_kv_cache_config_from_groups(
                    _mock_vllm_config(), groups, 8 * 1024 * 1024
                )
        finally:
            set_kv_cache_layout(None)

    def test_non_uniform_multi_group_uses_slot_slabs(self):
        full_specs = {
            f"full.{i}": FullAttentionSpec(
                block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
            )
            for i in range(2)
        }
        sw_specs = {
            f"sw.{i}": FullAttentionSpec(
                block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
            )
            for i in range(2)
        }
        groups = [
            KVCacheGroupSpec(list(full_specs), next(iter(full_specs.values()))),
            KVCacheGroupSpec(list(sw_specs), next(iter(sw_specs.values()))),
        ]
        config = get_kv_cache_config_from_groups(
            _mock_vllm_config(), groups, 8 * 1024 * 1024
        )
        for tensor in config.kv_cache_tensors:
            assert tensor.block_stride is None
            assert tensor.slot_offsets is None
