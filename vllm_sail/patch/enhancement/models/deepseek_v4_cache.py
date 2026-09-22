# SPDX-License-Identifier: Apache-2.0
"""Route packed MLA cache reads to software FP8 decoding on PPU 1.0."""

from __future__ import annotations

import torch
from vllm.models.deepseek_v4.common.ops import cache_utils
from vllm.platforms import current_platform

from vllm_sail.models.deepseek_v4.ops.cache import register_ops
from vllm_sail.patch.utils import patch

register_ops()
_upstream = cache_utils.dequantize_and_gather_k_cache_triton


@patch(
    "vllm.models.deepseek_v4.common.ops.cache_utils",
    "dequantize_and_gather_k_cache_triton",
    reason="PPU 1.0 Triton cannot compile the packed cache's hardware FP8 bitcast; use software E4M3 decoding.",
    affected_versions=">=0.30.0,<0.31.0",
    remove_when="vLLM exposes packed MLA cache kernel registration or supports software FP8 decoding on PPU 1.0.",
)
def dequantize_and_gather_k_cache_triton(
    out,
    k_cache,
    seq_lens,
    gather_lens,
    block_table,
    block_size,
    offset,
    use_fnuz=False,
):
    if current_platform.is_ppu() and current_platform.is_device_capability((8, 0)):
        assert not use_fnuz, "PPU packed MLA cache uses E4M3FN, not FNUZ"
        return torch.ops.vllm.ppu_deepseek_v4_dequant_gather(
            out,
            k_cache,
            seq_lens,
            gather_lens,
            block_table,
            block_size,
            offset,
        )
    return _upstream(
        out, k_cache, seq_lens, gather_lens, block_table, block_size, offset, use_fnuz
    )
