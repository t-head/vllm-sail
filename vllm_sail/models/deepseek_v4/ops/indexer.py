# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU 1.0 DeepSeek V4 INT8 indexer query and compressed-key quantization.

The query kernel follows the rebased PPU fork's scaled-FP32 implementation;
PyTorch converts the result to INT8. The compressor preserves vLLM 0.27's
compression, RMSNorm, RoPE and paged addressing, but stores INT8 bytes with
FP32 scales for the plugin's int8_mqa_logits / int8_paged_mqa_logits consumers.
Neither path emits Triton's unsupported float8e4nv type.
"""

from typing import Any

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fused_indexer_q_rope_fp32_kernel(
    pos_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    out_fp32_ptr,
    out_stride0,
    out_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    """Emit scaled FP32 Q for the wrapper's truncating PyTorch INT8 copy.

    Ported from the rebased PPU fork. The INT8 scale divisor is 127; Q's
    scale folds into FP32 weights, so it needs no UE8M0 rounding.
    """
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )
    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
    index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 127.0)

    out_base_ptr = out_fp32_ptr + tok_idx * out_stride0 + head_idx * out_stride1
    if INDEX_Q_NOPE_DIM > 0:
        tl.store(out_base_ptr + nope_offset, tl.div_rn(x_nope, index_q_scale))
    out_rot_base = out_base_ptr + INDEX_Q_NOPE_DIM
    tl.store(out_rot_base + half_offset * 2, tl.div_rn(r_even, index_q_scale))
    tl.store(out_rot_base + half_offset * 2 + 1, tl.div_rn(r_odd, index_q_scale))

    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    )
    index_weights = index_weights.to(tl.float32)
    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


@triton.jit
def _compress_indexer_rope_int8_kernel(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,  # 128 for indexer
    TOKEN_STRIDE: tl.constexpr,  # 128 for indexer
    SCALE_DIM: tl.constexpr,  # 4 for indexer (1 float32)
    KV_BLOCK_STRIDE: tl.constexpr,
):
    """Fused compress → RMSNorm → RoPE → INT8 quant → store.

    One program per token; early-exits for non-boundary positions.

    Cache block layout:
      [0, bs*128):       INT8 data (128 bytes/token)
      [bs*128, +bs*4):   float32 scales (4 bytes/token)

    For head_dim=128 we have exactly one quant block, so we skip the
    [N_QUANT_BLOCKS, QUANT_BLOCK] reshape entirely and use a flat
    ``tl.max`` reduction.
    """
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── KV cache pointers ────────────────────────────────────────────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    quant_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # fp32

    # INT8 quantization, one FP32 scale per compressed token.
    tl.static_assert(
        TRITON_BLOCK_SIZE == QUANT_BLOCK,
        "Indexer expects one quant block (QUANT_BLOCK == TRITON_BLOCK_SIZE)",
    )
    result_bf16 = result.to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(result_bf16), axis=0), 1e-4)
    scale = tl.div_rn(absmax, 127.0)
    # Match the query's truncating INT8 conversion, preserving signed bytes.
    quant = tl.clamp(tl.div_rn(result_bf16, scale), -127.0, 127.0).to(tl.int8)
    tl.store(quant_ptr + block, quant.to(tl.uint8, bitcast=True), mask=mask)
    tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), scale)


def fused_indexer_q_rope_quant_int8(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPT-J RoPE on Q's tail, INT8 quantization and FP32 weight-scale folding.

    As in the PPU fork, rotated Q passes through BF16 before quantization.
    Scale is max(abs(Q), 1e-4) / 127, without power-of-two rounding; INT8
    conversion truncates toward zero. Optional buffers hold INT8 Q and FP32
    weights, in that order, and are returned by identity.
    """
    assert positions.ndim == 1 and index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2
    num_tokens, num_heads, head_dim = index_q.shape
    assert positions.shape[0] == num_tokens
    assert index_weights.shape == (num_tokens, num_heads)
    assert index_q.stride(-1) == index_weights.stride(-1) == 1
    assert index_q_cos_sin_cache.stride(-1) == 1
    rope_dim = index_q_cos_sin_cache.shape[-1]
    assert 0 < rope_dim <= head_dim and rope_dim % 2 == 0
    if output_buffers is None:
        q_out = torch.empty_like(index_q, dtype=torch.int8)
        weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    else:
        assert len(output_buffers) == 2
        q_out, weights_out = output_buffers
        assert q_out.shape == index_q.shape and q_out.dtype == torch.int8
        assert weights_out.shape == index_weights.shape
        assert weights_out.dtype == torch.float32
        assert q_out.device == weights_out.device == index_q.device
        assert weights_out.stride(-1) == 1
    if num_tokens == 0:
        return q_out, weights_out
    out_fp32 = torch.empty(index_q.shape, dtype=torch.float32, device=index_q.device)
    _fused_indexer_q_rope_fp32_kernel[(num_tokens, num_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        rope_dim // 2,
        out_fp32,
        out_fp32.stride(0),
        out_fp32.stride(1),
        head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        weights_out,
        weights_out.stride(0),
        num_warps=1,
    )
    q_out.copy_(out_fp32)
    return q_out, weights_out


def compress_indexer_rope_store_int8(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Write INT8 indexer keys using vLLM's byte-plane/FP32-scale page layout."""
    assert head_dim == quant_block == token_stride == 128
    assert scale_dim == 4 and not use_fp4_cache
    assert kv_cache.dtype == torch.uint8
    if num_actual == 0:
        return
    kernel = _compress_indexer_rope_int8_kernel
    num_warps = 1

    kernel[(num_actual,)](
        # state cache
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        # metadata
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        # RMSNorm
        rms_norm_weight,
        rms_norm_eps,
        # RoPE
        cos_sin_cache,
        cos_sin_cache.stride(0),
        # KV cache
        kv_cache,
        k_cache_metadata.slot_mapping,
        kv_cache.shape[1],  # paged KV cache block size (tokens per block)
        # constexprs
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=num_warps,
        **pdl_kwargs,
    )
