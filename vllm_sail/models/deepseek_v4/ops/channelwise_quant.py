# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PPU channelwise inverse-RoPE quantization for DeepSeek V4.

These kernels come from the PPU fork's rebased v0.28 DeepSeek V4 path, while
target vLLM 0.27 only provides block-scaled FP8 quantization. Keep the
compatibility implementation inside the plugin and use PPU-prefixed custom-op
names to avoid collisions.
"""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _fused_inv_rope_int8_quant_channelwise_per_group(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    int8_ptr,
    scale_ptr,
    num_tokens,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    int8_stride_token,
    int8_stride_group,
    scale_stride_token,
    scale_stride_group,
    heads_per_group: tl.constexpr,
    int8_max: tl.constexpr,
    eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Fused inverse RoPE + per-(token, group) symmetric INT8 quantisation.

    One program per (token, group) pair. Each program loads the full
    ``[heads_per_group, HEAD_DIM]`` tile, applies inverse RoPE to the
    trailing rope_dim slice of every head, computes a single absmax over
    the whole tile (one scale per (token, group)), then stores the
    symmetric INT8 quantised values into ``[T, G, heads_per_group * HEAD_DIM]``
    and the fp32 scale into ``[T, G, 1]`` so it broadcasts against the
    activation tensor of einsum equation ``"bhr,hdr->bhd"`` along the
    reduction axis.
    """
    pid_token = tl.program_id(0).to(tl.int64)
    pid_g = tl.program_id(1).to(tl.int64)

    if pid_token >= num_tokens:
        return

    head_offsets = tl.arange(0, heads_per_group)
    dim_offsets = tl.arange(0, HEAD_DIM)

    base = (
        o_ptr + pid_token * o_stride_token + (pid_g * heads_per_group) * o_stride_head
    )
    addrs = base + head_offsets[:, None] * o_stride_head + dim_offsets[None, :]
    x = tl.load(addrs).to(tl.float32)

    # -- inverse RoPE on trailing rope_dim per head -------------------------
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = dim_offsets >= ROPE_START
    rope_local = dim_offsets - ROPE_START
    partner_addrs = (
        base + head_offsets[:, None] * o_stride_head + (dim_offsets ^ 1)[None, :]
    )
    x_partner = tl.load(partner_addrs, mask=is_rope[None, :], other=0.0).to(tl.float32)
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v[None, :] + x_partner * sin_v[None, :]
    x_sub = x * cos_v[None, :] - x_partner * sin_v[None, :]
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even[None, :], x_add, x_sub)
    x = tl.where(is_rope[None, :], rotated, x)

    # -- per-(token, group) absmax → channelwise scale ----------------------
    absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    scale = absmax / int8_max

    # -- symmetric INT8 quantise --------------------------------------------
    x_q = tl.clamp(x / scale, -int8_max, int8_max)
    x_int8 = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    # -- store INT8 to [T, G, heads_per_group * HEAD_DIM] -------------------
    out_base = int8_ptr + pid_token * int8_stride_token + pid_g * int8_stride_group
    full_offsets = head_offsets[:, None] * HEAD_DIM + dim_offsets[None, :]
    tl.store(out_base + full_offsets, x_int8)

    # -- store scalar scale to [T, G, 1] ------------------------------------
    scale_addr = scale_ptr + pid_token * scale_stride_token + pid_g * scale_stride_group
    tl.store(scale_addr, scale)


def fused_inv_rope_int8_quant_channelwise(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + channelwise INT8 quant matching ``"bhr,hdr->bhd"``.

    Designed to feed PPU DeepGEMM ``int8_einsum``. The activation tensor is
    laid out as ``[T, G, R]`` (b=T, h=G, r=R = heads_per_group * head_dim),
    so the corresponding per-(token, group) scale has shape ``[T, G, 1]``
    — broadcasting against the reduction axis ``r`` as ``int8_einsum``
    expects (`(b, h, 1)` for `(b, h, r)`).

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16/fp16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] cos||sin (fp32).
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).

    Returns:
        o_int8: [T, G, heads_per_group * head_dim] int8.
        o_scale: [T, G, 1] float32 (one scale per (token, group)).
    """
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    int8_max_val = 127
    return torch.ops.vllm.ppu_fused_inv_rope_int8_quant_channelwise_kernel(
        o,
        positions,
        cos_sin_cache,
        heads_per_group,
        nope_dim,
        rope_dim // 2,
        int8_max_val,
        num_tokens,
        n_groups,
        d,
        head_dim,
    )


def _ppu_fused_inv_rope_int8_quant_channelwise_kernel_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    int8_max_val: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    int8_buf = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.int8,
        device=o.device,
    )
    scale_buf = torch.empty(
        (num_tokens, n_groups, 1),
        dtype=torch.float32,
        device=o.device,
    )
    grid = (num_tokens, n_groups)
    _fused_inv_rope_int8_quant_channelwise_per_group[grid](
        o,
        positions,
        cos_sin_cache,
        int8_buf,
        scale_buf,
        num_tokens,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        int8_stride_token=int8_buf.stride(0),
        int8_stride_group=int8_buf.stride(1),
        scale_stride_token=scale_buf.stride(0),
        scale_stride_group=scale_buf.stride(1),
        heads_per_group=heads_per_group,
        int8_max=int8_max_val,
        eps=1e-10,
        HEAD_DIM=head_dim,
        ROPE_START=rope_start,
        HALF_ROPE=half_rope,
        num_warps=1,
        num_stages=1,
    )
    return int8_buf, scale_buf


def _ppu_fused_inv_rope_int8_quant_channelwise_kernel_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    int8_max_val: int,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    int8_buf = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.int8,
        device=o.device,
    )
    scale_buf = torch.empty(
        (num_tokens, n_groups, 1),
        dtype=torch.float32,
        device=o.device,
    )
    return int8_buf, scale_buf


direct_register_custom_op(
    op_name="ppu_fused_inv_rope_int8_quant_channelwise_kernel",
    op_func=_ppu_fused_inv_rope_int8_quant_channelwise_kernel_impl,
    fake_impl=_ppu_fused_inv_rope_int8_quant_channelwise_kernel_fake,
)


@triton.jit
def _fused_inv_rope_fp8_quant_channelwise_per_group(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    fp8_ptr,
    scale_ptr,
    num_tokens,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    fp8_stride_token,
    fp8_stride_group,
    scale_stride_token,
    scale_stride_group,
    heads_per_group: tl.constexpr,
    fp8_max: tl.constexpr,
    eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Fused inverse RoPE + per-(token, group) symmetric FP8 quantisation.

    One program per (token, group) pair. Identical structure to the INT8
    channelwise kernel but casts to ``float8_e4m3fn`` and uses ``fp8_max``
    (448.0) as the symmetric range. The output scale is raw fp32 — no
    TMA / UE8M0 pre-transform — matching how ``int8_einsum`` receives raw
    per-channel scales.
    """
    pid_token = tl.program_id(0).to(tl.int64)
    pid_g = tl.program_id(1).to(tl.int64)

    if pid_token >= num_tokens:
        return

    head_offsets = tl.arange(0, heads_per_group)
    dim_offsets = tl.arange(0, HEAD_DIM)

    base = (
        o_ptr + pid_token * o_stride_token + (pid_g * heads_per_group) * o_stride_head
    )
    addrs = base + head_offsets[:, None] * o_stride_head + dim_offsets[None, :]
    x = tl.load(addrs).to(tl.float32)

    # -- inverse RoPE on trailing rope_dim per head -------------------------
    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
    is_rope = dim_offsets >= ROPE_START
    rope_local = dim_offsets - ROPE_START
    partner_addrs = (
        base + head_offsets[:, None] * o_stride_head + (dim_offsets ^ 1)[None, :]
    )
    x_partner = tl.load(partner_addrs, mask=is_rope[None, :], other=0.0).to(tl.float32)
    cs_idx = tl.maximum(rope_local >> 1, 0)
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
    x_add = x * cos_v[None, :] + x_partner * sin_v[None, :]
    x_sub = x * cos_v[None, :] - x_partner * sin_v[None, :]
    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even[None, :], x_add, x_sub)
    x = tl.where(is_rope[None, :], rotated, x)

    # -- per-(token, group) absmax -> channelwise scale ----------------------
    absmax = tl.maximum(tl.max(tl.abs(x)), eps)
    scale = absmax / fp8_max

    # -- symmetric FP8 quantise ---------------------------------------------
    x_fp8 = (x / scale).to(tl.float8e4nv)

    # -- store FP8 to [T, G, heads_per_group * HEAD_DIM] --------------------
    out_base = fp8_ptr + pid_token * fp8_stride_token + pid_g * fp8_stride_group
    full_offsets = head_offsets[:, None] * HEAD_DIM + dim_offsets[None, :]
    tl.store(out_base + full_offsets, x_fp8)

    # -- store scalar scale to [T, G, 1] ------------------------------------
    scale_addr = scale_ptr + pid_token * scale_stride_token + pid_g * scale_stride_group
    tl.store(scale_addr, scale)


def fused_inv_rope_fp8_quant_channelwise(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused inverse RoPE + channelwise FP8 quant matching ``"bhr,hdr->bhd"``.

    Designed to feed PPU DeepGEMM ``fp8_einsum``.  Same output layout as the
    INT8 channelwise variant: ``o_fp8 [T, G, R]`` + ``o_scale [T, G, 1]``.
    The scale is raw fp32 (no TMA / UE8M0 pre-transform), mirroring how
    ``int8_einsum`` receives raw per-channel scales.

    Args:
        o: Attention output [num_tokens, num_heads, head_dim] bf16/fp16.
        positions: Token positions [num_tokens] int64.
        cos_sin_cache: Precomputed [max_pos, rope_dim] cos||sin (fp32).
        n_groups: Number of output groups.
        heads_per_group: Heads per group.
        nope_dim: Non-RoPE dimensions per head (default 448).
        rope_dim: RoPE dimensions per head (default 64).

    Returns:
        o_fp8: [T, G, heads_per_group * head_dim] float8_e4m3fn.
        o_scale: [T, G, 1] float32 (one scale per (token, group)).
    """
    num_tokens, num_heads, head_dim = o.shape
    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    return torch.ops.vllm.ppu_fused_inv_rope_fp8_quant_channelwise_kernel(
        o,
        positions,
        cos_sin_cache,
        heads_per_group,
        nope_dim,
        rope_dim // 2,
        fp8_max,
        num_tokens,
        n_groups,
        d,
        head_dim,
    )


def _ppu_fused_inv_rope_fp8_quant_channelwise_kernel_impl(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    fp8_max: float,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_buf = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_buf = torch.empty(
        (num_tokens, n_groups, 1),
        dtype=torch.float32,
        device=o.device,
    )
    grid = (num_tokens, n_groups)
    _fused_inv_rope_fp8_quant_channelwise_per_group[grid](
        o,
        positions,
        cos_sin_cache,
        fp8_buf,
        scale_buf,
        num_tokens,
        o_stride_token=o.stride(0),
        o_stride_head=o.stride(1),
        cache_stride_pos=cos_sin_cache.stride(0),
        fp8_stride_token=fp8_buf.stride(0),
        fp8_stride_group=fp8_buf.stride(1),
        scale_stride_token=scale_buf.stride(0),
        scale_stride_group=scale_buf.stride(1),
        heads_per_group=heads_per_group,
        fp8_max=fp8_max,
        eps=1e-10,
        HEAD_DIM=head_dim,
        ROPE_START=rope_start,
        HALF_ROPE=half_rope,
        num_warps=1,
        num_stages=1,
    )
    return fp8_buf, scale_buf


def _ppu_fused_inv_rope_fp8_quant_channelwise_kernel_fake(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    heads_per_group: int,
    rope_start: int,
    half_rope: int,
    fp8_max: float,
    num_tokens: int,
    n_groups: int,
    d: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_buf = torch.empty(
        (num_tokens, n_groups, d),
        dtype=torch.float8_e4m3fn,
        device=o.device,
    )
    scale_buf = torch.empty(
        (num_tokens, n_groups, 1),
        dtype=torch.float32,
        device=o.device,
    )
    return fp8_buf, scale_buf


direct_register_custom_op(
    op_name="ppu_fused_inv_rope_fp8_quant_channelwise_kernel",
    op_func=_ppu_fused_inv_rope_fp8_quant_channelwise_kernel_impl,
    fake_impl=_ppu_fused_inv_rope_fp8_quant_channelwise_kernel_fake,
)
