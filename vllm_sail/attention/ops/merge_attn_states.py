# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merge attention partials on PPU, including software E4M3FN output.

Uses the LSE merge contract of vLLM's triton_merge_attn_states.py. All tensor
strides are explicit, and FP8 storage is passed as bytes so PPU 1.0 never needs
a Triton fp8e4nv cast. No host reads or synchronization are needed for capture.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _encode_e4m3fn(value):
    # Saturating round-to-nearest-even, including subnormals and signed zero.
    bits = value.to(tl.uint32, bitcast=True)
    sign = (bits >> 24) & 0x80
    magnitude = tl.abs(value)
    magnitude = tl.minimum(magnitude, 448.0)
    raw = magnitude.to(tl.uint32, bitcast=True)
    exponent = ((raw >> 23) & 255).to(tl.int32)
    # A normal E4M3FN mantissa keeps three of the 23 FP32 fraction bits.
    rounded = raw + 0x7FFFF + ((raw >> 20) & 1)
    normal = ((rounded >> 20).to(tl.int32) - (120 << 3)).to(tl.uint32)
    # All FP8 subnormals are integer multiples of 2**-9. Adding the FP32
    # magic bias performs an exact RNE rounding of the nine-bit-scaled value.
    subnormal = (magnitude * 512.0 + 8388608.0).to(tl.uint32, bitcast=True) - 0x4B000000
    encoded = tl.where(exponent < 121, subnormal, normal)
    encoded = tl.where(value != value, 0x7F, encoded)
    return (sign | encoded).to(tl.uint8)


@triton.jit
def _merge_kernel(
    output,
    output_lse,
    prefix,
    prefix_lse,
    suffix,
    suffix_lse,
    output_scale,
    out_strides: tl.constexpr,
    pre_strides: tl.constexpr,
    suf_strides: tl.constexpr,
    pl_strides: tl.constexpr,
    sl_strides: tl.constexpr,
    ol_strides: tl.constexpr,
    context_tokens,
    HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    FP8: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    dim = tl.arange(0, BLOCK)
    mask = dim < HEAD
    s = tl.load(
        suffix + token * suf_strides[0] + head * suf_strides[1] + dim * suf_strides[2],
        mask=mask,
        other=0,
    ).to(tl.float32)
    sl = tl.load(suffix_lse + head * sl_strides[0] + token * sl_strides[1])
    if token < context_tokens:
        p = tl.load(
            prefix
            + token * pre_strides[0]
            + head * pre_strides[1]
            + dim * pre_strides[2],
            mask=mask,
            other=0,
        ).to(tl.float32)
        pl = tl.load(prefix_lse + head * pl_strides[0] + token * pl_strides[1])
        # FA2 reports +inf for empty attention; FA3 reports -inf.
        pl = tl.where(pl == float("inf"), float("-inf"), pl)
        sl = tl.where(sl == float("inf"), float("-inf"), sl)
        maximum = tl.maximum(pl, sl)
        empty = maximum == float("-inf")
        maximum = tl.where(empty, 0.0, maximum)
        pw, sw = tl.exp(pl - maximum), tl.exp(sl - maximum)
        total = pw + sw
        denominator = tl.where(empty, 1.0, total)
        # Empty partials may contain uninitialized NaN scratch values.
        p = tl.where(pl == float("-inf"), 0.0, p)
        s = tl.where(sl == float("-inf"), 0.0, s)
        result = p * (pw / denominator) + s * (sw / denominator)
        lse = tl.where(empty, float("-inf"), tl.log(total) + maximum)
    else:
        result, lse = s, sl
    if FP8:
        result = _encode_e4m3fn(result / tl.load(output_scale))
    tl.store(
        output + token * out_strides[0] + head * out_strides[1] + dim * out_strides[2],
        result,
        mask=mask,
    )
    if WRITE_LSE:
        tl.store(output_lse + head * ol_strides[0] + token * ol_strides[1], lse)


def merge_attn_states(
    output,
    output_lse,
    prefix_output,
    prefix_lse,
    suffix_output,
    suffix_lse,
    prefill_tokens_with_context,
    output_scale=None,
):
    """Implement the upstream _C op's argument order and in-place outputs."""
    if (
        output.ndim != 3
        or prefix_output.shape != output.shape
        or suffix_output.shape != output.shape
    ):
        raise ValueError("attention outputs must share [tokens, heads, head_dim]")
    tokens, heads, head_dim = output.shape
    for lse in (prefix_lse, suffix_lse, output_lse):
        if lse is not None and (
            lse.shape != (heads, tokens) or lse.dtype != torch.float32
        ):
            raise ValueError("LSE tensors must be float32 [heads, tokens]")
    floating = (torch.float32, torch.float16, torch.bfloat16)
    if prefix_output.dtype not in floating or suffix_output.dtype not in floating:
        raise ValueError("merge inputs must be FP32, FP16 or BF16")
    fp8 = output.dtype == torch.float8_e4m3fn
    if fp8:
        if (
            output_scale is None
            or output_scale.numel() != 1
            or output_scale.dtype != torch.float32
        ):
            raise ValueError("FP8 merge output requires a float32 scalar output_scale")
    elif output.dtype not in floating or output_scale is not None:
        raise ValueError("output_scale is only valid for E4M3FN output")
    if head_dim == 0 or tokens == 0 or heads == 0:
        return
    context = (
        tokens if prefill_tokens_with_context is None else prefill_tokens_with_context
    )
    tensors = (
        prefix_output,
        suffix_output,
        prefix_lse,
        suffix_lse,
        output_lse,
        output_scale,
    )
    if any(value is not None and value.device != output.device for value in tensors):
        raise ValueError("merge tensors must be on the same device")
    with torch.cuda.device(output.device):
        _merge_kernel[(tokens, heads)](
            output.view(torch.uint8) if fp8 else output,
            output_lse,
            prefix_output,
            prefix_lse,
            suffix_output,
            suffix_lse,
            output_scale,
            output.stride(),
            prefix_output.stride(),
            suffix_output.stride(),
            prefix_lse.stride(),
            suffix_lse.stride(),
            output_lse.stride() if output_lse is not None else (0, 0),
            context,
            head_dim,
            triton.next_power_of_2(head_dim),
            output_lse is not None,
            fp8,
            enable_fp_fusion=False,
        )
