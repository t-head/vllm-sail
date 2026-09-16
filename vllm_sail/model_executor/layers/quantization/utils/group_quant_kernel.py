# SPDX-License-Identifier: Apache-2.0
"""CUDA-free implementation of the fork's optional group quantization op."""

from vllm.triton_utils import tl, triton

from vllm_sail.models.deepseek_v4.ops.cache import _encode_e4m3fn


@triton.jit
def _quantize(
    x,
    q,
    scales,
    M: tl.constexpr,
    GROUPS: tl.constexpr,
    stride_batch: tl.constexpr,
    stride_row: tl.constexpr,
    stride_group: tl.constexpr,
    eps: tl.constexpr,
    qmin: tl.constexpr,
    qmax: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    UE8M0: tl.constexpr,
    SOFTWARE_FP8: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    offsets = (row * GROUPS + group) * GROUP_SIZE + cols
    values = tl.load(x + offsets, mask=cols < GROUP_SIZE, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(values), 0), eps) / qmax
    if UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.abs(scale), 1e-10))))
    quantized = tl.minimum(tl.maximum(values / scale, qmin), qmax)
    if SOFTWARE_FP8:
        quantized = _encode_e4m3fn(quantized)
    tl.store(q + offsets, quantized, mask=cols < GROUP_SIZE)
    soff = row // M * stride_batch + row % M * stride_row + group * stride_group
    tl.store(scales + soff, scale)


def per_token_group_fp8_quant_ppu_opt(
    input,
    output_q,
    output_s,
    group_size,
    eps,
    min_8bit,
    max_8bit,
    scale_ue8m0,
    column_major_scales,
    tma_aligned_scales,
):
    import torch
    from vllm.platforms import current_platform

    if input.ndim not in (2, 3) or not input.is_contiguous():
        raise ValueError("PPU group quantization requires contiguous 2D or 3D input")
    if group_size <= 0 or input.shape[-1] % group_size:
        raise ValueError("Group size must divide the hidden dimension")
    if output_q.shape != input.shape or not output_q.is_contiguous():
        raise ValueError(
            "Quantized output must have matching shape and contiguous storage"
        )
    if output_q.dtype != torch.float8_e4m3fn:
        raise ValueError("PPU group quantization requires E4M3FN output")
    m, n = input.shape[-2:]
    expected = (*input.shape[:-1], n // group_size)
    if tuple(output_s.shape) != expected or output_s.dtype != torch.float32:
        raise ValueError("Scale output must be float32 with one value per group")
    software = not current_platform.has_device_capability(89)
    _quantize[(input.numel() // n, n // group_size)](
        input,
        output_q.view(torch.uint8) if software else output_q,
        output_s,
        m,
        n // group_size,
        output_s.stride(0) if input.ndim == 3 else 0,
        output_s.stride(-2),
        output_s.stride(-1),
        eps,
        min_8bit,
        max_8bit,
        GROUP_SIZE=group_size,
        BLOCK=triton.next_power_of_2(group_size),
        UE8M0=scale_ue8m0,
        SOFTWARE_FP8=software,
        num_warps=4,
    )
