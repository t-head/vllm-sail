# SPDX-License-Identifier: Apache-2.0
"""PPU group FP8 quantization with portable Triton kernels and scale layouts."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.quantization.utils.quant_utils import get_fp8_min_max
from vllm.platforms import current_platform

from vllm_sail.utils.deep_gemm import get_tma_aligned_size, is_deep_gemm_e8m0_used


def per_token_group_quant_fp8_ppu_opt(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPU per-token-group FP8 quantization.

    Uses the plugin-owned portable Triton implementation of the native schema.
    Same interface as :func:`per_token_group_quant_fp8`.

    Args:
        x: The input tensor with ndim >= 2.
        group_size: The group size used for quantization.
        eps: The minimum to avoid dividing zero.
        dtype: The dtype of output tensor.
        column_major_scales: Outputs scales in column major.
        tma_aligned_scales: Outputs scales in TMA-aligned layout.
        out_q: Optional output tensor.
    Returns:
        tuple[torch.Tensor, torch.Tensor]: quantized tensor and
        scaling factor.
    """
    assert x.ndim in (2, 3)
    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()
    dtype = current_platform.fp8_dtype() if dtype is None else dtype
    assert x.shape[-1] % group_size == 0, (
        f"the last dimension of `x` {x.shape[-1]} must be "
        f"divisible by `group_size` {group_size}"
    )
    assert x.stride(-1) == 1, "`x` groups must be contiguous"

    fp8_min, fp8_max = get_fp8_min_max()

    assert out_q is None or out_q.shape == x.shape
    x_q = out_q
    if x_q is None:
        x_q = torch.empty(x.shape, device=x.device, dtype=dtype)

    # Allocate the scale tensor row- or column-major.
    if column_major_scales:
        if tma_aligned_scales:
            m = x.shape[-2]
            sf_k = x.shape[-1] // group_size
            tma_aligned_m = get_tma_aligned_size(m, 4)
            shape = x.shape[:-2] + (m, sf_k)
            stride = (
                (1, tma_aligned_m)
                if x.dim() == 2
                else (tma_aligned_m * sf_k, 1, tma_aligned_m)
            )
            x_s = torch.empty_strided(
                shape, stride, device=x.device, dtype=torch.float32
            )
        else:
            shape = x.shape[:-2] + (x.shape[-1] // group_size, x.shape[-2])
            x_s = torch.empty(shape, device=x.device, dtype=torch.float32).transpose(
                -1, -2
            )
    else:
        shape = x.shape[:-1] + (x.shape[-1] // group_size,)
        x_s = torch.empty(shape, device=x.device, dtype=torch.float32)

    # Use PPU-optimized CUDA kernel path
    if (
        current_platform.is_cuda_alike() or current_platform.is_xpu()
    ) and x.is_contiguous():
        torch.ops._C.per_token_group_fp8_quant_ppu_opt(
            x,
            x_q,
            x_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
            use_ue8m0,
            column_major_scales,
            tma_aligned_scales,
        )
        return x_q, x_s

    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    return per_token_group_quant_fp8(
        x,
        group_size,
        eps=eps,
        dtype=dtype,
        column_major_scales=column_major_scales,
        tma_aligned_scales=tma_aligned_scales,
        out_q=out_q,
        use_ue8m0=use_ue8m0,
    )
