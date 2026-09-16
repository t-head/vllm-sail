# SPDX-License-Identifier: Apache-2.0
"""Tier A activation loading and numerical checks on a real PPU."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def ppu_runtime():
    import torch

    import vllm_sail
    from vllm_sail.native.extensions import import_kernels

    # Run before vLLM imports: preserve the actual extension error if the wheel
    # is missing, shadowed by a checkout, or incompatible with SAIL torch/SDK.
    print(f"vllm-sail source: {vllm_sail.__file__}")
    import_kernels(strict=True)

    from vllm.platforms import current_platform

    assert current_platform.device_name == "ppu", current_platform
    current_platform.import_kernels()
    return torch


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("shape", [(1, 64), (7, 130), (32, 4096)])
def test_silu_and_mul(ppu_runtime, dtype_name, shape):
    torch = ppu_runtime

    tokens, hidden = shape
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(
        (tokens, hidden * 2), device="cuda", dtype=dtype, generator=generator
    )
    out = torch.empty((tokens, hidden), device=x.device, dtype=dtype)
    reference = (
        torch.nn.functional.silu(x[:, :hidden].float()) * x[:, hidden:].float()
    ).to(dtype)

    # vLLM's SiluAndMul binds this op directly; current vLLM has no
    # vllm._custom_ops.silu_and_mul Python wrapper.
    torch.ops._C.silu_and_mul(out, x)
    torch.cuda.synchronize()
    tolerance = {"float16": 2e-3, "bfloat16": 2e-2, "float32": 1e-5}[dtype_name]
    torch.testing.assert_close(out, reference, rtol=tolerance, atol=tolerance)
