# SPDX-License-Identifier: Apache-2.0
"""Numerical gates for the fork port; run on PPU after rebuilding the plugin."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture
def runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires PPU")
    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.platforms import current_platform

    if not current_platform.is_ppu():
        pytest.skip("PPU kernel validation")
    return torch, current_platform


@pytest.mark.parametrize("hidden", [2048, 3072])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_fused_norm_matches_native_then_quantize(runtime, hidden, column_major, dtype):
    torch, platform = runtime
    if not platform.has_device_capability(89):
        pytest.skip("fused RMSNorm FP8 path is enabled only on PPU 1.5")
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    from vllm_sail.model_executor.layers.quantization.utils.fused_add_rmsnorm_quant import (
        fused_add_rmsnorm_group_quant,
    )

    torch.manual_seed(17)
    x = torch.randn((7, hidden), device="cuda", dtype=getattr(torch, dtype))
    residual = torch.randn_like(x)
    weight = torch.randn((hidden,), device="cuda", dtype=x.dtype)
    normalized = x.clone()
    expected_residual = residual.clone()
    ops.fused_add_rms_norm(normalized, expected_residual, weight, 1e-6)
    expected_q, expected_s = per_token_group_quant_fp8(
        normalized, 128, column_major_scales=column_major, use_ue8m0=False
    )
    q, scales, result_residual = fused_add_rmsnorm_group_quant(
        x, residual, weight, 1e-6, column_major_scales=column_major
    )
    torch.testing.assert_close(result_residual, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_s, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(q.float(), expected_q.float(), rtol=0.125, atol=0.02)


@pytest.mark.parametrize("shape", [(7, 384), (2, 7, 384)])
@pytest.mark.parametrize(
    "column_major,tma", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("ue8m0", [False, True])
def test_portable_group_quant_layouts_and_scale_floor(
    runtime, shape, column_major, tma, ue8m0
):
    torch, _ = runtime
    from vllm_sail.model_executor.layers.quantization.utils.group_quant import (
        per_token_group_quant_fp8_ppu_opt,
    )

    torch.manual_seed(17)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    x.reshape(-1, shape[-1])[0].zero_()
    q, scales = per_token_group_quant_fp8_ppu_opt(
        x,
        128,
        column_major_scales=column_major,
        tma_aligned_scales=tma,
        use_ue8m0=ue8m0,
    )
    groups = x.float().reshape(*shape[:-1], -1, 128)
    expected_s = groups.abs().amax(-1).clamp_min(1e-10) / 448.0
    if ue8m0:
        expected_s = torch.exp2(torch.ceil(torch.log2(expected_s.clamp_min(1e-10))))
    expected_q = (
        (groups / expected_s.unsqueeze(-1))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape(shape)
    )
    torch.testing.assert_close(scales, expected_s, rtol=1e-6, atol=0)
    torch.testing.assert_close(q.float(), expected_q.float(), rtol=0, atol=0)
