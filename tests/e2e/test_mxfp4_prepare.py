# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual prepare/quantizer boundary used by PPU MXFP4 experts."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float32"])
def test_mxfp4_prepare_uses_ppu_quantizer(dtype_name):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.model_executor.layers.fused_moe.config import ocp_mx_moe_quant_config
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        PPUDeepGemmExpertsMXFP4,
    )

    quant = ocp_mx_moe_quant_config(
        quant_dtype="mxfp4",
        weight_dtype="mxfp4",
        w1_scale=torch.ones(1, device="cuda"),
        w2_scale=torch.ones(1, device="cuda"),
        block_shape=[1, 16],
    )
    # Constant rows have exact FP4 encodings: +/-1 at scale 2^-2 and
    # +2 at scale 2^-1 all have magnitude 4 (E2M1 code 6).
    x = torch.tensor([1, -1, 2], device="cuda", dtype=getattr(torch, dtype_name))
    x = x[:, None].expand(3, 128).contiguous()
    expert = PPUDeepGemmExpertsMXFP4.__new__(PPUDeepGemmExpertsMXFP4)
    packed, scales, *_ = MoEPrepareAndFinalizeNoDPEPModular().prepare(
        x,
        torch.ones(3, 1, device="cuda"),
        torch.zeros(3, 1, device="cuda", dtype=torch.int32),
        1,
        None,
        False,
        quant,
        defer_input_quant=expert.expects_unquantized_inputs,
    )
    torch.cuda.synchronize()
    assert packed.dtype == torch.uint8 and packed.shape == (3, 64)
    assert scales.dtype == torch.uint16 and scales.shape == (3, 2)
    torch.testing.assert_close(
        packed.cpu(),
        torch.tensor([0x66, 0xEE, 0x66], dtype=torch.uint8)[:, None].expand(3, 64),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scales.cpu().to(torch.int32),
        torch.tensor([0x7D7D, 0x7D7D, 0x7E7E], dtype=torch.int32)[:, None].expand(3, 2),
        rtol=0,
        atol=0,
    )
