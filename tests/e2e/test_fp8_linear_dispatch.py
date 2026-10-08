# SPDX-License-Identifier: Apache-2.0
"""FP8 scale selection and dense numerical checks on a real PPU."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize(
    ("activation_key", "weight_key"),
    [
        ("kFp8DynamicTokenSym", "kFp8StaticChannelSym"),
        ("kFp8DynamicTokenSym", "kFp8StaticTensorSym"),
        ("kFp8DynamicTensorSym", "kFp8StaticChannelSym"),
        ("kFp8StaticTensorSym", "kFp8StaticChannelSym"),
        ("kFp8DynamicTensorSym", "kFp8StaticTensorSym"),
        ("kFp8StaticTensorSym", "kFp8StaticTensorSym"),
    ],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_fp8_linear_scale_granularity(activation_key, weight_key, with_bias):
    import torch

    import vllm_sail

    vllm_sail.register_out_of_tree()
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.layers.quantization.utils import quant_utils
    from vllm.platforms import current_platform

    assert current_platform.is_ppu()
    tokens, outputs, hidden = 7, 128, 256
    generator = torch.Generator().manual_seed(31)
    # Integer inputs remain exact after E4M3 quantization with a 2/448 scale.
    x = torch.randint(-2, 3, (tokens, hidden), generator=generator).float()
    weight = torch.randint(-3, 4, (hidden, outputs), generator=generator).float()
    channelwise = weight_key == "kFp8StaticChannelSym"
    scale = (
        torch.arange(outputs).remainder(2).add(1).float().view(outputs, 1)
        if channelwise
        else torch.tensor([2.0])
    )
    bias = torch.arange(outputs).remainder(3).float() if with_bias else None
    reference = x @ (weight * scale.view(1, -1))
    if bias is not None:
        reference += bias

    with set_current_vllm_config(VllmConfig()):
        kernel = init_fp8_linear_kernel(
            getattr(quant_utils, activation_key),
            getattr(quant_utils, weight_key),
            torch.bfloat16,
            torch.bfloat16,
            (outputs, hidden),
        )
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            weight.to(device="cuda", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(scale.to("cuda"), requires_grad=False)
        layer.input_scale = (
            torch.tensor([2.0 / 448], device="cuda")
            if activation_key == "kFp8StaticTensorSym"
            else None
        )
        kernel.process_weights_after_loading(layer)
        output = kernel.apply_weights(
            layer,
            x.to(device="cuda", dtype=torch.bfloat16),
            None if bias is None else bias.to(device="cuda", dtype=torch.bfloat16),
        )
        torch.cuda.synchronize()
    assert output.shape == (tokens, outputs)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output.cpu().float(), reference, rtol=1e-2, atol=1)
