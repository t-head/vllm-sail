# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric E2E oracles for the ACEXT fused MoE kernels (CI plan PART 4).

ACEXT only ships for ZW-810E, so the whole module is gated on ``cap80``; on a
ZW-890P host every item is skipped by ``tests/conftest.py`` rather than
failing.  Three precision families are covered:

* ``bf16``       - unquantized weights and activations
* ``w8a8-int8``  - per-output-channel int8 weights, per-token int8 activations
* ``w4a8-int8``  - packed int4 weights (``W4AInt8MoEMethod``), int8 activations

The golden for all three is the dense per-expert PyTorch MoE in
``tests/e2e/kernels/_reference.py``: routing weights are applied to the
per-expert down-projection output and summed, exactly like
``TopKWeightAndReduceNoOP`` expects the experts kernel to have done.

Weight layout note (w4a8): ``W4AInt8MoEMethod.create_weights`` declares the
raw buffers as ``[E, 2I, H // 2]`` / ``[E, H, I // 2]``, i.e. N-major with the
two int4 values of a byte adjacent along ``K``, and the per-output-channel
scales as ``[E, 2I, 1]`` / ``[E, H, 1]``.  ``process_weights_after_loading``
only relabels those buffers with ``view`` (to ``[E, H, I]`` / ``[E, I, H // 2]``)
without moving data, so the golden is built from the physical N-major order
and the ``view`` contract is asserted separately.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

import vllm_sail  # noqa: E402

vllm_sail.register_out_of_tree()

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

from tests.e2e.kernels import _reference as ref  # noqa: E402

pytestmark = [pytest.mark.ppu, pytest.mark.cap80]

# (num_tokens, num_experts, top_k, hidden, intermediate) - a curated subset of
# the plan's ranges.  The full cross product (4 x 3 x 3 x 3 = 108 cases) would
# dominate the suite runtime and blow the fp32 golden's memory budget, so each
# axis keeps its extremes: tokens 1..256, experts 8..256, top_k 2..8 and
# hidden 2048..4096.
MOE_CASES = [
    pytest.param(1, 8, 2, 2048, 1024, id="t1-e8-k2-h2048"),
    pytest.param(8, 8, 2, 2048, 1024, id="t8-e8-k2-h2048"),
    pytest.param(16, 64, 6, 2048, 512, id="t16-e64-k6-h2048"),
    pytest.param(64, 64, 8, 4096, 512, id="t64-e64-k8-h4096"),
    pytest.param(256, 128, 8, 2048, 512, id="t256-e128-k8-h2048"),
    pytest.param(4, 256, 8, 2048, 256, id="t4-e256-k8-h2048"),
]


@pytest.fixture(scope="module", autouse=True)
def acext_runtime():
    """Load the native extension and the ACEXT wheel strictly.

    A missing ``acext`` symbol or an unavailable ACEXT MoE path must fail the
    run; silently falling back to another backend would defeat the oracle.
    """
    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()

    import acext

    for name in ("fusedmoe_wrapper", "get_enum_from_booleans", "pad_to_multiple_of_16"):
        assert hasattr(acext, name), f"acext.{name} is missing from the PPU wheel"

    from vllm_sail.model_executor.layers.fused_moe.experts.acext import (
        is_acext_supported,
    )

    assert is_acext_supported(), "ACEXT fused MoE is unavailable on this device"
    assert current_platform.is_device_capability((8, 0))


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed(seed)


def _routing(tokens: int, experts: int, top_k: int, seed: int):
    """Distinct experts per token plus normalized routing weights."""
    generator = _generator(seed)
    ids = torch.stack(
        [
            torch.randperm(experts, device="cuda", generator=generator)[:top_k]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    raw = torch.rand((tokens, top_k), generator=generator, device="cuda") + 0.1
    weights = raw / raw.sum(dim=-1, keepdim=True)
    return weights, ids


def _make_experts(quant_config):
    """Instantiate ``AcextExperts`` without a full ``FusedMoEConfig``."""
    from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

    expert = AcextExperts.__new__(AcextExperts)
    expert.quant_config = quant_config
    expert.ep_rank = 0
    expert.ep_size = 1
    return expert


def _run_acext(expert, hidden, w1, w2, weights, ids):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    output = torch.empty_like(hidden)
    empty = hidden.new_empty((0,))
    expert.apply(
        output,
        hidden,
        w1,
        w2,
        weights,
        ids,
        MoEActivation.SILU,
        w1.shape[0],
        None,  # expert_map
        None,  # a1q_scale
        None,  # a2_scale
        empty,  # workspace13 (ACEXT allocates internally)
        empty,  # workspace2
        None,  # expert_tokens_meta
        False,  # apply_router_weight_on_input
    )
    torch.cuda.synchronize()
    return output


# ---------------------------------------------------------------------------
# Contract-level checks (cheap, always run)
# ---------------------------------------------------------------------------


def test_acext_support_contract():
    """ACEXT advertises exactly the schemes the capability matrix promises."""
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kInt8DynamicTokenSym,
        kInt8StaticChannelSym,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.acext import AcextExperts

    assert AcextExperts._supports_current_device() is True
    assert AcextExperts.activation_format() == mk.FusedMoEActivationFormat.Standard
    assert AcextExperts._supports_no_act_and_mul() is False
    assert AcextExperts._supports_activation(MoEActivation.SILU) is True
    assert AcextExperts._supports_activation(MoEActivation.GELU) is False
    assert AcextExperts._supports_quant_scheme(None, None) is True
    assert (
        AcextExperts._supports_quant_scheme(kInt8StaticChannelSym, kInt8DynamicTokenSym)
        is True
    )
    assert AcextExperts._supports_quant_scheme(kInt8StaticChannelSym, None) is False

    expert = _make_experts(None)
    assert expert.expects_unquantized_inputs is True
    assert expert.supports_expert_map() is False
    assert isinstance(
        expert.finalize_weight_and_reduce_impl(), mk.TopKWeightAndReduceNoOP
    )
    assert expert.workspace_shapes(
        8, 2048, 2048, 2, 8, 8, None, MoEActivation.SILU
    ) == (
        (0,),
        (0,),
        (8, 2048),
    )


# ---------------------------------------------------------------------------
# bf16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("tokens", "experts", "top_k", "hidden", "inter"), MOE_CASES)
def test_acext_fused_moe_bf16(tokens, experts, top_k, hidden, inter):
    """Unquantized ACEXT fused MoE against the dense per-expert golden."""
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

    generator = _generator(101)
    x = torch.randn(
        (tokens, hidden), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    w1 = (
        torch.randn(
            (experts, 2 * inter, hidden),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            (experts, hidden, inter),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    weights, ids = _routing(tokens, experts, top_k, seed=102)

    expert = _make_experts(FusedMoEQuantConfig.make(torch.bfloat16))
    assert expert.quant_config.use_int8_w8a8 is False
    assert expert.quant_config.use_fp8_w8a8 is False
    output = _run_acext(expert, x, w1, w2, weights, ids)

    expected = ref.ref_moe(x, w1, w2, weights, ids)
    assert output.shape == (tokens, hidden) and output.dtype == torch.bfloat16
    ref.assert_close(output, expected, "bf16", chip="cap80")


# ---------------------------------------------------------------------------
# w8a8 int8
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("tokens", "experts", "top_k", "hidden", "inter"), MOE_CASES)
def test_acext_fused_moe_w8a8_int8(tokens, experts, top_k, hidden, inter):
    """Per-channel int8 weights + per-token int8 activations (kernel-side)."""
    from vllm.model_executor.layers.fused_moe.config import int8_w8a8_moe_quant_config

    generator = _generator(201)
    x = torch.randn(
        (tokens, hidden), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    w1 = (
        torch.randn(
            (experts, 2 * inter, hidden), generator=generator, device="cuda"
        ).float()
        * 0.05
    )
    w2 = (
        torch.randn(
            (experts, hidden, inter), generator=generator, device="cuda"
        ).float()
        * 0.05
    )
    # Per-output-channel int8 weights: the reduction runs over K (dim=-1 is
    # wrong here, dim=1 is the N axis of the [E, N, K] ACEXT layout).
    w1_q, w1_s = ref.quant_per_channel_int8(w1, dim=1)
    w2_q, w2_s = ref.quant_per_channel_int8(w2, dim=1)
    assert tuple(w1_s.shape) == (experts, 2 * inter, 1)
    assert tuple(w2_s.shape) == (experts, hidden, 1)
    weights, ids = _routing(tokens, experts, top_k, seed=202)

    quant_config = int8_w8a8_moe_quant_config(w1_s, w2_s, None, None)
    expert = _make_experts(quant_config)
    assert expert.quant_config.use_int8_w8a8 is True
    output = _run_acext(expert, x, w1_q, w2_q, weights, ids)

    expected = ref.ref_moe(
        x,
        ref.dequant_int8(w1_q, w1_s),
        ref.dequant_int8(w2_q, w2_s),
        weights,
        ids,
    )
    assert output.shape == (tokens, hidden) and output.dtype == torch.bfloat16
    ref.assert_close(output, expected, "int8", chip="cap80")


# ---------------------------------------------------------------------------
# w4a8 int8
# ---------------------------------------------------------------------------


def _w4a8_method(experts: int, hidden: int, inter: int):
    """Build a ``W4AInt8MoEMethod`` bound to a stub layer holding int4 weights."""
    from vllm_sail.model_executor.layers.quantization.mixed_precision_w4 import (
        MixedPrecisionW4Config,
        W4AInt8MoEMethod,
    )

    quant_config = MixedPrecisionW4Config()
    method = W4AInt8MoEMethod.__new__(W4AInt8MoEMethod)
    method.quant_config = quant_config
    method.ep_rank = 0
    method.ep_size = 1

    layer = torch.nn.Module()
    layer.activation = None
    pack = quant_config.pack_factor
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.empty(experts, 2 * inter, hidden // pack, dtype=torch.int8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.empty(experts, hidden, inter // pack, dtype=torch.int8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.ones(experts, 2 * inter, 1, dtype=torch.float32), requires_grad=False
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.ones(experts, hidden, 1, dtype=torch.float32), requires_grad=False
        ),
    )
    return method, layer, quant_config


@pytest.mark.parametrize(("tokens", "experts", "top_k", "hidden", "inter"), MOE_CASES)
def test_acext_fused_moe_w4a8_int8(tokens, experts, top_k, hidden, inter):
    """Packed int4 weights with kernel-side int8 activations."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    method, layer, quant_config = _w4a8_method(experts, hidden, inter)
    assert quant_config.pack_factor == 2

    generator = _generator(301)
    x = torch.randn(
        (tokens, hidden), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    # Logical int4 weights in checkpoint order: w13 is [E, 2I, H] and w2 is
    # [E, H, I]; the two int4 values sharing a byte are adjacent along K.
    w13_logical = torch.randint(
        -8, 8, (experts, 2 * inter, hidden), generator=generator, device="cuda"
    )
    w2_logical = torch.randint(
        -8, 8, (experts, hidden, inter), generator=generator, device="cuda"
    )
    w13_scale = (
        ((torch.arange(2 * inter, device="cuda") % 3 + 1).float() * 0.005)
        .view(1, -1, 1)
        .expand(experts, 2 * inter, 1)
        .contiguous()
    )
    w2_scale = (
        ((torch.arange(hidden, device="cuda") % 2 + 1).float() * 0.005)
        .view(1, -1, 1)
        .expand(experts, hidden, 1)
        .contiguous()
    )

    w13_packed = ref.pack_int4_along_last(w13_logical)
    w2_packed = ref.pack_int4_along_last(w2_logical)
    # The pack/unpack codec must round-trip before it can anchor a golden.
    assert torch.equal(
        ref.unpack_int4_along_last(w13_packed), w13_logical.to(torch.int32)
    )
    assert torch.equal(
        ref.unpack_int4_along_last(w2_packed), w2_logical.to(torch.int32)
    )

    layer.w13_weight = torch.nn.Parameter(w13_packed, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_packed, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    layer.activation = MoEActivation.SILU
    assert tuple(layer.w13_weight.shape) == (experts, 2 * inter, hidden // 2)
    assert tuple(layer.w2_weight.shape) == (experts, hidden, inter // 2)

    # process_weights_after_loading is a pure relabel: same storage, new shape.
    w13_ptr, w2_ptr = layer.w13_weight.data_ptr(), layer.w2_weight.data_ptr()
    method.process_weights_after_loading(layer)
    assert tuple(layer.w13_weight.shape) == (experts, hidden, inter)
    assert tuple(layer.w2_weight.shape) == (experts, inter, hidden // 2)
    assert layer.w13_weight.data_ptr() == w13_ptr
    assert layer.w2_weight.data_ptr() == w2_ptr

    weights, ids = _routing(tokens, experts, top_k, seed=302)
    output = method.apply(layer, x, weights, ids, None, None)
    torch.cuda.synchronize()

    expected = ref.ref_moe_from_packed_w4(
        x,
        w13_packed,
        w2_packed,
        weights,
        ids,
        w13_scale,
        w2_scale,
    )
    assert output.shape == (tokens, hidden) and output.dtype == torch.bfloat16
    ref.assert_close(output, expected, "w4a8", chip="cap80")
