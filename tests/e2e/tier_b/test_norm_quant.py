# SPDX-License-Identifier: Apache-2.0
"""CPU numerical oracles for the generated native normalization/quant kernels."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.fixture(scope="module")
def runtime():
    import torch

    from vllm_sail.native import install
    from vllm_sail.native.extensions import import_kernels

    import_kernels(strict=True)
    install()
    return torch


def _check_quant(torch, actual, expected):
    if expected.dtype == torch.int8:
        # FP32 reduction order can move a result across one integer boundary.
        assert (actual.cpu().short() - expected.short()).abs().max().item() <= 1
    else:
        got, ref = actual.cpu(), expected.contiguous()
        # E4M3FN NaN is adjacent to +/-448 in byte space. All references here
        # are finite, so reject overflow-to-NaN before allowing a one-code error.
        assert torch.isfinite(got.float()).all().item()
        distance = (got.view(torch.uint8).short() - ref.view(torch.uint8).short()).abs()
        # Equal signed zeros have different byte encodings.
        assert ((distance <= 1) | (got.float() == ref.float())).all().item()


def _encode(torch, values, dtype):
    if dtype == torch.int8:
        return values.round().clamp(-128, 127).to(dtype)
    return values.clamp(-448, 448).to(dtype)


def _quantize(torch, values, dtype, group, upper=None, upper_is_scale=False):
    tiles = values.reshape(values.shape[0], -1, group)
    maximum = tiles.abs().amax(-1)
    qmax = 127 if dtype == torch.int8 else 448
    minimum = torch.finfo(torch.float32).eps if dtype == torch.int8 else 1 / (448 * 512)
    if upper is not None and not upper_is_scale:
        maximum = maximum.clamp(max=upper)
    scales = maximum / qmax
    if upper is not None and upper_is_scale:
        scales = scales.clamp(max=upper)
    scales = scales.clamp(min=minimum)
    return _encode(torch, (tiles / scales[..., None]).reshape_as(values), dtype), scales


def _scales(torch, tokens, groups, layout):
    if layout == "row":
        storage = torch.full((tokens, groups), -123.0, device="cuda")
        return storage, storage
    padded = tokens if layout == "column" else (tokens + 3) // 4 * 4
    storage = torch.full((groups, padded), -123.0, device="cuda")
    return storage[:, :tokens].T, storage


def _run(torch, call, reset, graph=False):
    call()
    if graph:
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            reset()
            call()
        capture.replay()
    torch.cuda.synchronize()


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize(
    "tokens,hidden,scale_value", [(3, 128, 0.25), (257, 129, 0.003)]
)
def test_static_rms_fp8(runtime, dtype_name, add_residual, tokens, hidden, scale_value):
    torch = runtime
    torch.manual_seed(11)
    dtype = getattr(torch, dtype_name)
    # Padded row strides exercise the upstream scalar and vector load paths.
    storage = torch.randn(tokens, hidden + 8, device="cuda", dtype=dtype)
    x = storage[:, :hidden]
    before = storage.clone()
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x) if add_residual else None
    initial = residual.clone() if residual is not None else None
    scale = torch.tensor([scale_value], device="cuda")
    out = torch.empty(x.shape, device="cuda", dtype=torch.float8_e4m3fn)

    def call():
        if add_residual:
            torch.ops._C.fused_add_rms_norm_static_fp8_quant(
                out, x, residual, weight, scale, 1e-6
            )
        else:
            torch.ops._C.rms_norm_static_fp8_quant(out, x, weight, scale, 1e-6)

    def reset():
        if residual is not None:
            residual.copy_(initial)

    _run(torch, call, reset, graph=add_residual and tokens == 3)
    values = x.cpu().float()
    if initial is not None:
        # Static residual fusion rounds the sum before variance accumulation.
        values = (values + initial.cpu().float()).to(dtype).float()
        torch.testing.assert_close(residual.cpu(), values.to(dtype), rtol=0, atol=0)
    norm = (
        (
            values
            * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
            * weight.cpu().float()
        )
        .to(dtype)
        .float()
    )
    _check_quant(torch, out, _encode(torch, norm / scale_value, out.dtype))
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("quant_name", ["int8", "float8_e4m3fn"])
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize(
    "group,hidden,layout",
    [
        (None, 513, "row"),
        (None, 512, "row"),
        (64, 512, "row"),
        (128, 640, "column"),
        (64, 320, "tma"),
    ],
)
def test_dynamic_rms_quant(
    runtime, dtype_name, quant_name, add_residual, group, hidden, layout
):
    torch = runtime
    torch.manual_seed(23)
    dtype, quant_dtype = getattr(torch, dtype_name), getattr(torch, quant_name)
    tokens = 5
    storage = torch.randn(tokens, hidden + 4, device="cuda", dtype=dtype)
    x = storage[:, :hidden]
    x[-1].zero_()
    before = storage.clone()
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    residual = torch.randn_like(x) if add_residual else None
    if residual is not None:
        residual[-1].zero_()
    initial = residual.clone() if residual is not None else None
    # RMS kernels bound the absmax before dividing by qmax.
    upper_value = 0.5 if quant_dtype != torch.int8 else None
    upper = (
        torch.tensor([upper_value], device="cuda") if upper_value is not None else None
    )
    out = torch.empty(x.shape, device="cuda", dtype=quant_dtype)
    scales, scale_storage = _scales(torch, tokens, hidden // (group or hidden), layout)

    def call():
        if group is None:
            torch.ops._C.rms_norm_dynamic_per_token_quant(
                out, x, weight, scales, 1e-6, upper, residual
            )
        else:
            torch.ops._C.rms_norm_per_block_quant(
                out, x, weight, scales, 1e-6, upper, residual, group, layout != "row"
            )

    def reset():
        if residual is not None:
            residual.copy_(initial)

    _run(torch, call, reset, graph=layout == "tma" and add_residual)
    values = x.cpu().float()
    if initial is not None:
        # Dynamic fusion computes variance from the FP32 sum, while the saved
        # residual is rounded separately to the input dtype.
        values = values + initial.cpu().float()
        torch.testing.assert_close(residual.cpu(), values.to(dtype), rtol=0, atol=0)
    normalized = (
        values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    ).to(dtype)
    normalized = (normalized * weight.cpu()).float()
    expected, expected_scales = _quantize(
        torch, normalized, quant_dtype, group or hidden, upper_value
    )
    _check_quant(torch, out, expected)
    torch.testing.assert_close(scales.cpu(), expected_scales, rtol=1e-5, atol=1e-7)
    if layout == "tma":
        padding = scale_storage[:, tokens:]
        torch.testing.assert_close(
            padding, torch.full_like(padding, -123), rtol=0, atol=0
        )
    torch.testing.assert_close(storage, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize(
    "quant_name,bound",
    [("int8", False), ("float8_e4m3fn", False), ("float8_e4m3fn", True)],
)
@pytest.mark.parametrize("group", [64, 128])
@pytest.mark.parametrize("transpose", [False, True])
def test_silu_block_quant(runtime, dtype_name, quant_name, group, transpose, bound):
    torch = runtime
    torch.manual_seed(37)
    dtype, quant_dtype = getattr(torch, dtype_name), getattr(torch, quant_name)
    tokens, hidden = 7, 384
    x = torch.randn(tokens, hidden * 2, device="cuda", dtype=dtype)
    x[-1].zero_()
    before = x.clone()
    out = torch.empty(tokens, hidden, device="cuda", dtype=quant_dtype)
    scales, _ = _scales(
        torch, tokens, hidden // group, "column" if transpose else "row"
    )
    # SiLU's bound applies to the scale itself, unlike the RMS kernels.
    upper_value = 0.002 if bound else None
    upper = torch.tensor([upper_value], device="cuda") if bound else None

    def call():
        torch.ops._C.silu_and_mul_per_block_quant(
            out, x, scales, group, upper, transpose
        )

    _run(torch, call, lambda: None, graph=transpose and bound)
    gate, up = x.cpu().float().chunk(2, -1)
    values = (gate * (1.0 / (1.0 + torch.exp(-gate)))) * up
    expected, expected_scales = _quantize(
        torch, values, quant_dtype, group, upper_value, upper_is_scale=True
    )
    _check_quant(torch, out, expected)
    torch.testing.assert_close(scales.cpu(), expected_scales, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@pytest.mark.parametrize("group", [None, 64, 128])
def test_int8_rounds_halfway_values_to_even(runtime, group):
    torch = runtime
    # RMS(ones) with epsilon=0 leaves the weights unchanged. Each group has
    # absmax=127, giving scale=1 and exactly representable halfway values.
    x = torch.ones(1, 128, device="cuda", dtype=torch.float32)
    weight = torch.tensor(
        [-127, -126.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 126.5, 127] + [0] * 54,
        device="cuda",
        dtype=x.dtype,
    ).repeat(2)
    out = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(1, 128 // (group or 128), device="cuda")
    if group is None:
        torch.ops._C.rms_norm_dynamic_per_token_quant(
            out, x, weight, scales, 0.0, None, None
        )
    else:
        torch.ops._C.rms_norm_per_block_quant(
            out, x, weight, scales, 0.0, None, None, group, False
        )
    torch.testing.assert_close(scales, torch.ones_like(scales), rtol=0, atol=0)
    expected = torch.tensor(
        [-127, -126, -2, -2, 0, 0, 2, 2, 126, 127] + [0] * 54,
        dtype=torch.int8,
    ).repeat(2)
    torch.testing.assert_close(out.cpu()[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("cache_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("neox", [False, True])
@pytest.mark.parametrize(
    "head,rotary,nq,nk,nv,forced,tokens",
    [
        (64, 64, 8, 2, 2, -1, 1),
        (128, 64, 5, 2, 1, 1, 5),
        (256, 64, 5, 3, 2, 2, 5),
        (128, 32, 5, 2, 2, 4, 5),
        (256, 256, 7, 2, 1, 8, 257),
    ],
)
def test_qk_norm_rope(
    runtime, dtype_name, cache_name, neox, head, rotary, nq, nk, nv, forced, tokens
):
    torch = runtime
    torch.manual_seed(41)
    dtype = getattr(torch, dtype_name)
    qkv = torch.randn(tokens, (nq + nk + nv) * head, device="cuda", dtype=dtype)
    before = qkv.clone()
    qw, kw = (torch.randn(head, device="cuda", dtype=dtype) for _ in range(2))
    phase = torch.randn(1024, rotary // 2, device="cuda")
    cache = torch.cat((phase.cos(), phase.sin()), -1).to(getattr(torch, cache_name))
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64) * 3

    def call():
        torch.ops._C.fused_qk_norm_rope(
            qkv, nq, nk, nv, head, 1e-6, qw, kw, cache, neox, positions, forced
        )

    _run(torch, call, lambda: qkv.copy_(before), graph=forced == 8)
    expected = before.cpu().view(tokens, nq + nk + nv, head).float()
    cos, sin = cache.cpu().float()[positions.cpu()].chunk(2, -1)
    for start, count, weight in ((0, nq, qw), (nq, nk, kw)):
        values = expected[:, start : start + count]
        values = values * (
            torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
            * weight.cpu().float()
        )
        left = slice(0, rotary // 2) if neox else slice(0, rotary, 2)
        right = slice(rotary // 2, rotary) if neox else slice(1, rotary, 2)
        a, b = values[..., left].clone(), values[..., right].clone()
        values[..., left] = a * cos[:, None] - b * sin[:, None]
        values[..., right] = b * cos[:, None] + a * sin[:, None]
        expected[:, start : start + count] = values
    tolerance = 0.01 if dtype == torch.bfloat16 else 0.002
    torch.testing.assert_close(
        qkv.cpu(), expected.reshape_as(qkv).to(dtype), rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        qkv[:, (nq + nk) * head :], before[:, (nq + nk) * head :], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "name",
    [
        "rms_norm_static_fp8_quant",
        "fused_add_rms_norm_static_fp8_quant",
        "rms_norm_dynamic_per_token_quant",
        "rms_norm_per_block_quant",
        "silu_and_mul_per_block_quant",
        "fused_qk_norm_rope",
    ],
)
def test_empty_batches(runtime, name):
    torch = runtime
    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    weight = torch.ones(128, device="cuda", dtype=x.dtype)
    scale = torch.ones(1, device="cuda")
    if name == "rms_norm_static_fp8_quant":
        args = (out, x, weight, scale, 1e-6)
    elif name == "fused_add_rms_norm_static_fp8_quant":
        args = (out, x, x.clone(), weight, scale, 1e-6)
    elif name == "rms_norm_dynamic_per_token_quant":
        args = (out, x, weight, scale[:0], 1e-6, None, None)
    elif name == "rms_norm_per_block_quant":
        args = (
            out,
            x,
            weight,
            torch.empty(0, 2, device="cuda"),
            1e-6,
            None,
            None,
            64,
            False,
        )
    elif name == "silu_and_mul_per_block_quant":
        args = (
            out,
            torch.empty(0, 256, device="cuda", dtype=x.dtype),
            torch.empty(0, 2, device="cuda"),
            64,
            None,
            False,
        )
    else:
        args = (
            torch.empty(0, 384, device="cuda", dtype=x.dtype),
            1,
            1,
            1,
            128,
            1e-6,
            weight,
            weight,
            torch.ones(1, 128, device="cuda", dtype=x.dtype),
            True,
            torch.empty(0, device="cuda", dtype=torch.int64),
        )
    getattr(torch.ops._C, name)(*args)
    torch.cuda.synchronize()
