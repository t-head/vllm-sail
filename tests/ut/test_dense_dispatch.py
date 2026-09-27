# SPDX-License-Identifier: Apache-2.0
"""PPU dense dispatch and its upstream linear-constructor boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.support.source import assert_accepts_upstream_keywords, function


@pytest.mark.parametrize("is_ppu", [False, True])
@pytest.mark.parametrize(
    "args,kwargs,expected_backend",
    [
        ((), {}, "auto"),
        (("auto",), {}, "auto"),
        (("torch",), {}, "torch"),
        ((), {"linear_backend": "flashinfer_cublas"}, "flashinfer_cublas"),
    ],
)
def test_dense_dispatch_accepts_and_preserves_linear_backend(
    is_ppu, args, kwargs, expected_backend
):
    ppu_gemm, upstream_gemm = object(), object()
    calls = []

    def original(linear_backend="auto"):
        calls.append(linear_backend)
        return upstream_gemm

    dispatch = function(
        "vllm_sail/patch/enhancement/dense.py",
        "dispatch_unquantized_gemm",
        {
            "current_platform": SimpleNamespace(is_ppu=lambda: is_ppu),
            "ppu_unquantized_gemm": ppu_gemm,
            "_original": original,
        },
    )

    assert dispatch(*args, **kwargs) is (ppu_gemm if is_ppu else upstream_gemm)
    assert calls == ([] if is_ppu else [expected_backend])


@pytest.mark.upstream_source
@pytest.mark.parametrize("linear_backend", [None, "auto", "torch"])
def test_upstream_unquantized_linear_init_accepts_ppu_dispatch(
    upstream_source_root, linear_backend
):
    """Run the real upstream constructor that compressed-tensors calls first."""
    ppu_gemm = object()
    namespace = {
        "current_platform": SimpleNamespace(is_ppu=lambda: True),
        "ppu_unquantized_gemm": ppu_gemm,
        "get_current_vllm_config_or_none": lambda: (
            None
            if linear_backend is None
            else SimpleNamespace(
                kernel_config=SimpleNamespace(linear_backend=linear_backend)
            )
        ),
    }
    function(
        "vllm_sail/patch/enhancement/dense.py",
        "dispatch_unquantized_gemm",
        namespace,
    )
    init = function(
        upstream_source_root / "vllm/model_executor/layers/linear.py",
        "UnquantizedLinearMethod.__init__",
        namespace,
    )
    method = SimpleNamespace()
    init(method)
    assert method._gemm_impl is ppu_gemm


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "dense",
            "dispatch_unquantized_gemm",
            "model_executor/layers/utils",
            "dispatch_unquantized_gemm",
        )
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
