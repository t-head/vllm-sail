# SPDX-License-Identifier: Apache-2.0
"""Exercise native loading and preflight without torch or a PPU SDK."""

from __future__ import annotations

import sys
import types

import pytest

from vllm_sail.native import extensions


@pytest.fixture
def loader(monkeypatch):
    torch = types.ModuleType("torch")
    torch.ops = types.SimpleNamespace(
        _C=types.SimpleNamespace(), _moe_C=types.SimpleNamespace()
    )
    kernels = set()
    torch._C = types.SimpleNamespace(
        _dispatch_has_kernel_for_dispatch_key=lambda name, key: (
            key == "CUDA" and name in kernels
        )
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(extensions, "_imported", set())
    calls = []

    def load(name):
        calls.append(name)
        if name == "vllm_sail._upstream_C":
            torch.ops._C.rms_norm = object()
            torch.ops._C.silu_and_mul = object()
            kernels.update({"_C::rms_norm", "_C::silu_and_mul"})
        elif name == "vllm_sail._upstream_moe_C":
            torch.ops._moe_C.moe_sum = object()
            kernels.add("_moe_C::moe_sum")
        else:
            raise AssertionError(name)
        return types.ModuleType(name)

    return torch, kernels, calls, load


def test_failed_load_can_be_retried_before_silu_lookup(loader, monkeypatch):
    torch, _, calls, load = loader

    def unavailable(name):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    monkeypatch.setattr(extensions.importlib, "import_module", unavailable)
    assert extensions.import_kernels() == ()
    monkeypatch.setattr(extensions.importlib, "import_module", load)
    extensions.import_kernels()
    assert torch.ops._C.silu_and_mul is not None
    assert len(calls) == 2


def test_strict_load_is_idempotent(loader, monkeypatch):
    _, _, calls, load = loader
    monkeypatch.setattr(extensions.importlib, "import_module", load)
    assert extensions.import_kernels(strict=True) == (
        "vllm_sail._upstream_C",
        "vllm_sail._upstream_moe_C",
    )
    assert extensions.import_kernels(strict=True) == ()
    assert len(calls) == 2


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'vllm_sail._upstream_C'"),
        ImportError("undefined symbol: sail_symbol"),
        OSError("libhggc.so: cannot open shared object file"),
    ],
)
def test_strict_load_reports_original_error_and_build_recipe(
    loader, monkeypatch, error
):
    def broken(name):
        raise error

    monkeypatch.setattr(extensions.importlib, "import_module", broken)
    with pytest.raises(RuntimeError) as caught:
        extensions.import_kernels(strict=True)
    assert caught.value.__cause__ is error
    assert str(error) in str(caught.value)
    assert "_upstream_C" in str(caught.value)
    assert "--no-build-isolation" in str(caught.value)


def test_strict_load_detects_partially_claimed_namespace(loader, monkeypatch):
    torch, _, calls, load = loader
    torch.ops._C.rms_norm = object()
    monkeypatch.setattr(extensions.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match="silu_and_mul"):
        extensions.import_kernels(strict=True)
    assert calls == []  # importing would duplicate the existing rms_norm schema


def test_strict_load_rejects_schemas_without_device_implementations(
    loader, monkeypatch
):
    torch, _, calls, load = loader
    torch.ops._C.rms_norm = object()
    torch.ops._C.silu_and_mul = object()
    monkeypatch.setattr(extensions.importlib, "import_module", load)
    with pytest.raises(RuntimeError, match="CUDA implementation"):
        extensions.import_kernels(strict=True)
    assert calls == []
