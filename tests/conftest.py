# SPDX-License-Identifier: Apache-2.0
"""Shared test harness with two explicit tiers.

``tests/ut`` is CPU-only and must pass anywhere, including environments without
vLLM, torch, or an accelerator. ``tests/e2e`` requires a real PPU and is gated
by the ``ppu`` marker. SDK fakes are opt-in fixtures so no module-level stub can
leak into another test and hide an unexpected dependency.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import pytest


class DeviceCapability(NamedTuple):
    """Small CUDA-compatible capability value used by platform unit tests."""

    major: int
    minor: int


class FakePPUPlatform:
    """Minimal platform double exposing only the contract tests may rely on."""

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        del device_id
        return "PPU-ZW810E"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        del device_id
        return DeviceCapability(8, 0)

    @classmethod
    def is_ppu(cls) -> bool:
        return True

    @classmethod
    def is_cuda(cls) -> bool:
        return True


class StrictStubModule(types.ModuleType):
    """A real module that rejects every attribute not explicitly provided."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(
            f"PPU SDK test stub {self.__name__!r} has no attribute {name!r}; "
            "declare the test-facing API explicitly instead of mocking it."
        )


def _stub_module(name: str, *, package: bool = False) -> StrictStubModule:
    module = StrictStubModule(name)
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    module.__package__ = name.rpartition(".")[0]
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def _real_device_available() -> bool:
    """Probe CUDA/PPU availability without making torch a test dependency."""
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if bool(torch.cuda.is_available()):
                return True
        except (AttributeError, ImportError, OSError, RuntimeError):
            pass
    try:
        probe = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and bool(probe.stdout.strip())


HAS_REAL_DEVICE = _real_device_available()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "ppu: requires a real PPU/CUDA device")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if HAS_REAL_DEVICE:
        return
    skip = pytest.mark.skip(reason="requires a real PPU/CUDA device")
    for item in items:
        if "ppu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fake_ppu_platform() -> FakePPUPlatform:
    return FakePPUPlatform()


@pytest.fixture
def ppu_sdk_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, StrictStubModule]:
    """Install strict SDK package stubs for one requesting test only."""
    stubs = {
        name: _stub_module(name, package=name in {"flash_attn", "flash_attn_3"})
        for name in (
            "acext",
            "deep_gemm",
            "tokenspeed_mla",
            "flash_attn",
            "flash_attn_3",
        )
    }
    flash_attn_3_c = _stub_module("flash_attn_3._C")
    stubs["flash_attn_3._C"] = flash_attn_3_c
    stubs["flash_attn_3"]._C = flash_attn_3_c  # type: ignore[attr-defined]
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    return stubs


@pytest.fixture
def patch_utils_module() -> Iterator[types.ModuleType]:
    """Load patch/utils.py without executing the runtime patch package init."""
    name = f"_vllm_sail_patch_utils_test_{uuid4().hex}"
    path = Path(__file__).parents[1] / "vllm_sail" / "patch" / "utils.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)
