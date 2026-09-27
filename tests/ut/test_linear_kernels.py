# SPDX-License-Identifier: Apache-2.0
"""Registration and support-predicate coverage for the PPU dense-GEMM kernels.

Phase-3 exit criterion: every registered kernel's support predicate is covered
by a unit test. The five ``scaled_mm/ppu.py`` kernels register through
``vllm_sail.registry.linear_kernels``; these tests assert they land *first* in
upstream's candidate lists and that their ``is_supported`` / ``can_implement``
predicates behave as specified on and off PPU.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib.metadata import PackageNotFoundError, distribution

import pytest

_KERNEL_CLASSES = [
    "PPUInt8ScaledMMLinearKernel",
    "PPUDeepGemmFP8ScaledMMLinearKernel",
    "PPUCutlassFP8ScaledMMLinearKernel",
    "PPUDeepGemmFp8BlockScaledMMKernel",
    "PPUCutlassFp8BlockScaledMMKernel",
]

_EXPECTED_FIRST = {
    "_POSSIBLE_INT8_KERNELS": ["PPUInt8ScaledMMLinearKernel"],
    "_POSSIBLE_FP8_KERNELS": [
        "PPUDeepGemmFP8ScaledMMLinearKernel",
        "PPUCutlassFP8ScaledMMLinearKernel",
    ],
    "_POSSIBLE_FP8_BLOCK_KERNELS": [
        "PPUDeepGemmFp8BlockScaledMMKernel",
        "PPUCutlassFp8BlockScaledMMKernel",
    ],
}


@pytest.fixture(scope="module")
def kernels():
    """Load the real linear-kernel stack, or skip without vLLM/torch."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")

    from vllm.model_executor.kernels import linear as linear_module

    from vllm_sail.registry import linear_kernels

    linear_kernels.register()
    return linear_module, linear_kernels


def test_registry_package_is_importable_without_vllm() -> None:
    import importlib

    package = importlib.import_module("vllm_sail.registry.linear_kernels")
    assert callable(package.register)


@pytest.mark.parametrize("loader_error", ["OSError", "ImportError"])
def test_registration_precedes_general_patches_and_optional_sdk_loading(
    tmp_path,
    loader_error,
) -> None:
    """Direct registry imports must work in a fresh installed-wheel process."""
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    try:
        distribution("vllm-sail")
    except PackageNotFoundError:
        pytest.skip("requires an installed vllm-sail distribution")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import builtins
                import sys
                import vllm_sail
                from vllm.platforms import current_platform

                assert not vllm_sail._PATCHES_APPLIED
                assert current_platform.is_ppu() is True
                original_import = builtins.__import__
                loader_error = getattr(builtins, sys.argv[1])
                attempted = []
                def guarded_import(name, *args, **kwargs):
                    if name == 'acext' or name.startswith('acext.'):
                        attempted.append(name)
                        raise loader_error('test SDK ABI load failure')
                    return original_import(name, *args, **kwargs)
                builtins.__import__ = guarded_import

                from vllm_sail.registry import linear_kernels
                linear_kernels.register()
                linear_kernels.register()
                assert attempted == []

                from vllm_sail.model_executor.kernels.linear.scaled_mm import ppu
                assert ppu.PPUInt8ScaledMMLinearKernel.is_supported() == (True, None)
                assert attempted == []
                try:
                    ppu._get_acext_int8_gemm()
                except loader_error as exc:
                    assert str(exc) == 'test SDK ABI load failure'
                else:
                    raise AssertionError('SDK ABI failure was hidden')
                assert attempted == ['acext']
            """),
            loader_error,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_register_is_idempotent(kernels) -> None:
    linear_module, linear_kernels = kernels
    from vllm.platforms.interface import PlatformEnum

    snapshot = {
        attribute: list(getattr(linear_module, attribute)[PlatformEnum.CUDA])
        for attribute in _EXPECTED_FIRST
    }
    linear_kernels.register()
    linear_kernels.register()
    for attribute, candidates in snapshot.items():
        assert getattr(linear_module, attribute)[PlatformEnum.CUDA] == candidates


def test_every_ppu_kernel_is_first_in_its_candidate_list(kernels) -> None:
    """register_linear_kernel appends; PPU kernels must win selection instead."""
    linear_module, _ = kernels
    from vllm.platforms.interface import PlatformEnum

    from vllm_sail.model_executor.kernels.linear.scaled_mm import ppu as ppu_kernels

    for attribute, class_names in _EXPECTED_FIRST.items():
        candidates = getattr(linear_module, attribute)[PlatformEnum.CUDA]
        prefix = candidates[: len(class_names)]
        assert [cls.__name__ for cls in prefix] == class_names, attribute
        for cls in prefix:
            assert getattr(ppu_kernels, cls.__name__) is cls


@pytest.mark.parametrize("class_name", _KERNEL_CLASSES)
def test_is_supported_is_false_off_ppu(monkeypatch, kernels, class_name) -> None:
    from vllm.platforms import current_platform

    from vllm_sail.model_executor.kernels.linear.scaled_mm import ppu as ppu_kernels

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: False)
    kernel_cls = getattr(ppu_kernels, class_name)
    supported, reason = kernel_cls.is_supported()
    assert supported is False
    assert reason


@pytest.mark.parametrize(
    ("class_name", "sm80", "deep_gemm", "expected"),
    [
        # Int8 and Cutlass kernels only require the platform itself.
        ("PPUInt8ScaledMMLinearKernel", True, False, True),
        ("PPUCutlassFP8ScaledMMLinearKernel", True, False, True),
        ("PPUCutlassFp8BlockScaledMMKernel", True, False, True),
        # DeepGEMM FP8 kernels are sm90-class only and need the SDK wrapper.
        ("PPUDeepGemmFP8ScaledMMLinearKernel", True, True, False),
        ("PPUDeepGemmFP8ScaledMMLinearKernel", False, False, False),
        ("PPUDeepGemmFP8ScaledMMLinearKernel", False, True, True),
        ("PPUDeepGemmFp8BlockScaledMMKernel", True, True, False),
        ("PPUDeepGemmFp8BlockScaledMMKernel", False, False, False),
        ("PPUDeepGemmFp8BlockScaledMMKernel", False, True, True),
    ],
)
def test_is_supported_on_ppu_matrix(
    monkeypatch, kernels, class_name, sm80, deep_gemm, expected
) -> None:
    from vllm.platforms import current_platform

    from vllm_sail.model_executor.kernels.linear.scaled_mm import ppu as ppu_kernels

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: True)
    monkeypatch.setattr(
        type(current_platform),
        "is_device_capability",
        lambda self, cap: cap == (8, 0) and sm80,
    )
    monkeypatch.setattr(ppu_kernels, "is_deep_gemm_supported", lambda: deep_gemm)
    kernel_cls = getattr(ppu_kernels, class_name)
    supported, reason = kernel_cls.is_supported()
    assert supported is expected, (class_name, sm80, deep_gemm, reason)
    if not expected:
        assert reason


def test_int8_and_cutlass_can_implement_accept_every_config(kernels) -> None:
    from vllm_sail.model_executor.kernels.linear.scaled_mm import ppu as ppu_kernels

    assert ppu_kernels.PPUInt8ScaledMMLinearKernel.can_implement(None) == (True, None)
    assert ppu_kernels.PPUCutlassFP8ScaledMMLinearKernel.can_implement(None) == (
        True,
        None,
    )


@pytest.mark.parametrize(
    "class_name",
    ["PPUCutlassFp8BlockScaledMMKernel", "PPUDeepGemmFp8BlockScaledMMKernel"],
)
def test_fp8_block_can_implement_requires_128_group_activations(
    monkeypatch, kernels, class_name
) -> None:
    """The PPU-added gate accepts only dynamic (1,128) activation groups."""
    import torch
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        GroupShape,
        ScaleDesc,
    )

    from vllm_sail.model_executor.kernels.linear.scaled_mm import ppu as ppu_kernels

    kernel_cls = getattr(ppu_kernels, class_name)
    base = kernel_cls.__mro__[1]
    monkeypatch.setattr(
        base, "can_implement", classmethod(lambda cls, config: (True, None))
    )

    def _config(group_shape):
        config = type("C", (), {})()
        config.activation_quant_key = type(
            "K", (), {"scale": ScaleDesc(torch.float32, False, group_shape)}
        )
        config.out_dtype = torch.bfloat16
        config.weight_shape = (4096, 4096)
        return config

    supported, reason = kernel_cls.can_implement(_config(GroupShape(1, 64)))
    assert supported is False
    assert "group_shape=(1,128)" in reason

    if class_name == "PPUDeepGemmFp8BlockScaledMMKernel":
        # The DeepGEMM variant additionally needs bf16 output and a live model
        # config; stop at the group-shape gate for the non-128 case and accept
        # that the 128 case proceeds to gates that need a real vLLM config.
        return
    supported, reason = kernel_cls.can_implement(_config(GroupShape(1, 128)))
    assert supported is True
