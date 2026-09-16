# SPDX-License-Identifier: Apache-2.0
"""Landing and behaviour tests for the Phase-3 compute-core patches.

Covers the patch modules that wire the compute core into stock vLLM:

* ``deep_gemm_redirect``  — fp8_utils / fp8 / input_quant_fp8 helper rebinding
* ``kernel_warmup``       — PPU DeepGEMM warmup appended to kernel_warmup
* ``quant_keys``          — int8 QuantKey constants for the Acext path
* ``tuned_config_lookup`` — block-quant configs found in the plugin directory
* ``int8_quant``          — Triton int8 quant with rounding + forced-Triton knobs

(The MoE Marlin default-off gate is covered by
``patch/enhancement/models/moe_marlin_gate.py`` and tested in
``test_model_patches.py``.)

Every test asserts the patch *landed* (marker present) before asserting
behaviour, mirroring ``test_patch_install.py``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def installed():
    pytest.importorskip("torch", reason="requires torch")
    pytest.importorskip("vllm", reason="requires vLLM")

    import vllm_sail.patch as patch_pkg

    patch_pkg.install()
    return patch_pkg


def _patched(module_name: str, attribute: str):
    import importlib

    from vllm_sail.patch.utils import PATCH_MARKER

    replacement = getattr(importlib.import_module(module_name), attribute)
    marker = getattr(replacement, PATCH_MARKER, None)
    assert marker is not None, f"{module_name}.{attribute} is not patched"
    return replacement


# ---------------------------------------------------------------------------
# deep_gemm_redirect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "attribute"),
    [
        ("vllm.model_executor.layers.quantization.utils.fp8_utils", "get_tma_aligned_size"),
        ("vllm.model_executor.layers.quantization.utils.fp8_utils", "is_deep_gemm_e8m0_used"),
        ("vllm.model_executor.layers.quantization.utils.fp8_utils", "transform_sf_into_required_layout"),
        ("vllm.model_executor.layers.quantization.fp8", "is_deep_gemm_supported"),
        ("vllm.model_executor.layers.quantization.input_quant_fp8", "DeepGemmQuantScaleFMT"),
        ("vllm.model_executor.layers.quantization.input_quant_fp8", "is_deep_gemm_e8m0_used"),
        ("vllm.model_executor.layers.quantization.input_quant_fp8", "is_deep_gemm_supported"),
    ],
)
def test_deep_gemm_helpers_are_rebound_to_the_ppu_wrapper(
    installed, module_name, attribute
) -> None:
    import vllm_sail.utils.deep_gemm as ppu_deep_gemm
    from vllm_sail.patch.utils import PATCH_MARKER

    target = _patched(module_name, attribute)
    assert target is getattr(ppu_deep_gemm, attribute)
    # The marker must record this exact target, proving the rebinding replaced
    # the module's own binding rather than landing somewhere else.
    assert f"{module_name}.{attribute}" in getattr(target, PATCH_MARKER)


# ---------------------------------------------------------------------------
# kernel_warmup
# ---------------------------------------------------------------------------


def test_kernel_warmup_patch_landed(installed) -> None:
    import vllm_sail.patch.enhancement.kernel_warmup as kw_patch

    if not hasattr(kw_patch, "_upstream_kernel_warmup"):
        pytest.skip("kernel_warmup target module is not importable here")
    _patched("vllm.model_executor.warmup.kernel_warmup", "kernel_warmup")


def test_ppu_deep_gemm_warmup_gate(monkeypatch: pytest.MonkeyPatch, installed) -> None:
    """The fork's gate: skip wins, support required, backends must allow it."""
    import vllm_sail.patch.enhancement.kernel_warmup as kw_patch
    import vllm_sail.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)

    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is False
    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: False)
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is False
    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)

    assert kw_patch._ppu_deep_gemm_warmup_enabled() is True

    monkeypatch.setenv("VLLM_PPU_MOE_BACKEND", "deepgemm")
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is True

    monkeypatch.setenv("VLLM_PPU_MOE_BACKEND", "acext")
    monkeypatch.setenv("VLLM_PPU_DENSE_BACKEND", "acext")
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is False

    # Either selector allowing DeepGEMM is enough.
    monkeypatch.setenv("VLLM_PPU_DENSE_BACKEND", "deepgemm")
    assert kw_patch._ppu_deep_gemm_warmup_enabled() is True


def test_kernel_warmup_delegates_off_ppu_and_runs_ppu_warmup_on_ppu(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    import sys
    import types

    import vllm_sail.patch.enhancement.kernel_warmup as kw_patch

    if not hasattr(kw_patch, "_upstream_kernel_warmup"):
        pytest.skip("kernel_warmup target module is not importable here")

    calls: list[str] = []
    monkeypatch.setattr(
        kw_patch, "_upstream_kernel_warmup", lambda worker, **kwargs: calls.append("upstream")
    )
    worker = types.SimpleNamespace(
        get_model=lambda: calls.append("get_model") or object(),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=512),
    )

    from vllm.platforms import current_platform

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: False)
    kw_patch.kernel_warmup(worker)
    assert calls == ["upstream"]

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: True)
    monkeypatch.setenv("VLLM_DEEP_GEMM_WARMUP", "skip")
    calls.clear()
    kw_patch.kernel_warmup(worker)
    assert calls == ["upstream"]

    monkeypatch.delenv("VLLM_DEEP_GEMM_WARMUP", raising=False)
    monkeypatch.delenv("VLLM_PPU_MOE_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_PPU_DENSE_BACKEND", raising=False)
    import vllm_sail.utils.deep_gemm as deep_gemm

    monkeypatch.setattr(deep_gemm, "is_deep_gemm_supported", lambda: True)

    warmup_module = types.ModuleType("vllm_sail.model_executor.warmup.deep_gemm_warmup")
    warmup_module.deep_gemm_warmup = lambda model, max_tokens: calls.append(
        ("ppu_warmup", max_tokens)
    )
    monkeypatch.setitem(
        sys.modules, "vllm_sail.model_executor.warmup.deep_gemm_warmup", warmup_module
    )
    calls.clear()
    kw_patch.kernel_warmup(worker)
    assert calls == ["upstream", "get_model", ("ppu_warmup", 512)]

    # process_local_only skips everything after the upstream body.
    calls.clear()
    kw_patch.kernel_warmup(worker, process_local_only=True)
    assert calls == ["upstream"]


# ---------------------------------------------------------------------------
# quant_keys
# ---------------------------------------------------------------------------


def test_int8_quant_keys_constants_exist(installed) -> None:
    import torch
    from vllm.model_executor.layers.quantization.utils import quant_utils

    assert quant_utils.kStaticChannelScale.group_shape is quant_utils.GroupShape.PER_CHANNEL
    assert quant_utils.kDynamicTokenScale.group_shape is quant_utils.GroupShape.PER_TOKEN
    assert quant_utils.kInt8StaticChannelSym == quant_utils.QuantKey(
        torch.int8, quant_utils.kStaticChannelScale, symmetric=True
    )
    assert quant_utils.kInt8DynamicTokenSym == quant_utils.QuantKey(
        torch.int8, quant_utils.kDynamicTokenScale, symmetric=True
    )


# ---------------------------------------------------------------------------
# tuned_config_lookup
# ---------------------------------------------------------------------------


def test_block_fp8_configs_upstream_wins(monkeypatch: pytest.MonkeyPatch, installed) -> None:
    import vllm_sail.patch.enhancement.tuned_config_lookup as lookup

    sentinel = {1: {"BLOCK_SIZE_M": 16}}
    monkeypatch.setattr(lookup, "_upstream_fp8", lambda *args: sentinel)
    assert lookup.get_w8a8_block_fp8_configs(1536, 7168, 128, 128) is sentinel


def test_block_fp8_configs_fall_back_to_plugin_directory(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    """Pins the duplicated filename format against a real shipped config."""
    import vllm_sail.patch.enhancement.tuned_config_lookup as lookup

    monkeypatch.setattr(lookup, "_upstream_fp8", lambda *args: None)
    monkeypatch.setattr(lookup, "get_device_name_as_file_name", lambda: "ZW-M890P")
    config = lookup.get_w8a8_block_fp8_configs(1536, 7168, 128, 128)
    assert config is not None
    assert all(isinstance(key, int) for key in config)


def test_block_config_filename_formats_differ_by_space(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    """fp8 uses block_shape=[n,k]; int8 uses block_shape=[n, k]. Ugly but real."""
    import vllm_sail.patch.enhancement.tuned_config_lookup as lookup

    seen: list[str] = []

    def _record(json_file_name: str, kind: str):
        seen.append(json_file_name)
        return None

    monkeypatch.setattr(lookup, "_upstream_fp8", lambda *args: None)
    monkeypatch.setattr(lookup, "_upstream_int8", lambda *args: None)
    monkeypatch.setattr(lookup, "_load_plugin_config", _record)
    monkeypatch.setattr(lookup, "get_device_name_as_file_name", lambda: "PPU-ZW810E")

    lookup.get_w8a8_block_fp8_configs(1024, 2048, 128, 128)
    lookup.get_w8a8_block_int8_configs(1024, 2048, 128, 128)
    assert seen[0] == (
        "N=1024,K=2048,device_name=PPU-ZW810E,dtype=fp8_w8a8,"
        "block_shape=[128,128].json"
    )
    assert seen[1] == (
        "N=1024,K=2048,device_name=PPU-ZW810E,dtype=int8_w8a8,"
        "block_shape=[128, 128].json"
    )


# ---------------------------------------------------------------------------
# int8_quant
# ---------------------------------------------------------------------------


def test_int8_quant_wrapper_gained_ppu_knobs(installed) -> None:
    import inspect

    wrapper = _patched(
        "vllm.model_executor.layers.quantization.utils.int8_utils",
        "per_token_group_quant_int8",
    )
    params = inspect.signature(wrapper).parameters
    assert "use_triton" in params
    assert "use_rounding" in params
    assert params["use_triton"].default is False
    assert params["use_rounding"].default is False


# ---------------------------------------------------------------------------
# modular_kernel (packed-MXFP4 K fix)
# ---------------------------------------------------------------------------


def test_moe_problem_size_doubles_k_for_packed_mxfp4_on_ppu(
    monkeypatch: pytest.MonkeyPatch, installed
) -> None:
    import torch
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk_module
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsModular,
    )
    from vllm.platforms import current_platform

    from vllm_sail.patch.utils import PATCH_MARKER

    marker = getattr(FusedMoEExpertsModular.moe_problem_size, PATCH_MARKER, None)
    assert marker is not None, "FusedMoEExpertsModular.moe_problem_size not patched"
    target = f"{mk_module.__name__}.FusedMoEExpertsModular.moe_problem_size"
    assert target in marker

    E, M, N = 2, 4, 32
    packed_k = 64
    w1 = torch.zeros((E, N, packed_k), dtype=torch.uint8)
    w2 = torch.zeros((E, 2 * packed_k, N), dtype=torch.uint8)
    topk_ids = torch.zeros((M, 2), dtype=torch.int64)

    def _solve(a1):
        e, m, n, k, topk = FusedMoEExpertsModular.moe_problem_size(
            None, a1, w1, w2, topk_ids
        )
        return (e, m, n, k, topk)

    mxfp4_act = torch.zeros((M, packed_k), dtype=torch.uint8)
    bf16_act = torch.zeros((M, 2 * packed_k), dtype=torch.bfloat16)

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: True)
    assert _solve(mxfp4_act) == (E, M, N, 2 * packed_k, 2)
    assert _solve(bf16_act) == (E, M, N, 2 * packed_k, 2)

    monkeypatch.setattr(type(current_platform), "is_ppu", lambda self: False)
    assert _solve(mxfp4_act) == (E, M, N, packed_k, 2)
