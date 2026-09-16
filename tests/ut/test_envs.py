# SPDX-License-Identifier: Apache-2.0
"""Tests for lazy SAIL environment parsing and PPU compatibility."""

from __future__ import annotations

import pytest

from vllm_sail import envs

BOOL_DEFAULTS = {
    "FUSED_RMSNORM_QUANT": False,
    "USE_OPT_TOKEN_GROUP_QUANT": False,
    "DENSE_BF16_DEEPGEMM": False,
    "USE_PLA": True,
    "DEEPGEMM_MOE_TP_FUSED": True,
    "FUSED_GDN_DECODE": True,
    "DISABLE_MOE_WNA16_CUDA": False,
    "FORCE_MOE_WNA16_CUDA": False,
    "ENABLE_MOE_MARLIN": False,
    "USE_TRITON_INT8_QUANT": True,
    "NVTX_PROFILE": False,
    "NVTX_DUMP_TOPK": False,
    "NVTX_VFA_DUMP_SEQLEN": False,
}
BACKENDS = ("MOE_BACKEND", "DENSE_BACKEND")


@pytest.fixture(autouse=True)
def clean_plugin_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in dir(envs):
        monkeypatch.delenv(name, raising=False)


def test_defaults_and_canonical_snapshot() -> None:
    expected = {f"VLLM_SAIL_{key}": value for key, value in BOOL_DEFAULTS.items()}
    expected.update({f"VLLM_SAIL_{key}": None for key in BACKENDS})
    expected["VLLM_DEEPEPLL_RECV_HOOK"] = True
    assert envs.snapshot() == expected
    assert set(envs.environment_variables) == set(expected)
    for suffix in (*BOOL_DEFAULTS, *BACKENDS):
        assert getattr(envs, f"VLLM_PPU_{suffix}") == expected[f"VLLM_SAIL_{suffix}"]


@pytest.mark.parametrize("suffix", BOOL_DEFAULTS)
def test_boolean_aliases_are_lazy_and_sail_takes_precedence(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    sail, ppu = f"VLLM_SAIL_{suffix}", f"VLLM_PPU_{suffix}"
    for raw, expected in (("1", True), ("0", False)):
        monkeypatch.setenv(ppu, raw)
        assert getattr(envs, sail) is expected
        assert getattr(envs, ppu) is expected
    monkeypatch.setenv(ppu, "1")
    for raw in ("0", "false", ""):
        monkeypatch.setenv(sail, raw)
        assert getattr(envs, sail) is False
        assert getattr(envs, ppu) is False
        assert envs.snapshot()[sail] is False
    monkeypatch.setenv(ppu, "0")
    monkeypatch.setenv(sail, "1")
    assert getattr(envs, sail) is True
    assert getattr(envs, ppu) is True
    monkeypatch.delenv(sail)
    assert getattr(envs, sail) is False


@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_truthy_parsing(monkeypatch: pytest.MonkeyPatch, prefix: str, raw: str) -> None:
    monkeypatch.setenv(f"{prefix}_NVTX_PROFILE", raw)
    assert envs.VLLM_SAIL_NVTX_PROFILE is True


@pytest.mark.parametrize("suffix", BACKENDS)
def test_backend_aliases_and_precedence(
    monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    sail, ppu = f"VLLM_SAIL_{suffix}", f"VLLM_PPU_{suffix}"
    for name in (ppu, sail):
        for raw, expected in (
            (" DeepGEMM ", "deepgemm"),
            ("ACEXT", "acext"),
            ("triton", "triton"),
        ):
            monkeypatch.setenv(name, raw)
            assert getattr(envs, sail) == expected
            assert getattr(envs, ppu) == expected
    monkeypatch.setenv(ppu, "deepgemm")
    for raw in ("", " "):
        monkeypatch.setenv(sail, raw)
        assert getattr(envs, sail) is None
        assert getattr(envs, ppu) is None
    monkeypatch.setenv(sail, "invalid")
    with pytest.raises(ValueError, match=sail):
        getattr(envs, ppu)
    monkeypatch.delenv(sail)
    monkeypatch.setenv(ppu, "invalid")
    with pytest.raises(ValueError, match=ppu):
        getattr(envs, sail)
    monkeypatch.setenv(sail, "acext")
    assert getattr(envs, sail) == "acext"


def test_oldest_nvtx_alias_has_lowest_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SAIL_NVTX_PROFILE", "1")
    assert envs.VLLM_SAIL_NVTX_PROFILE is True
    assert envs.is_set("VLLM_SAIL_NVTX_PROFILE") is True
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "0")
    assert envs.VLLM_SAIL_NVTX_PROFILE is False
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "1")
    assert envs.VLLM_PPU_NVTX_PROFILE is True


@pytest.mark.parametrize("suffix", ["USE_PLA", "MOE_BACKEND"])
@pytest.mark.parametrize("prefix", ["VLLM_SAIL", "VLLM_PPU"])
def test_is_set_recognizes_either_name_even_when_empty(
    monkeypatch: pytest.MonkeyPatch, suffix: str, prefix: str
) -> None:
    sail, ppu = f"VLLM_SAIL_{suffix}", f"VLLM_PPU_{suffix}"
    assert envs.is_set(sail) is False
    assert envs.is_set(ppu) is False
    monkeypatch.setenv(f"{prefix}_{suffix}", "")
    assert envs.is_set(sail) is True
    assert envs.is_set(ppu) is True


def test_shared_upstream_variable_keeps_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_DEEPEPLL_RECV_HOOK", "0")
    assert envs.VLLM_DEEPEPLL_RECV_HOOK is False
    assert envs.is_set("VLLM_DEEPEPLL_RECV_HOOK") is True


def test_choice_helper_supports_unregistered_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    getter = envs._choice("TEST_SAIL_CHOICE", None, ("alpha", "beta"))
    monkeypatch.setenv("TEST_SAIL_CHOICE", "BeTa")
    assert getter() == "beta"
    monkeypatch.setenv("TEST_SAIL_CHOICE", "gamma")
    with pytest.raises(ValueError, match="TEST_SAIL_CHOICE"):
        getter()


def test_dir_lists_resolvable_names_and_unknown_attributes_raise() -> None:
    assert set(envs.environment_variables).issubset(dir(envs))
    assert "VLLM_PPU_NVTX_PROFILE" in dir(envs)
    for name in dir(envs):
        getattr(envs, name)
    with pytest.raises(AttributeError, match="VLLM_SAIL_UNKNOWN"):
        _ = envs.VLLM_SAIL_UNKNOWN
