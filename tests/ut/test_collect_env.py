# SPDX-License-Identifier: Apache-2.0
"""Tests for lifecycle-critical fields in the environment report."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

collect_env = importlib.import_module("vllm_sail.collect_env")


def test_report_includes_global_plugin_controls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VLLM_PLUGINS", "ppu")
    monkeypatch.setenv("VLLM_VERSION", "0.27.0")
    monkeypatch.setenv("VLLM_PPU_NVTX_PROFILE", "1")
    monkeypatch.setenv("VLLM_SAIL_NVTX_PROFILE", "0")
    monkeypatch.delenv("VLLM_SAIL_NVTX_DUMP_TOPK", raising=False)
    monkeypatch.setenv("VLLM_PPU_NVTX_DUMP_TOPK", "1")
    monkeypatch.setattr(collect_env, "_smi_report", lambda: ("ppu-smi", "ok"))
    monkeypatch.setattr(collect_env, "_torch_report", lambda _driver: [])
    monkeypatch.setattr(collect_env, "_module_version", lambda _name: "stub")

    collect_env.main()

    output = capsys.readouterr().out
    assert "VLLM_PLUGINS: ppu" in output
    assert "VLLM_VERSION: 0.27.0" in output
    assert "VLLM_SAIL_NVTX_PROFILE=False" in output
    assert "VLLM_SAIL_NVTX_DUMP_TOPK=True" in output
    assert "VLLM_PPU_NVTX" not in output


@pytest.mark.parametrize("available", [True, False, RuntimeError("driver unavailable")])
def test_torch_report_uses_one_failure_tolerant_availability_snapshot(
    monkeypatch, available
):
    probe = Mock()
    if isinstance(available, Exception):
        probe.side_effect = available
    else:
        probe.return_value = available
    device = Mock(return_value="PPU-ZW810E")
    capability = Mock(return_value=(8, 0))
    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="test"),
        cuda=SimpleNamespace(
            is_available=probe,
            get_device_name=device,
            get_device_capability=capability,
        ),
    )
    monkeypatch.setattr(collect_env.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(collect_env.importlib, "import_module", lambda name: torch)

    report = dict(collect_env._torch_report("driver"))

    probe.assert_called_once_with()
    if available is True:
        assert report["CUDA device"] == "PPU-ZW810E"
        assert report["CUDA capability"] == "(8, 0)"
        assert report["CUDA driver"] == "driver"
    else:
        device.assert_not_called()
        capability.assert_not_called()
        assert "CUDA device" not in report
    assert report["CUDA available"] == (
        "unavailable (RuntimeError: driver unavailable)"
        if isinstance(available, Exception)
        else str(available)
    )
