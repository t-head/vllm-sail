# SPDX-License-Identifier: Apache-2.0
"""Print a compact, failure-tolerant environment report for bug reports."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable

from vllm_sail import envs
from vllm_sail.version import __version__

SDK_PACKAGES = (
    "acext",
    "deep_gemm",
    "tokenspeed_mla",
    "flash_attn",
    "flash_attn_3",
)


def _safe(operation: Callable[[], object]) -> str:
    """Return a printable value even when an optional runtime API is broken."""
    try:
        value = operation()
    except Exception as exc:  # environment diagnostics must never mask the report
        return f"unavailable ({type(exc).__name__}: {exc})"
    return str(value)


def _module_version(name: str) -> str:
    if importlib.util.find_spec(name) is None:
        return "absent"
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return f"present but import failed ({type(exc).__name__}: {exc})"
    return f"present ({getattr(module, '__version__', 'version unknown')})"


def _torch_report(driver: str) -> list[tuple[str, str]]:
    if importlib.util.find_spec("torch") is None:
        return [("torch", "absent")]
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return [("torch", f"present but import failed ({type(exc).__name__}: {exc})")]

    cuda_available = _safe(lambda: bool(torch.cuda.is_available()))
    rows = [
        ("torch", str(getattr(torch, "__version__", "version unknown"))),
        (
            "torch CUDA build",
            str(getattr(getattr(torch, "version", None), "cuda", None)),
        ),
        ("CUDA available", cuda_available),
    ]
    if cuda_available == "True":
        rows.extend(
            [
                (
                    "CUDA device",
                    _safe(lambda: torch.cuda.get_device_name(0)),
                ),
                (
                    "CUDA capability",
                    _safe(lambda: torch.cuda.get_device_capability(0)),
                ),
                ("CUDA driver", driver),
            ]
        )
    return rows


def _smi_report() -> tuple[str, str]:
    """(tool name, output) from ppu-smi, falling back to nvidia-smi.

    Mirrors in-tree vLLM collect_env, which shells out to the SMI tool
    instead of touching torch._C private bindings (absent in PPU torch).
    """
    for tool in ("ppu-smi", "nvidia-smi"):
        executable = shutil.which(tool)
        if executable is None:
            continue
        try:
            result = subprocess.run(
                [executable],
                capture_output=True,
                check=False,
                text=True,
                timeout=15,
            )
        except Exception as exc:
            return tool, f"failed ({type(exc).__name__}: {exc})"
        output = result.stdout.strip() or result.stderr.strip()
        if result.returncode == 0:
            return tool, output
        return tool, f"failed ({result.returncode}): {output}"
    return "smi", "absent (neither ppu-smi nor nvidia-smi found)"


def _driver_version(smi_output: str) -> str:
    match = re.search(r"Driver Version:\s*([0-9][0-9.]*)", smi_output)
    if match:
        return match.group(1)
    return "unavailable (no Driver Version line in SMI output)"


def _env_snapshot() -> dict[str, str]:
    return {name: _safe(getter) for name, getter in envs.environment_variables.items()}


def main() -> None:
    """Print all useful state while tolerating every optional dependency."""
    smi_tool, smi_output = _smi_report()
    rows = [
        ("vllm-sail", __version__),
        ("vLLM", _module_version("vllm")),
        ("Python", sys.version.replace("\n", " ")),
        ("Platform", platform.platform()),
        ("PPU_SDK", os.getenv("PPU_SDK", "unset")),
        ("VLLM_PLUGINS", os.getenv("VLLM_PLUGINS", "unset")),
        ("VLLM_VERSION", os.getenv("VLLM_VERSION", "unset")),
        *_torch_report(_driver_version(smi_output)),
    ]
    print("vLLM SAIL environment report")
    print("============================")
    for label, value in rows:
        print(f"{label}: {value}")

    print("\nSAIL SDK Python packages")
    print("------------------------")
    for name in SDK_PACKAGES:
        print(f"{name}: {_module_version(name)}")

    print(f"\n{smi_tool}")
    print("-" * len(smi_tool))
    print(smi_output)

    print("\nvllm_sail.envs.snapshot()")
    print("-------------------------")
    for name, value in _env_snapshot().items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
