# SPDX-License-Identifier: Apache-2.0
"""Import-safe PPU FA API; load torch and SDK binaries only when used.

Platform discovery installs this module before a platform exists. Importing an
SDK wheel there can recursively resolve the platform, so kernel imports belong
inside these calls. Both upstream FA import paths resolve to this API.
"""

from __future__ import annotations

import importlib

compile_flash_attn_varlen_func_from_specs = None  # FA4 is CUDA-only.

__all__ = [
    "compile_flash_attn_varlen_func_from_specs",
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
    "sparse_attn_func",
    "sparse_attn_varlen_func",
]

_AVAILABILITY_EXPORTS = (
    "FA2_AVAILABLE",
    "FA3_AVAILABLE",
    "FA2_UNAVAILABLE_REASON",
    "FA3_UNAVAILABLE_REASON",
)


def _kernels():
    try:
        return importlib.import_module("vllm_sail.attention.flash_attn._kernels")
    except (ImportError, OSError) as exc:
        raise ImportError(
            f"Cannot load the PPU FlashAttention interface: {exc}. "
            "Install compatible PPU torch and the FlashAttention wheels shipped "
            "with your PPU SDK; check their shared-library dependencies."
        ) from exc


def __getattr__(name: str):
    if name in _AVAILABILITY_EXPORTS:
        return getattr(_kernels(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_fa_version_supported(fa_version: int, device=None) -> bool:
    if fa_version == 4:
        return False
    return _kernels().is_fa_version_supported(fa_version, device)


def fa_version_unsupported_reason(fa_version: int, device=None) -> str | None:
    if fa_version == 4:
        return "SAIL provides FlashAttention 2 and 3; FA4 requires NVIDIA CuteDSL."
    return _kernels().fa_version_unsupported_reason(fa_version, device)


def flash_attn_varlen_func(*args, **kwargs):
    return _kernels().flash_attn_varlen_func(*args, **kwargs)


def get_scheduler_metadata(*args, **kwargs):
    return _kernels().get_scheduler_metadata(*args, **kwargs)


def sparse_attn_func(*args, **kwargs):
    return _kernels().sparse_attn_func(*args, **kwargs)


def sparse_attn_varlen_func(*args, **kwargs):
    return _kernels().sparse_attn_varlen_func(*args, **kwargs)
