# SPDX-License-Identifier: Apache-2.0
"""Locate and package the native runtime metadata.

The source tree keeps the kernel manifest beside the generated corpus because
the port tool owns both.  Installed wheels do not contain ``csrc/``, so the
build copies the two runtime TOMLs into ``vllm_sail/native/data``.  Consumers use
this module to prefer that installed copy while retaining a source-tree
fallback for editable installs and repository tooling.

This module is deliberately stdlib-only: ``setup.py`` loads it without
importing :mod:`vllm_sail`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

__all__ = [
    "RUNTIME_METADATA",
    "build_outputs",
    "copy_to_build",
    "resolve",
    "source_path",
]

RUNTIME_METADATA = ("manifest.toml", "excluded_ops.toml")
_PACKAGE_DATA = Path("vllm_sail/native/data")


def source_path(repo_root: str | Path, name: str) -> Path:
    """Return one canonical metadata path in a repository or source archive."""
    _validate_name(name)
    return Path(repo_root) / "csrc" / "upstream" / name


def resolve(native_dir: str | Path, name: str) -> Path:
    """Resolve metadata from an installed package, then from the source tree."""
    _validate_name(name)
    native_dir = Path(native_dir)
    packaged = native_dir / "data" / name
    if packaged.is_file():
        return packaged
    return source_path(native_dir.parents[1], name)


def copy_to_build(repo_root: str | Path, build_lib: str | Path) -> tuple[Path, ...]:
    """Copy canonical metadata into a wheel's package tree."""
    repo_root = Path(repo_root)
    destination = Path(build_lib) / _PACKAGE_DATA
    destination.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for name in RUNTIME_METADATA:
        source = source_path(repo_root, name)
        if not source.is_file():
            raise FileNotFoundError(f"native runtime metadata is missing: {source}")
        target = destination / name
        shutil.copy2(source, target)
        copied.append(target)
    return tuple(copied)


def build_outputs(build_lib: str | Path) -> tuple[Path, ...]:
    """Paths added to ``build_py`` outputs by :func:`copy_to_build`."""
    root = Path(build_lib) / _PACKAGE_DATA
    return tuple(root / name for name in RUNTIME_METADATA)


def _validate_name(name: str) -> None:
    if name not in RUNTIME_METADATA:
        raise ValueError(f"unknown native runtime metadata: {name!r}")
