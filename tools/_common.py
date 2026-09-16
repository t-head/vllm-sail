# SPDX-License-Identifier: Apache-2.0
"""Shared maintenance checks; importing this module needs only the stdlib."""

from __future__ import annotations

import filecmp
import importlib.util
from pathlib import Path
from typing import Any


class PatchEnvironmentError(RuntimeError):
    """The real vLLM environment needed by patch maintenance is unavailable."""


def load_patch_records(command: str) -> list[Any]:
    """Load live patch metadata, retaining the calling command's diagnostics."""
    missing = [
        name for name in ("torch", "vllm") if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise PatchEnvironmentError(
            f"{command} requires a real vLLM environment; missing "
            f"{', '.join(missing)}. Install vLLM (and its torch dependency) "
            "before running this tool."
        )

    try:
        import vllm_sail

        vllm_sail.register_out_of_tree()
        from vllm_sail.patch import PATCH_REGISTRY
    except (ImportError, OSError) as exc:
        raise PatchEnvironmentError(
            "the installed vLLM/torch/PPU environment could not load all "
            f"plugin patches: {exc}"
        ) from exc
    return list(PATCH_REGISTRY)


def corpus_differences(
    fresh: Path, committed: Path, *, ignore: frozenset[str] = frozenset()
) -> list[str]:
    """Compare generated files by content, ignoring the named input files."""
    if not committed.is_dir():
        return [f"{committed} does not exist"]

    def files(root: Path) -> set[str]:
        return {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.name not in ignore
        }

    left = files(fresh)
    right = files(committed)
    differences = [f"only in a fresh port: {name}" for name in sorted(left - right)]
    differences += [f"only in the tree: {name}" for name in sorted(right - left)]
    differences += [
        f"differs: {name}"
        for name in sorted(left & right)
        if not filecmp.cmp(fresh / name, committed / name, shallow=False)
    ]
    return differences
