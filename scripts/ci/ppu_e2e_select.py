#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the ``e2e-ppu`` job matrix from the checked-in model catalog.

The catalog (``scripts/ci/ppu_e2e_models.json``) lists every model the PPU E2E
smoke workflow knows how to run. This helper turns it into a GitHub Actions
``strategy.matrix`` payload, optionally narrowed to a caller-supplied subset so a
manual dispatch can target a single model without editing the catalog.

Usage:
    ppu_e2e_select.py [MODELS]

``MODELS`` is a comma-separated list of catalog keys. An empty value selects
every model. Unknown keys are rejected so a typo fails fast instead of silently
running nothing. The resulting matrix is printed as ``matrix=<json>`` so a
workflow step can append it to ``$GITHUB_OUTPUT``.

Offline E2E outputs
-------------------
The nightly offline lanes (``e2e-ppu.yaml``) reuse the same catalog as the
source of truth for the model checkpoint and its pytest routing, so this helper
also prints three scalar outputs derived from the *first* selected entry:

``model_root``
    The checkpoint path injected as ``VLLM_SAIL_MODEL_ROOT`` for the offline
    model lane. It is emitted **only when the selection resolves to a single
    model** (a manual dispatch targeting one catalog key), where overriding the
    runtime weight path is both meaningful and correct. When several models are
    selected (the nightly default), ``model_root`` is left **empty**: the offline
    lane then runs the whole ``tests/e2e/models`` directory and each test
    resolves its own hard-coded checkpoint constant, because a single global
    ``VLLM_SAIL_MODEL_ROOT`` cannot point four different models at four
    different weight directories. ``ppu_e2e_offline.sh`` only exports the
    override when ``MODEL_PATH`` is non-empty, so an empty value cleanly falls
    back to per-test constants.
``offline_test_path`` / ``offline_marker``
    The pytest path and ``-m`` expression the offline model lane runs, so the
    workflow never hard-codes them (defaults keep older catalogs working).
    ``offline_test_path`` follows the selection width: a single-model dispatch
    runs that entry's own test file, while a multi-model selection (nightly)
    falls back to the ``tests/e2e/models`` directory default so every model
    test is collected and resolves its own checkpoint -- an entry's file path
    must never narrow the nightly lane to one model.

These are additive: the ``matrix=`` line and the smoke flow are unchanged, and
unknown model keys still raise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CATALOG = Path(__file__).resolve().parent / "ppu_e2e_models.json"


def build_matrix(catalog: list[dict], models: str) -> dict:
    """Return a ``{"include": [...]}`` matrix filtered by ``models``.

    ``models`` is a comma-separated list of catalog keys; blank selects all.
    Requested keys preserve the caller's order. Unknown keys raise ``ValueError``.
    """
    by_key = {entry["key"]: entry for entry in catalog}
    requested = [item.strip() for item in models.split(",") if item.strip()]
    if not requested:
        return {"include": list(catalog)}
    unknown = [key for key in requested if key not in by_key]
    if unknown:
        available = ", ".join(by_key)
        raise ValueError(
            f"unknown model key(s): {', '.join(unknown)}; available: {available}"
        )
    return {"include": [by_key[key] for key in requested]}


def _first_entry(matrix: dict) -> dict:
    """Return the first ``include`` entry, or ``{}`` when the matrix is empty."""
    include = matrix.get("include") or []
    return include[0] if include else {}


def _model_root(matrix: dict) -> str:
    """Return the ``VLLM_SAIL_MODEL_ROOT`` override for the offline model lane.

    A single-model selection overrides the runtime weight path with that model's
    checkpoint (meaningful and unambiguous). A multi-model selection returns an
    empty string so no global override is exported and every ``model_e2e`` test
    resolves its own hard-coded checkpoint constant — see the module docstring.
    """
    include = matrix.get("include") or []
    if len(include) == 1:
        return include[0].get("checkpoint", "")
    return ""


def _offline_test_path(matrix: dict) -> str:
    """Return the pytest path the offline model lane runs.

    A single-model selection runs that entry's own test file, keeping the
    dispatch narrow (the exported ``model_root`` override then applies to
    exactly that model's tests). A multi-model selection falls back to the
    directory default regardless of what the first entry declares, so the
    nightly lane still collects every model test -- mirrors ``_model_root``.
    """
    include = matrix.get("include") or []
    if len(include) == 1:
        return include[0].get("offline_test_path", "tests/e2e/models")
    return "tests/e2e/models"


def main(argv: list[str]) -> int:
    models = argv[0] if argv else ""
    catalog = json.loads(CATALOG.read_text())
    matrix = build_matrix(catalog, models)
    print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
    # Additive scalar outputs for the offline nightly lanes (see module docstring).
    first = _first_entry(matrix)
    print(f"model_root={_model_root(matrix)}")
    print(f"offline_test_path={_offline_test_path(matrix)}")
    print(f"offline_marker={first.get('offline_marker', 'model_e2e')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
