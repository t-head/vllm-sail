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


def main(argv: list[str]) -> int:
    models = argv[0] if argv else ""
    catalog = json.loads(CATALOG.read_text())
    matrix = build_matrix(catalog, models)
    print(f"matrix={json.dumps(matrix, separators=(',', ':'))}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
