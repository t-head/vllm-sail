#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare patched upstream source with the committed per-target baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tools" / "patch_baseline.json"
ADDITIVE = "additive"
sys.path.insert(0, str(ROOT))

from tools._common import PatchEnvironmentError, load_patch_records  # noqa: E402


def _fingerprints(records: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in records:
        result[record.target] = (
            ADDITIVE
            if record.original_source is None
            else hashlib.sha256(record.original_source.encode("utf-8")).hexdigest()
        )
    return dict(sorted(result.items()))


def _read_baseline() -> dict[str, str]:
    try:
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            "tools/patch_baseline.json is missing; run this tool with --update"
        ) from exc
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ValueError("tools/patch_baseline.json must map target strings to hashes")
    return data


def _describe(record: Any) -> str:
    return f"reason={record.reason}; remove when={record.remove_when}"


def compare(records: list[Any], baseline: dict[str, str]) -> bool:
    current = _fingerprints(records)
    by_target = {record.target: record for record in records}
    added = sorted(current.keys() - baseline.keys())
    removed = sorted(baseline.keys() - current.keys())
    drifted = sorted(
        target
        for target in current.keys() & baseline.keys()
        if current[target] != baseline[target]
    )

    if not (added or removed or drifted):
        print(f"OK: {len(current)} patch targets match tools/patch_baseline.json")
        return False

    print("Patch drift detected. Review each target against the pinned upstream:")
    for target in added:
        print(f"ADDED   {target}")
        print(f"        {_describe(by_target[target])}")
    for target in removed:
        print(f"REMOVED {target}")
        print("        delete its stale baseline entry or restore the patch")
    for target in drifted:
        print(f"DRIFTED {target}")
        print(f"        {_describe(by_target[target])}")
        print(
            "        compare the upstream body, update/remove the patch, then "
            "rerun with --update"
        )
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="replace the baseline with hashes from the installed vLLM",
    )
    args = parser.parse_args(argv)
    try:
        records = load_patch_records("patch drift checking")
        if args.update:
            current = _fingerprints(records)
            BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
            print(f"wrote {BASELINE} ({len(current)} patch targets)")
            return 0
        return 1 if compare(records, _read_baseline()) else 0
    except (PatchEnvironmentError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
