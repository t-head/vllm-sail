#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Vendor upstream vLLM kernels into csrc/upstream/ as a CUDA-free corpus.

    python tools/port_upstream_kernels.py --vllm-src /path/to/vllm
    python tools/port_upstream_kernels.py --vllm-src /path/to/vllm --check

The port itself lives in :mod:`vllm_sail.native.porting`; this script is the
environment-facing half -- locating a vLLM checkout, confirming it sits at the
ref the manifest pins, and diffing a fresh port against the committed tree.

Requires the sailify translator (https://github.com/t-head/sailify) on
PYTHONPATH. It needs no torch, no vLLM install and no PPU SDK: translation is a
source-to-source rewrite, and compiling the result is setup.py's job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools._common import corpus_differences  # noqa: E402
from vllm_sail.native import manifest as manifest_reader  # noqa: E402
from vllm_sail.native import porting  # noqa: E402

DEST = ROOT / manifest_reader.VENDOR_ROOT


def _check_ref(src: Path, expected: str) -> None:
    """Warn if the checkout is not at the pinned ref. Never fatal.

    A worktree can legitimately be a detached build of the right code, and the
    manifest's own path assertions catch the case that actually matters, so this
    stays advisory rather than blocking a deliberate port.
    """
    try:
        found = subprocess.run(
            ["git", "-C", str(src), "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        print(
            f"WARNING: cannot determine the ref of {src}; the manifest pins {expected}",
            file=sys.stderr,
        )
        return
    if found != expected:
        print(
            f"WARNING: {src} is at {found} but the manifest pins {expected}; "
            "bump upstream_ref (and review the diff) if this is intentional",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vllm-src",
        required=True,
        type=Path,
        help="a vLLM source checkout (the directory containing csrc/)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify csrc/upstream/ matches a fresh port instead of rewriting it",
    )
    args = parser.parse_args(argv)

    src = args.vllm_src.expanduser().resolve()
    if not (src / "csrc").is_dir():
        print(f"ERROR: {src} has no csrc/ directory", file=sys.stderr)
        return 2

    try:
        manifest = manifest_reader.load()
    except manifest_reader.ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _check_ref(src, manifest.upstream_ref)

    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / "upstream"
            fresh.mkdir()
            try:
                porting.port(src, fresh, manifest)
            except porting.PortError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            drift = corpus_differences(fresh, DEST, ignore=porting.PRESERVED)
        if drift:
            print(
                "ERROR: csrc/upstream/ is stale; re-run without --check:",
                file=sys.stderr,
            )
            for line in drift:
                print(f"  {line}", file=sys.stderr)
            return 1
        print(f"csrc/upstream/ matches a fresh port of vLLM {manifest.upstream_ref}")
        return 0

    try:
        result = porting.port(src, DEST, manifest)
    except porting.PortError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(porting.report(manifest, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
