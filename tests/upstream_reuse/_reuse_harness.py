# SPDX-License-Identifier: Apache-2.0
"""Harness that loads whitelisted upstream vLLM tests into the vllm-sail runtime.

vllm-sail pins an upstream vLLM release and disguises its PPU platform as CUDA
(``PPUPlatform._enum = PlatformEnum.CUDA``, so ``current_platform.is_cuda()`` is
True). Upstream tests written against the CUDA platform -- but that assert
scheduling / config / sampling / tokenization *logic* rather than concrete
kernel numerics -- therefore run unchanged inside the vllm-sail runtime. This
module is the mechanism that makes that reuse explicit, auditable and CI-gated.

Design rules (mirroring :mod:`tests.ut.test_upstream_version_precheck`):

* Importing this module never imports torch or vLLM. It only reads the
  committed ``REUSE_MANIFEST.toml`` and shells out. Collection stays CPU-only,
  so ``--collect-only`` reports 0 errors on a bare host.
* The upstream checkout is located from ``VLLM_SOURCE_ROOT`` (the same variable
  the rest of the suite uses). Without it, the whole directory is skipped.
* An upstream test file is executed as an **isolated pytest subprocess** rooted
  at the upstream checkout. This is deliberate: it lets upstream's own
  ``conftest.py`` fixtures and markers resolve, and it confines an upstream
  ``ImportError`` (missing optional dependency, drifted module) to that one
  parameterized case instead of corrupting our collection. The subprocess
  inherits the ambient environment, which -- in CI -- has vllm-sail installed,
  so ``current_platform`` is the CUDA-disguised PPU platform while the upstream
  test body believes it is talking to CUDA.
* Results are read back from a JUnit XML report so pass/fail/skip/error counts
  and the exact failing node names are available for the expected-xfail logic in
  ``test_reuse_adapted.py``.

Everything here is Python 3.9-compatible: annotations are strings (PEP 563 via
``from __future__ import annotations``) and no runtime ``X | Y`` union or
``zip(strict=)`` is used.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, NamedTuple

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.9/3.10
    import tomli as tomllib

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
MANIFEST_PATH = HERE / "REUSE_MANIFEST.toml"

MANIFEST_SCHEMA = 1
VALID_MODES = ("direct", "adapted")
VALID_CHIPS = ("cpu", "810e", "890p")
#: Subprocess wall-clock ceiling for one upstream test file.
RUN_TIMEOUT_SECONDS = 1800


class ReuseEntry(NamedTuple):
    """One whitelisted upstream test file and how to run it."""

    id: str
    path: str
    mode: str
    chips: tuple[str, ...]
    category: str
    reason: str
    min_tests: int
    markers: str
    keywords: str
    deselect: tuple[str, ...]
    env: dict[str, str]
    xfail: tuple[str, ...]
    skip_reason: str

    @property
    def cpu_runnable(self) -> bool:
        """True when the entry can execute on a device-less CPU runner."""
        return "cpu" in self.chips

    @property
    def device_only(self) -> bool:
        """True when the entry needs a real PPU (excluded from the CPU lane)."""
        return not self.cpu_runnable


class RunResult(NamedTuple):
    """Parsed outcome of one upstream pytest subprocess."""

    returncode: int
    collected: int
    passed: int
    failures: int
    errors: int
    skipped: int
    failed_names: tuple[str, ...]
    tail: str
    skipped_by_manifest: bool = False


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------
def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def load_manifest(path: Path = MANIFEST_PATH) -> list[ReuseEntry]:
    """Parse ``REUSE_MANIFEST.toml`` into validated :class:`ReuseEntry` rows."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    schema = meta.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise AssertionError(f"unsupported REUSE_MANIFEST schema: {schema!r}")
    raw_entries = data.get("entry", [])
    if not isinstance(raw_entries, list) or not raw_entries:
        raise AssertionError("REUSE_MANIFEST.toml declares no [[entry]] rows")

    entries: list[ReuseEntry] = []
    seen: set[str] = set()
    for raw in raw_entries:
        entry_id = raw["id"]
        if entry_id in seen:
            raise AssertionError(f"duplicate manifest id: {entry_id!r}")
        seen.add(entry_id)
        mode = raw["mode"]
        if mode not in VALID_MODES:
            raise AssertionError(f"{entry_id}: invalid mode {mode!r}")
        chips = _as_str_tuple(raw.get("chips"))
        for chip in chips:
            if chip not in VALID_CHIPS:
                raise AssertionError(f"{entry_id}: invalid chip {chip!r}")
        if not chips:
            raise AssertionError(f"{entry_id}: declares no chips")
        env_raw = raw.get("env", {}) or {}
        entries.append(
            ReuseEntry(
                id=entry_id,
                path=raw["path"],
                mode=mode,
                chips=chips,
                category=raw.get("category", ""),
                reason=raw.get("reason", ""),
                min_tests=int(raw.get("min_tests", 1)),
                markers=raw.get("markers", ""),
                keywords=raw.get("keywords", ""),
                deselect=_as_str_tuple(raw.get("deselect")),
                env={str(key): str(val) for key, val in env_raw.items()},
                xfail=_as_str_tuple(raw.get("xfail")),
                skip_reason=raw.get("skip_reason", ""),
            )
        )
    return entries


def entries_for_mode(mode: str, path: Path = MANIFEST_PATH) -> list[ReuseEntry]:
    """Manifest rows whose ``mode`` equals ``mode`` (order preserved)."""
    return [entry for entry in load_manifest(path) if entry.mode == mode]


def entry_marks(entry: ReuseEntry) -> list[pytest.MarkDecorator]:
    """Markers that route one entry through the shared device/chip gates.

    Every reused upstream test is an ``upstream_source`` check (it needs the
    selected checkout). Entries that cannot run on a device-less host are also
    marked ``ppu`` -- so the PR CPU lane's ``-m "upstream_source and not ppu"``
    excludes them and ``tests/conftest.py`` auto-skips them without a real
    device. Chip-exclusive entries additionally carry ``cap80`` / ``cap89``.
    """
    marks = [pytest.mark.upstream_source]
    if entry.device_only:
        marks.append(pytest.mark.ppu)
        has_810e = "810e" in entry.chips
        has_890p = "890p" in entry.chips
        if has_810e and not has_890p:
            marks.append(pytest.mark.cap80)
        elif has_890p and not has_810e:
            marks.append(pytest.mark.cap89)
    return marks


def param_for_entry(entry: ReuseEntry) -> pytest.param:
    """A ``pytest.param`` carrying the entry, its id and its routing marks."""
    return pytest.param(entry, id=entry.id, marks=entry_marks(entry))


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------
def resolve_source_root(raw: str | None) -> Path | None:
    """Return the upstream checkout for ``raw``, or ``None`` when unusable.

    Unlike ``tests.support.source.source_root`` this never raises or calls
    ``pytest.skip`` -- the caller decides. ``None`` means "no upstream source",
    which the directory conftest turns into a whole-directory skip.
    """
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    if not (root / "vllm" / "__init__.py").is_file():
        return None
    return root


def runtime_available() -> tuple[bool, str]:
    """Whether an installed vLLM + torch runtime can actually execute tests.

    Upstream test modules ``import vllm`` / ``import torch`` at module scope, so
    without both installed the reuse subprocess can only fail on import. We
    detect that up front and skip gracefully instead -- this is what keeps a
    bare host (no vLLM) at "all skipped, 0 errors".
    """
    if importlib.util.find_spec("torch") is None:
        return False, "torch is not installed in this environment"
    if importlib.util.find_spec("vllm") is None:
        return False, "vllm is not installed in this environment"
    return True, ""


# ---------------------------------------------------------------------------
# JUnit XML parsing
# ---------------------------------------------------------------------------
def _parse_junit(path: Path) -> tuple[int, int, int, int, list[str]]:
    """Return (collected, failures, errors, skipped, failed_names) from XML."""
    collected = failures = errors = skipped = 0
    failed_names: list[str] = []
    if not path.is_file():
        return collected, failures, errors, skipped, failed_names
    root = ET.parse(path).getroot()  # noqa: S314 - trusted local CI artifact
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        collected += int(suite.get("tests", "0") or 0)
        skipped += int(suite.get("skipped", "0") or 0)
        for case in suite.iter("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            node = f"{classname}::{name}" if classname else name
            has_failure = case.find("failure") is not None
            has_error = case.find("error") is not None
            if has_failure:
                failures += 1
                failed_names.append(node)
            elif has_error:
                errors += 1
                failed_names.append(node)
    return collected, failures, errors, skipped, failed_names


# ---------------------------------------------------------------------------
# Upstream test runner
# ---------------------------------------------------------------------------
class UpstreamRunner:
    """Executes whitelisted upstream test files inside the ambient runtime."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def target(self, entry: ReuseEntry) -> Path:
        return self.root / entry.path

    def build_command(self, entry: ReuseEntry, junit_path: Path) -> list[str]:
        """Assemble the pytest argv for one entry (no execution)."""
        command = [
            sys.executable,
            "-m",
            "pytest",
            entry.path,
            "-q",
            "-rN",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            f"--junitxml={junit_path}",
        ]
        if entry.markers:
            command += ["-m", entry.markers]
        if entry.keywords:
            command += ["-k", entry.keywords]
        for node in entry.deselect:
            command += ["--deselect", node]
        return command

    def build_env(self, entry: ReuseEntry) -> dict[str, str]:
        """Ambient env + upstream root on PYTHONPATH + per-entry overrides."""
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        parts = [str(self.root)]
        if existing:
            parts.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        env.update(entry.env)
        return env

    def run(self, entry: ReuseEntry) -> RunResult:
        """Run one upstream test file and parse its JUnit report."""
        if entry.skip_reason:
            return RunResult(
                returncode=0,
                collected=0,
                passed=0,
                failures=0,
                errors=0,
                skipped=0,
                failed_names=(),
                tail=entry.skip_reason,
                skipped_by_manifest=True,
            )
        target = self.target(entry)
        if not target.is_file():
            return RunResult(
                returncode=-1,
                collected=0,
                passed=0,
                failures=0,
                errors=1,
                skipped=0,
                failed_names=(f"MISSING::{entry.path}",),
                tail=f"upstream test file not found at pinned commit: {entry.path}",
            )
        with tempfile.TemporaryDirectory(prefix="vllm_sail_reuse_") as workdir:
            junit_path = Path(workdir) / "report.xml"
            command = self.build_command(entry, junit_path)
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.root),
                    env=self.build_env(entry),
                    capture_output=True,
                    text=True,
                    timeout=RUN_TIMEOUT_SECONDS,
                    check=False,
                )
                returncode = completed.returncode
                output = completed.stdout + "\n" + completed.stderr
            except subprocess.TimeoutExpired as exc:
                returncode = -1
                output = f"upstream reuse run timed out after {RUN_TIMEOUT_SECONDS}s\n"
                output += (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            collected, failures, errors, skipped, failed_names = _parse_junit(
                junit_path
            )
        passed = max(collected - failures - errors - skipped, 0)
        tail = "\n".join(line for line in output.splitlines() if line.strip())[-2000:]
        return RunResult(
            returncode=returncode,
            collected=collected,
            passed=passed,
            failures=failures,
            errors=errors,
            skipped=skipped,
            failed_names=tuple(failed_names),
            tail=tail,
        )
