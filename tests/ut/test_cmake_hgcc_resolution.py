# SPDX-License-Identifier: Apache-2.0
"""Host-only checks for cmake-hgcc discovery and fetching."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
RESOLVER = ROOT / "cmake" / "ResolveCmakeHgcc.cmake"
PINNED_COMMIT = "5ca64a13a1487a2ba6a4ca7972d1cb73b2b56303"


def _cmake() -> str:
    adjacent = Path(sys.executable).with_name("cmake")
    cmake = str(adjacent) if adjacent.is_file() else shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake-hgcc resolver checks require CMake")
    return cmake


def _fake_cmake_hgcc(path: Path) -> Path:
    module_dir = path / "cmake" / "Modules"
    module_dir.mkdir(parents=True)
    (module_dir / "CMakeDetermineHGCompiler.cmake").write_text(
        "# test fixture\n", encoding="utf-8"
    )
    return path


def _resolve(
    tmp_path: Path,
    *,
    cmake_hgcc_dir: Path | None = None,
    module_path: Path | None = None,
    fetch_source: Path | None = None,
) -> list[str]:
    source_dir = tmp_path / "project"
    build_dir = tmp_path / "build"
    source_dir.mkdir()
    (source_dir / "CMakeLists.txt").write_text(
        f"""cmake_minimum_required(VERSION 3.26)
project(test_cmake_hgcc_resolver LANGUAGES NONE)
include(\"{RESOLVER.as_posix()}\")
vllm_sail_resolve_cmake_hgcc()
file(WRITE \"${{CMAKE_BINARY_DIR}}/resolved.txt\"
  \"${{VLLM_SAIL_CMAKE_HGCC_SOURCE}}\\n\"
  \"${{CMAKE_HGCC_DIR}}\\n\"
  \"${{CMAKE_MODULE_PATH}}\\n\"
  \"${{VLLM_SAIL_CMAKE_HGCC_GIT_TAG}}\\n\")
""",
        encoding="utf-8",
    )
    argv = [_cmake(), "-S", str(source_dir), "-B", str(build_dir)]
    if cmake_hgcc_dir is not None:
        argv.append(f"-DCMAKE_HGCC_DIR={cmake_hgcc_dir}")
    if module_path is not None:
        argv.append(f"-DCMAKE_MODULE_PATH={module_path}")
    if fetch_source is not None:
        argv.append(f"-DFETCHCONTENT_SOURCE_DIR_CMAKE_HGCC={fetch_source}")
    result = subprocess.run(argv, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return (build_dir / "resolved.txt").read_text(encoding="utf-8").splitlines()


def test_resolver_prefers_an_explicit_checkout(tmp_path: Path) -> None:
    checkout = _fake_cmake_hgcc(tmp_path / "explicit")

    source, resolved, modules, commit = _resolve(tmp_path, cmake_hgcc_dir=checkout)

    assert source == "explicit"
    assert resolved == str(checkout)
    assert str(checkout / "cmake" / "Modules") in modules.split(";")
    assert commit == PINNED_COMMIT


def test_resolver_reuses_an_existing_module_path(tmp_path: Path) -> None:
    checkout = _fake_cmake_hgcc(tmp_path / "configured")
    module_dir = checkout / "cmake" / "Modules"

    source, resolved, modules, _ = _resolve(tmp_path, module_path=module_dir)

    assert source == "configured"
    assert resolved == ""
    assert str(module_dir) in modules.split(";")


def test_resolver_fetches_the_pinned_checkout_when_unconfigured(
    tmp_path: Path,
) -> None:
    checkout = _fake_cmake_hgcc(tmp_path / "fetched")

    source, resolved, modules, commit = _resolve(tmp_path, fetch_source=checkout)

    assert source == "fetched"
    assert resolved == str(checkout)
    assert str(checkout / "cmake" / "Modules") in modules.split(";")
    assert commit == PINNED_COMMIT
