# SPDX-License-Identifier: Apache-2.0
"""Distribution metadata and built-artifact invariants."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

import pytest

from vllm_sail.native import resources

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by Python 3.10 CI
    import tomli as tomllib


def test_every_runtime_json_matches_explicit_package_data() -> None:
    """A wheel must carry every tuned config consumed at runtime."""
    root = Path(__file__).parents[2]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    patterns = pyproject["tool"]["setuptools"]["package-data"]["vllm_sail"]
    runtime_json = sorted((root / "vllm_sail").rglob("*.json"))

    missing = [
        path.relative_to(root / "vllm_sail").as_posix()
        for path in runtime_json
        if not any(
            PurePosixPath(path.relative_to(root / "vllm_sail").as_posix()).match(
                pattern
            )
            for pattern in patterns
        )
    ]

    assert runtime_json, "the package-data test needs at least one runtime JSON"
    assert missing == []


def test_native_runtime_metadata_has_one_canonical_source() -> None:
    root = Path(__file__).parents[2]

    assert set(resources.RUNTIME_METADATA) == {"manifest.toml", "excluded_ops.toml"}
    for name in resources.RUNTIME_METADATA:
        assert resources.source_path(root, name).is_file()
        assert resources.resolve(root / "vllm_sail" / "native", name) == (
            root / "csrc" / "upstream" / name
        )


def test_native_runtime_metadata_is_self_contained_after_build(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    canonical = source_root / "csrc" / "upstream"
    canonical.mkdir(parents=True)
    for name in resources.RUNTIME_METADATA:
        (canonical / name).write_text(f"source = {name!r}\n", encoding="utf-8")

    build_lib = tmp_path / "build"
    copied = resources.copy_to_build(source_root, build_lib)
    native_dir = build_lib / "vllm_sail" / "native"

    assert copied == resources.build_outputs(build_lib)
    for name in resources.RUNTIME_METADATA:
        resolved = resources.resolve(native_dir, name)
        assert resolved == native_dir / "data" / name
        assert resolved.read_text(encoding="utf-8") == f"source = {name!r}\n"


def test_native_runtime_metadata_refuses_unknown_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown native runtime metadata"):
        resources.source_path(tmp_path, "surprise.toml")


def test_sdist_includes_the_native_build_inputs() -> None:
    root = Path(__file__).parents[2]
    manifest_in = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()

    assert "include CMakeLists.txt" in manifest_in
    assert "recursive-include cmake *.cmake" in manifest_in
    assert (
        "recursive-include csrc *.cpp *.cu *.cuh *.hg *.h *.hpp *.toml" in manifest_in
    )
    assert "recursive-include requirements *.txt" in manifest_in


def test_built_wheel_and_sdist_are_self_contained(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    source = tmp_path / "source"
    shutil.copytree(
        root,
        source,
        ignore=shutil.ignore_patterns(
            ".git", ".venv-test", "build", "dist", "*.egg-info", "__pycache__"
        ),
    )
    artifacts = tmp_path / "artifacts"
    env = os.environ.copy()
    env.update(
        VLLM_SAIL_SKIP_EXT="1",
        SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_SAIL="0.0.0",
    )
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(artifacts),
            "sdist",
            "--dist-dir",
            str(artifacts),
        ],
        cwd=source,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(artifacts.glob("*.whl"))
    sdist = next(artifacts.glob("*.tar.gz"))
    assert wheel.name == "vllm_sail-0.0.0-py3-none-any.whl"
    assert sdist.name == "vllm_sail-0.0.0.tar.gz"
    wheel_metadata = {
        "vllm_sail/native/data/manifest.toml",
        "vllm_sail/native/data/excluded_ops.toml",
    }
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        assert wheel_metadata <= wheel_names
        assert not any(name.startswith("vllm_ppu/") for name in wheel_names)
        metadata = Parser().parsestr(
            archive.read("vllm_sail-0.0.0.dist-info/METADATA").decode()
        )
        assert metadata["Name"] == "vllm-sail"
        assert metadata["Version"] == "0.0.0"
        config_paths = {
            path.relative_to(root).as_posix()
            for path in (root / "vllm_sail").rglob("*.json")
        }
        assert config_paths <= wheel_names

    with tarfile.open(sdist) as archive:
        sdist_names = set(archive.getnames())
    prefix = next(name.split("/", 1)[0] for name in sdist_names if "/" in name)
    native_suffixes = {".cpp", ".cu", ".cuh", ".hg", ".h", ".hpp", ".toml"}
    required_build_inputs = {
        f"{prefix}/{path.relative_to(root).as_posix()}"
        for path in (root / "csrc").rglob("*")
        if path.is_file() and path.suffix in native_suffixes
    }
    required_build_inputs.add(f"{prefix}/CMakeLists.txt")
    required_build_inputs.update(
        f"{prefix}/{path.relative_to(root).as_posix()}"
        for path in (root / "cmake").rglob("*.cmake")
    )
    required_build_inputs.update(
        f"{prefix}/{path.relative_to(root).as_posix()}"
        for path in (root / "requirements").glob("*.txt")
    )
    assert required_build_inputs <= sdist_names
    # The sdist must build directly from the same authored HGGC files.
    assert any(name.endswith("/csrc/plugin/sampler_bf16.hg") for name in sdist_names)
    assert not any("/csrc/plugin_generated/" in name for name in sdist_names)

    # Install the actual wheel, then import it with site-packages disabled so
    # the checkout, old plugin installations, torch, and vLLM cannot hide leaks.
    installed = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--no-compile",
            "--target",
            str(installed),
            str(wheel),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    import_env = os.environ.copy()
    import_env["PYTHONPATH"] = str(installed)
    checked = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            """
import importlib.metadata as metadata
import importlib.util
import sys
from pathlib import Path

import vllm_sail

installed = Path(sys.argv[1])
assert Path(vllm_sail.__file__).is_relative_to(installed)
assert vllm_sail.__version__ == metadata.version('vllm-sail') == '0.0.0'
assert all(importlib.util.find_spec(name) is None for name in
           ('vllm_ppu', 'torch', 'vllm'))
points = metadata.distribution('vllm-sail').entry_points
assert {(point.group, point.name, point.value) for point in points} == {
    ('vllm.platform_plugins', 'ppu', 'vllm_sail:register'),
    ('vllm.general_plugins', 'ppu', 'vllm_sail:register_out_of_tree'),
}
for point in points:
    assert point.load() is getattr(vllm_sail, point.attr)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
""",
            str(installed),
        ],
        cwd=tmp_path,
        env=import_env,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr

    # Metadata readers use the dev dependency tomli on Python 3.10, so check
    # those with site-packages enabled while keeping the installed wheel first.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm_sail.native import manifest, stubs; "
            "assert manifest.load().entries; assert stubs.load_schemas()",
        ],
        cwd=tmp_path,
        env=import_env,
        check=True,
        capture_output=True,
        text=True,
    )

    # A published source archive must also rebuild under the new package name
    # without a Git checkout or the original source directory.
    extracted = tmp_path / "sdist"
    shutil.unpack_archive(sdist, extracted)
    rebuilt = tmp_path / "rebuilt"
    sdist_env = env.copy()
    sdist_env.pop("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_SAIL")
    sdist_env.pop("SETUPTOOLS_SCM_PRETEND_VERSION", None)
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(rebuilt),
        ],
        cwd=extracted / prefix,
        env=sdist_env,
        check=True,
        capture_output=True,
        text=True,
    )
    with zipfile.ZipFile(rebuilt / wheel.name) as archive:
        assert set(archive.namelist()) == wheel_names
