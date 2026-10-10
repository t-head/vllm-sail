# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "build_ppu_wheels.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "build-ppu-wheels.yaml"


def test_build_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_build_script_preserves_required_build_contract() -> None:
    text = SCRIPT.read_text()
    required = (
        "set -Eeuo pipefail",
        "ppu_1.0.0_Ubuntu2404_v13_release.run",
        "ppu_sdk_hggcrt3-pytorch2.13.0-ubuntu2404-py312.tar.gz",
        ".github/vllm-main-verified.commit",
        "unset TORCH_CUDA_ARCH_LIST",
        "command -v hgcc",
        "command -v nvcc",
        "python tools/use_existing_torch.py",
        "VLLM_TARGET_DEVICE=empty",
        "VLLM_REQUIRE_RUST_FRONTEND=1",
        "CARGO_BUILD_JOBS",
        "rust-toolchain.toml",
        "rustc --version",
        "cargo --version",
        "--no-build-isolation --no-deps",
        'git -C "${VLLM_SRC}" fetch --tags',
        "check_vllm_compatibility()",
        "ppu_wheel_manifest.py",
        "VERBOSE=1",
        "build-manifest.txt",
        "printf 'build_image=%s\\n' \"${VLLM_SAIL_BUILD_IMAGE:-}\"",
    )
    for marker in required:
        assert marker in text


def test_build_script_uses_configurable_aliyun_pip_mirror() -> None:
    text = SCRIPT.read_text()
    forced = (
        'export PIP_INDEX_URL="${SAIL_PIP_INDEX_URL:-'
        'https://mirrors.aliyun.com/pypi/simple/}"'
    )
    assert forced in text
    # The PPU SDK envsetup.sh repoints pip at the unreachable internal mirror,
    # so the public mirror must be forced AFTER sourcing it.
    assert text.index("source /usr/local/PPU_SDK/envsetup.sh") < text.index(forced)


def test_build_script_caps_parallelism_to_avoid_oom() -> None:
    text = SCRIPT.read_text()
    # Native PPU kernel compiles are memory-heavy; an unbounded MAX_JOBS OOM-kills
    # the pod (exit 137). Parallelism must be capped by the cgroup memory limit
    # while staying overridable via BUILD_JOBS.
    assert "/sys/fs/cgroup/memory.max" in text
    assert 'BUILD_JOBS="${BUILD_JOBS:-' in text


def test_build_workflow_supports_prs_and_manual_runs_on_cpu_runner() -> None:
    text = WORKFLOW.read_text()
    assert "workflow_dispatch:" in text
    assert "pull_request:" in text
    assert "push:" not in text
    # ppu-scheduler-action does not support compiling on PPU resources; t-head
    # build jobs must run on the CPU runner scale set. The pod is sized with
    # enough memory for the memory-aware parallelism cap to compile in parallel.
    assert "runs-on: k8s-runner-group-cpu-thead" in text
    assert "ppu-scheduler-action" not in text
    assert "contents: read" in text


def test_build_workflow_uses_script_and_retains_artifacts() -> None:
    text = WORKFLOW.read_text()
    assert "bash scripts/ci/build_ppu_wheels.sh" in text
    import yaml

    jobs = yaml.safe_load(text)["jobs"]
    for job in ("build", "device-build"):
        uploads = {
            step["with"]["name"]: step["with"]
            for step in jobs[job]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        }
        assert set(uploads) == {
            "ppu-wheels-${{ github.run_id }}",
            "ppu-build-diagnostics-${{ github.run_id }}",
        }
        assert all(config["retention-days"] == 14 for config in uploads.values())
        assert (
            uploads["ppu-wheels-${{ github.run_id }}"]["if-no-files-found"] == "error"
        )
    for job, name in (
        ("detect-device-changes", "changes"),
        ("select-device-tests", "selection"),
    ):
        assert any(
            step.get("with", {}).get("name")
            == f"ppu-device-{name}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}"
            for step in jobs[job]["steps"]
        )
    assert "if-no-files-found: error" in text
    assert "if-no-files-found: warn" in text
    assert "if: always()" in text


def test_build_workflow_uploads_manifest_with_wheels() -> None:
    assert "path: artifacts/wheels/" in WORKFLOW.read_text()
    assert "--depth 1" not in SCRIPT.read_text()


@pytest.mark.parametrize(
    "cargo_env", ["", "export CARGO_NET_GIT_FETCH_WITH_CLI=false\n"]
)
def test_cargo_git_cli_is_exported_after_environment_setup(tmp_path, cargo_env):
    text = SCRIPT.read_text()
    start = text.index('    if [[ -r "${CARGO_HOME:-$HOME/.cargo}/env" ]]')
    end = text.index("    # Follow the Rust toolchain pinned")
    assert text.index("source /usr/local/PPU_SDK/envsetup.sh") < start
    assert start < end < text.index("cargo --version")
    assert end < text.index("python setup.py build_rust --release --inplace")
    (tmp_path / "env").write_text(cargo_env)
    preserved = {
        "GIT_CONFIG_GLOBAL": str(tmp_path / "git-config"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "HTTPS_PROXY": "http://proxy.invalid:8080",
    }
    # Run the actual config snippet and verify that sourced exports reach subprocesses.
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-euo",
            "pipefail",
            "-c",
            text[start:end]
            + "\n\"$1\" -c 'import json, os; print(json.dumps(dict(os.environ)))'\n",
            "bash",
            sys.executable,
        ],
        env={
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "CARGO_HOME": str(tmp_path),
            **preserved,
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    child_env = json.loads(result.stdout)
    assert child_env.get("CARGO_NET_GIT_FETCH_WITH_CLI") == "true"
    assert {key: child_env[key] for key in preserved} == preserved


def test_rust_build_is_explicit_before_wheel_packaging() -> None:
    text = SCRIPT.read_text()
    required = text.index(
        "export VLLM_TARGET_DEVICE=empty VLLM_REQUIRE_RUST_FRONTEND=1"
    )
    rust = text.index("python setup.py build_rust --release --inplace")
    check = text.index("required local Rust artifacts were not built")
    wheel = text.index("python -m pip wheel --verbose")
    assert required < rust < check < wheel
    assert "vllm-rust-build.log" in text
