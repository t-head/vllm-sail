# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
from pathlib import Path

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
