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
        "releases/v0.27.1",
        "unset TORCH_CUDA_ARCH_LIST",
        "command -v hgcc",
        "command -v nvcc",
        "python use_existing_torch.py",
        "VLLM_TARGET_DEVICE=empty",
        "python setup.py bdist_wheel",
        "VERBOSE=1",
        "build-manifest.txt",
    )
    for marker in required:
        assert marker in text


def test_build_workflow_is_manual_and_uses_cpu_runner() -> None:
    text = WORKFLOW.read_text()
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "runs-on: k8s-runner-group-cpu-thead" in text
    assert "ppu-scheduler-action" not in text
    assert "contents: read" in text


def test_build_workflow_uses_script_and_retains_artifacts() -> None:
    text = WORKFLOW.read_text()
    assert "bash scripts/ci/build_ppu_wheels.sh" in text
    assert text.count("actions/upload-artifact@v4") == 2
    assert text.count("retention-days: 14") == 2
    assert "if-no-files-found: error" in text
    assert "if-no-files-found: warn" in text
    assert "if: always()" in text
