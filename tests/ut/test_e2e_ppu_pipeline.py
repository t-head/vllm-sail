# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "ci" / "ppu_e2e_smoke.sh"
SELECT = ROOT / "scripts" / "ci" / "ppu_e2e_select.py"
MODELS = ROOT / "scripts" / "ci" / "ppu_e2e_models.json"
WORKFLOW = ROOT / ".github" / "workflows" / "e2e-ppu.yaml"
DIAGNOSTICS = ROOT / ".github" / "workflows" / "ppu-diagnostics.yaml"


def _load_select():
    spec = importlib.util.spec_from_file_location("ppu_e2e_select", SELECT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- model catalog -----------------------------------------------------------


def test_models_catalog_is_valid_json_with_required_fields() -> None:
    catalog = json.loads(MODELS.read_text())
    assert isinstance(catalog, list) and catalog
    required = {"key", "checkpoint", "served_name", "board", "tp"}
    for entry in catalog:
        assert required <= set(entry), entry
    keys = [entry["key"] for entry in catalog]
    assert len(keys) == len(set(keys)), "model keys must be unique"


def test_default_matrix_targets_qwen_on_810e() -> None:
    catalog = json.loads(MODELS.read_text())
    qwen = next(e for e in catalog if e["key"] == "qwen3.8-27b")
    # Checkpoints are read from /nas_aisw/datasets, the internal NAS root
    # recorded in the model catalog sheet.
    assert qwen["checkpoint"] == (
        "/nas_aisw/datasets/checkpoints/LLM/Qwen/v3.8/Qwen3.8-27B"
    )
    # The built wheels (arch ppu_15;ppu_10) are only verified on 810E.
    assert qwen["board"] == "OAM-810E"
    assert int(qwen["tp"]) == 2


# --- matrix selector ----------------------------------------------------------


def test_selector_default_returns_every_model() -> None:
    select = _load_select()
    catalog = json.loads(MODELS.read_text())
    matrix = select.build_matrix(catalog, "")
    assert [m["key"] for m in matrix["include"]] == [e["key"] for e in catalog]


def test_selector_filters_to_requested_keys() -> None:
    select = _load_select()
    catalog = [
        {"key": "a", "checkpoint": "/x", "served_name": "a", "board": "b", "tp": 1},
        {"key": "b", "checkpoint": "/y", "served_name": "b", "board": "b", "tp": 1},
    ]
    matrix = select.build_matrix(catalog, " b , a ")
    assert [m["key"] for m in matrix["include"]] == ["b", "a"]


def test_selector_rejects_unknown_key() -> None:
    select = _load_select()
    catalog = [
        {"key": "a", "checkpoint": "/x", "served_name": "a", "board": "b", "tp": 1},
    ]
    try:
        select.build_matrix(catalog, "does-not-exist")
    except (KeyError, ValueError, SystemExit):
        return
    raise AssertionError("unknown model key must be rejected")


def test_selector_cli_emits_github_output_matrix(tmp_path: Path) -> None:
    result = subprocess.run(
        ["python3", str(SELECT), ""],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("matrix="))
    payload = json.loads(line[len("matrix=") :])
    assert "include" in payload and payload["include"]


# --- pod smoke script ---------------------------------------------------------


def test_smoke_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SMOKE)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_smoke_script_preserves_required_e2e_contract() -> None:
    text = SMOKE.read_text()
    required = (
        "set -Eeuo pipefail",
        # wheels are installed without pulling deps to keep the SAIL native ABI.
        "--no-deps",
        "--force-reinstall",
        "from vllm_sail import collect_env",
        "vllm serve",
        "/v1/chat/completions",
        # readiness is confirmed by the vLLM startup banner, not a fixed sleep.
        "Application startup complete",
        # occupancy evidence for the run log.
        "ppu-smi",
        "unset VLLM_VERSION",
        "check_vllm_compatibility(force=True)",
        "ppu_wheel_manifest.py",
    )
    for marker in required:
        assert marker in text, marker


def test_smoke_script_fails_when_serve_dies_early() -> None:
    text = SMOKE.read_text()
    # The readiness poll must detect an early-exiting serve process instead of
    # blocking until timeout.
    assert "kill -0" in text


# --- workflow -----------------------------------------------------------------


def test_e2e_workflow_is_reusable_and_manual_and_uses_cpu_runner() -> None:
    text = WORKFLOW.read_text()
    assert "workflow_dispatch:" in text
    # PR orchestration calls the same workflow as manual smoke runs.
    assert "workflow_call:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    # ppu-scheduler-action itself runs on the CPU runner scale set and dispatches
    # a PPU pod; the workflow must not try to run on PPU directly.
    assert "runs-on: k8s-runner-group-cpu-thead" in text
    assert "t-head/ppu-scheduler-action@main" in text
    # board is data-driven from the matrix (the literal OAM-810E lives in the
    # catalog and is asserted by test_default_matrix_targets_qwen_on_810e).
    assert "board-type=${{ matrix.model.board }}" in text
    assert "contents: read" in text
    # cross-run artifact download needs the actions:read scope.
    assert "actions: read" in text


def test_e2e_workflow_only_exposes_model_smoke() -> None:
    text = WORKFLOW.read_text()
    workflow = yaml.safe_load(text)
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_call", "workflow_dispatch"}
    for event in trigger.values():
        assert set(event["inputs"]) == {"build_run_id", "models", "image"}
    assert set(workflow["jobs"]) == {"prepare", "e2e"}
    assert "if" not in workflow["jobs"]["prepare"]
    assert workflow["jobs"]["e2e"]["needs"] == "prepare"
    for marker in (
        "device_probe_only",
        "probe_suite",
        "probe_sail_commit",
        "ppu_ops_probe",
    ):
        assert marker not in text
    caller = yaml.safe_load(
        (ROOT / ".github/workflows/build-ppu-wheels.yaml").read_text()
    )
    assert caller["jobs"]["smoke"]["uses"] == "./.github/workflows/e2e-ppu.yaml"
    assert caller["jobs"]["gate"]["needs"] == ["build", "pin-unit-tests", "smoke"]


def test_manual_device_probe_is_separate_from_model_and_qualification() -> None:
    assert DIAGNOSTICS.is_file()
    text = DIAGNOSTICS.read_text()
    workflow = yaml.safe_load(text)
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"probe_suite", "probe_sail_commit", "build_run_id", "image"}
    for name in ("probe_sail_commit", "build_run_id", "image"):
        assert inputs[name]["required"] is True
        assert inputs[name]["type"] == "string"
    assert "device_probe_only" not in text
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["concurrency"] == {
        "group": "ppu-diagnostics-${{ github.ref }}-${{ inputs.build_run_id }}",
        "cancel-in-progress": False,
    }
    assert set(workflow["jobs"]) == {"device-probe"}
    job = workflow["jobs"]["device-probe"]
    assert "if" not in job and "needs" not in job
    assert job["runs-on"] == "k8s-runner-group-cpu-thead"
    assert job["container"]["image"] == "${{ inputs.image }}"
    assert job["timeout-minutes"] == 35
    assert job["strategy"] == {
        "fail-fast": False,
        "max-parallel": 2,
        "matrix": {
            "include": [
                {"board": "ppu10", "selector": "OAM-810E"},
                {"board": "ppu15", "selector": "ZW-M890P"},
            ]
        },
    }
    probe = text.split("  device-probe:", 1)[1]
    assert (
        "t-head/ppu-scheduler-action@a4e03cbbdb2624f4871fff30d7081b34cbc4b7d4" in probe
    )
    assert "nproc_per_node: 1" in probe
    assert "PROBE_CODE_SHA: ${{ github.sha }}" in probe
    assert "EXPECTED_SAIL_COMMIT: ${{ inputs.probe_sail_commit }}" in probe
    assert "ppu_ops_probe.py run" in probe
    assert probe.index("ppu_ops_probe.py prepare") < probe.index(
        "ppu_ops_probe.py watch"
    )
    assert probe.index("ppu_ops_probe.py watch") < probe.index(
        "uses: t-head/ppu-scheduler-action"
    )
    assert "ppu_e2e_smoke.sh" not in probe and "ppu_ops_runner.py run" not in probe
    scheduler = next(
        s for s in job["steps"] if "ppu-scheduler-action" in s.get("uses", "")
    )
    assert scheduler["with"]["image"] == job["container"]["image"]
    assert scheduler["with"]["cleanup_policy"] == "always"
    assert scheduler["with"]["job_suffix"] == "probe-${{ matrix.board }}"
    assert scheduler["with"]["timeout_minutes"] == 25
    assert scheduler["with"]["node_selector"] == "board-type=${{ matrix.selector }}"
    downloads = [
        s["with"]
        for s in job["steps"]
        if s.get("uses") == "actions/download-artifact@v4"
    ]
    assert {s["name"] for s in downloads} == {
        "ppu-wheels-${{ inputs.build_run_id }}",
        "ppu-build-diagnostics-${{ inputs.build_run_id }}",
        "ppu-ops-selection-${{ inputs.build_run_id }}-1",
    }
    assert all(s["run-id"] == "${{ inputs.build_run_id }}" for s in downloads)
    upload = next(
        s for s in job["steps"] if s.get("uses") == "actions/upload-artifact@v4"
    )
    assert upload["if"] == "always()"
    assert upload["with"] == {
        "name": "ppu-ops-probe-${{ github.run_id }}-${{ github.run_attempt }}-${{ matrix.board }}",
        "path": "/wl_nas/devops/${{ github.run_id }}-${{ github.run_attempt }}/device-probe/${{ matrix.board }}/",
        "if-no-files-found": "error",
        "retention-days": 14,
    }
    stop = next(s for s in job["steps"] if "watch.stop" in s.get("run", ""))
    assert stop["if"] == "always()"


def test_manual_gdn_probe_is_explicit_and_reuses_strict_evidence() -> None:
    assert DIAGNOSTICS.is_file()
    workflow = yaml.safe_load(DIAGNOSTICS.read_text())
    trigger = workflow.get("on", workflow.get(True))
    suite = trigger["workflow_dispatch"]["inputs"]["probe_suite"]
    assert suite["type"] == "choice"
    assert suite["default"] == "visibility"
    assert suite["options"] == ["visibility", "gdn"]
    assert "workflow_call" not in trigger
    probe = workflow["jobs"]["device-probe"]
    assert probe["env"]["PROBE_SUITE"] == "${{ inputs.probe_suite }}"
    scheduler = next(
        s for s in probe["steps"] if "ppu-scheduler-action" in s.get("uses", "")
    )
    assert "PROBE_SUITE=${{ inputs.probe_suite }}" in scheduler["with"]["extra_env"]
    finish = next(
        s for s in probe["steps"] if "ppu_ops_probe.py finish" in s.get("run", "")
    )
    assert finish["if"] == "always() && inputs.probe_suite == 'gdn'"
    assert "continue-on-error" not in DIAGNOSTICS.read_text()


def test_e2e_workflow_pulls_wheels_by_run_id() -> None:
    text = WORKFLOW.read_text()
    assert "build_run_id:" in text
    assert "actions/download-artifact@v4" in text
    assert "run-id: ${{ inputs.build_run_id }}" in text
    assert "ppu-wheels-${{ inputs.build_run_id }}" in text


def test_e2e_workflow_keeps_python_frontend_and_privileged() -> None:
    text = WORKFLOW.read_text()
    assert "VLLM_VERSION=" not in text
    assert "VLLM_USE_RUST_FRONTEND=0" in text
    assert "VLLM_USE_RUST_BENCH=0" in text
    assert "privileged: true" in text
    # smoke work is delegated to the checked-in script, not inlined in YAML.
    assert "scripts/ci/ppu_e2e_smoke.sh" in text


def test_e2e_workflow_gives_every_cpu_runner_job_a_container() -> None:
    text = WORKFLOW.read_text()
    # k8s-runner-group-cpu-thead rejects container-less jobs ("Jobs without a
    # job container are forbidden on this runner"), so every job on that runner
    # must declare a container image.
    runner_jobs = text.count("runs-on: k8s-runner-group-cpu-thead")
    container_decls = sum(
        1 for line in text.splitlines() if line.strip() == "container:"
    )
    assert runner_jobs >= 1
    assert container_decls >= runner_jobs, (runner_jobs, container_decls)


def test_smoke_checks_manifest_before_install_and_uses_installed_package() -> None:
    text = SMOKE.read_text()
    assert text.index("ppu_wheel_manifest.py") < text.index("python -m pip install")
    assert text.index("cd /tmp") < text.index("check_vllm_compatibility(force=True)")
    assert "export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0" in text
    assert "unset PYTHONPATH" in text
