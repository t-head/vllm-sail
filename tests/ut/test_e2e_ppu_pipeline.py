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


def test_catalog_carries_current_accuracy_models() -> None:
    catalog = json.loads(MODELS.read_text())
    by_key = {entry["key"]: entry for entry in catalog}
    assert set(by_key) == {
        "deepseek-v4-int8",
        "deepseek-v4-mxfp4",
        "minimax-m3-fp8",
        "qwen3.8-flash-next",
    }
    expected = {
        "minimax-m3-fp8": (
            "/ppusw/datasets/checkpoints/LLM/T-HEAD/m3/"
            "MiniMax-M3-Quant-W-FP8-PerChannel-A-FP8-PerToken",
            "OAM-890P",
            8,
        ),
        "qwen3.8-flash-next": (
            "/ppusw/datasets/checkpoints/LLM/Qwen/v3.8/"
            "Qwen3.8-Flash-Next",
            "OAM-890P",
            8,
        ),
    }
    for key, (checkpoint, board, tp) in expected.items():
        entry = by_key[key]
        assert entry["checkpoint"] == checkpoint
        assert entry["served_name"] == checkpoint.rsplit("/", 1)[-1]
        assert entry["board"] == board
        assert int(entry["tp"]) == tp
        assert entry["offline_test_path"].endswith("test_model_accuracy.py")
        assert entry["offline_marker"] == "model_e2e"


def test_catalog_carries_both_deepseek_v4_quantizations() -> None:
    """DeepSeek V4 ships two authoritative quantizations; both are catalogued.

    They share the accuracy test file, so single-model dispatch narrows by
    model key. The mxfp4+fp8-channel tree needs fp8 and therefore an 890P
    board; the W8A8-int8 tree is the 810E path.
    """
    catalog = json.loads(MODELS.read_text())
    by_key = {entry["key"]: entry for entry in catalog}
    assert "deepseek-v4-int8" in by_key and "deepseek-v4-mxfp4" in by_key
    int8, mxfp4 = by_key["deepseek-v4-int8"], by_key["deepseek-v4-mxfp4"]
    assert int8["checkpoint"].endswith("DeepSeek-V4-Flash-0731-W8A8-INT8")
    assert mxfp4["checkpoint"].endswith("DeepSeek-V4-Flash-0731-mxfp4-fp8channel")
    assert int8["checkpoint"] != mxfp4["checkpoint"]
    assert int8["offline_test_path"] == mxfp4["offline_test_path"]
    assert int8["offline_test_path"].endswith("test_model_accuracy.py")
    assert mxfp4["board"] == "OAM-890P" and int(mxfp4["tp"]) == 8
    # Golden identity is the last two path segments; they must stay distinct or
    # one variant could be compared against the other's baseline.
    assert "/".join(int8["checkpoint"].split("/")[-2:]) != "/".join(
        mxfp4["checkpoint"].split("/")[-2:]
    )


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


def test_selector_cli_scalar_outputs_guard_single_model_dispatch() -> None:
    """Scalar CLI outputs must narrow correctly for the offline model lane.

    Single-model dispatch exports VLLM_SAIL_MODEL_ROOT from ``model_root``, so
    ``offline_test_path`` must be one model's test file -- a directory would
    collect every model test and run (and with update_golden, re-golden) all
    of them against the single override checkpoint. The nightly (empty input)
    multi-model matrix must instead leave ``model_root`` empty and keep the
    directory default so each test resolves its own hard-coded checkpoint.
    """

    def _scalars(models: str) -> dict:
        result = subprocess.run(
            ["python3", str(SELECT), models],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        out = {}
        for ln in result.stdout.splitlines():
            key, sep, value = ln.partition("=")
            if sep and key != "matrix":
                out[key] = value
        return out

    # Nightly default: 4-model matrix (3 model families, DeepSeek V4 twice for
    # its two quantizations), no global override, directory path.
    nightly = _scalars("")
    assert nightly["model_root"] == ""
    assert nightly["offline_test_path"] == "tests/e2e/models"
    assert nightly["offline_marker"] == "model_e2e"

    # Single-model dispatch: own test file + non-empty checkpoint override.
    catalog = json.loads(MODELS.read_text())
    expected = {entry["key"]: entry for entry in catalog}
    qwen = _scalars("qwen3.8-flash-next")
    assert qwen["offline_test_path"].endswith(".py")
    assert "test_model_accuracy" in qwen["offline_test_path"]
    assert qwen["model_root"] == expected["qwen3.8-flash-next"]["checkpoint"]
    assert qwen["model_root"]

    # Every catalog key narrows to its own file and checkpoint.
    for key, entry in expected.items():
        out = _scalars(key)
        assert out["offline_test_path"] == entry["offline_test_path"], key
        assert out["offline_test_path"].endswith(".py"), key
        assert out["model_root"] == entry["checkpoint"], key


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


def test_e2e_workflow_exposes_smoke_and_accuracy_jobs() -> None:
    text = WORKFLOW.read_text()
    workflow = yaml.safe_load(text)
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_call", "workflow_dispatch"}
    for event in trigger.values():
        assert set(event["inputs"]) == {"build_run_id", "models", "image"}
    assert set(workflow["jobs"]) == {"prepare", "e2e", "accuracy"}
    assert "if" not in workflow["jobs"]["prepare"]
    assert workflow["jobs"]["e2e"]["needs"] == "prepare"
    accuracy = workflow["jobs"]["accuracy"]
    assert accuracy["needs"] == "prepare"
    matrix = accuracy["strategy"]["matrix"]["include"]
    assert {entry["model_key"] for entry in matrix} == {
        "deepseek_v4_int8",
        "deepseek_v4_mxfp4",
        "minimax-m3-fp8",
        "qwen3.8-flash-next",
    }
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


def test_e2e_workflow_pulls_wheels_by_run_id() -> None:
    text = WORKFLOW.read_text()
    # workflow_call/workflow_dispatch still accept the build_run_id input.
    assert "build_run_id:" in text
    assert "actions/download-artifact@v4" in text
    # PR3 keeps the required caller input contract. Both smoke and accuracy
    # download the exact wheel artifact selected by the caller.
    assert text.count("run-id: ${{ inputs.build_run_id }}") == 2
    assert text.count("ppu-wheels-${{ inputs.build_run_id }}") == 2


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
