# SPDX-License-Identifier: Apache-2.0
"""手动设备链路、汇总和未来门禁真值表的 CPU 契约。"""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def runner():
    path = ROOT / "scripts/ci/ppu_device_runner.py"
    spec = importlib.util.spec_from_file_location("device_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "selector,mode,device,complete",
    itertools.product(
        ["success", "failure", "cancelled", "skipped"],
        ["none", "subset", "full", "invalid"],
        ["success", "failure", "cancelled", "skipped"],
        [True, False],
    ),
)
def test_gate_truth_table(runner, selector, mode, device, complete):
    expected = selector == "success" and (
        (mode == "none" and device == "skipped")
        or (mode in ("subset", "full") and device == "success" and complete)
    )
    assert (
        runner.gate_passes(
            selector, mode, device, has_tests=mode != "none", reports_complete=complete
        )
        is expected
    )
    assert (
        not runner.gate_passes(
            selector, mode, device, has_tests=False, reports_complete=complete
        )
        or mode == "none"
    )


@pytest.fixture
def evidence(runner, tmp_path):
    config = json.loads((ROOT / "scripts/ci/ppu_device_tests.json").read_bytes())
    config["test_catalog"] = [
        {
            "id": "sample",
            "test": "tests/e2e/test_sample.py",
            "boards": ["ppu10", "ppu15"],
            "requires": ["torch", "triton", "vllm", "vllm_sail"],
            "board_overrides": {},
        }
    ]
    config_bytes = json.dumps(config).encode()
    env_bytes = (ROOT / "scripts/ci/ppu_device_environment.json").read_bytes()
    image = "registry/image@sha256:" + "a" * 64
    fingerprint = dict(
        sdk_sha256="a" * 64,
        torch_sha256="b" * 64,
        pytorch_sail_arch=["ppu_10", "ppu_15"],
        build_image=image,
        vllm_commit="c" * 40,
    )
    selection = {
        "schema": 1,
        "mode": "full",
        "reasons": ["forced_full"],
        "matched_rule_ids": [],
        "changed_files": [],
        "unmapped_files": [],
        "test_ids": ["sample"],
        "files": ["tests/e2e/test_sample.py"],
        **runner.select.route_tests(config, ["sample"]),
        "base_sha": None,
        "head_sha": None,
        "tested_sha": "d" * 40,
        "build_run_id": "123",
        "config_sha256": runner.select.checksum(config_bytes),
        "environment_sha256": runner.select.checksum(env_bytes),
        "build_fingerprint": fingerprint,
        "runtime_image": image,
    }
    raw = json.dumps(selection).encode()
    dirs = []
    for group in selection["groups"]:
        directory = tmp_path / group["group_id"]
        directory.mkdir()
        dirs.append(directory)
        identity = dict(
            run_id="123",
            run_attempt="1",
            tested_sha="d" * 40,
            board=group["board"],
            group_id=group["group_id"],
        )
        profile = dict(
            capability=[8, 0],
            sdk_identity="2.2",
            driver_identity="1.2",
            packages=dict(torch="2", triton="3", pla="1"),
            runtime_env={"VLLM_SAIL_USE_PLA": True},
        )
        documents = {
            "selection.json": selection,
            "effective-selection.json": runner.effective_selection(
                config, selection, raw, group["board"], True
            ),
            "environment.json": {
                "schema": 1,
                **identity,
                "profile": profile,
                "expanded": True,
                "expansion_reason": "runtime_environment_changed",
                "runtime_image": image,
            },
            "summary.json": {
                "schema": 1,
                **identity,
                "status": "success",
                "failure_stage": None,
                "reason": None,
            },
            "wheel-manifest.json": {
                "schema": 1,
                "commits": {"vllm": "c" * 40, "vllm-sail": "d" * 40},
            },
            "timings.json": {
                "schema": 1,
                **identity,
                "config_sha256": selection["config_sha256"],
                "build_fingerprint": fingerprint,
                "runtime_image": image,
                "nproc_per_node": 1,
                "phases": {
                    k: 1
                    for k in (
                        "queue_seconds",
                        "install_seconds",
                        "preflight_seconds",
                        "device_occupied_seconds",
                        "total_seconds",
                    )
                },
                "files": [
                    {
                        "test_id": "sample",
                        "file": "tests/e2e/test_sample.py",
                        "duration_seconds": 1,
                        "outcome": "success",
                    }
                ],
            },
        }
        for name, value in documents.items():
            (directory / name).write_text(json.dumps(value))
        diagnostics = {
            **fingerprint,
            "pytorch_sail_arch": "ppu_15;ppu_10",
            "vllm_sail_commit": "d" * 40,
            "status": "succeeded",
        }
        (directory / "build-manifest.txt").write_text(
            "".join(f"{k}={v}\n" for k, v in diagnostics.items())
        )
        for name in ("logs", "nodes", "junit"):
            (directory / name).mkdir()
        node = "tests/e2e/test_sample.py::test_ok"
        report = dict(
            schema=1,
            board=group["board"],
            file="tests/e2e/test_sample.py",
            status="success",
            exitstatus=0,
            errors=[],
            collected=[node],
            applicable=[node],
            hardware_excluded=[],
            nodes={
                node: {
                    phase: dict(outcome="passed", wasxfail=None, duration_seconds=1)
                    for phase in ("setup", "call", "teardown")
                }
            },
            runtime={
                "isolated_root": "/tmp/isolated",
                "checkout_root": "/workspace/source",
                "runtime_env": {"VLLM_SAIL_USE_PLA": True},
                "paths": {
                    **{
                        name: "/site-packages/" + name.replace(".", "/") + ".py"
                        for name in runner.load(
                            ROOT / "tests/support/ppu_ci.py"
                        ).RUNTIME_MODULES
                        | {"triton"}
                    },
                    "plugin": "/tmp/isolated/tests/support/ppu_ci.py",
                    "conftest": "/tmp/isolated/tests/conftest.py",
                },
            },
        )
        (directory / "nodes/sample.json").write_text(json.dumps(report))
        (directory / "junit/sample.xml").write_text(
            '<testsuites><testsuite><testcase classname="tests.e2e.test_sample" name="test_ok"/></testsuite></testsuites>'
        )
        (directory / "logs/sample.log").write_text("pytest output")
    return config, raw, dirs


def test_summary_requires_all_groups(runner, evidence):
    config, raw, dirs = evidence
    runner.validate_results(config, raw, dirs, "123", "1")
    for broken in (dirs[:1], dirs + dirs[:1]):
        with pytest.raises(ValueError):
            runner.validate_results(config, raw, broken, "123", "1")


@pytest.mark.parametrize(
    "name,mutation",
    [
        ("summary.json", lambda s: s.update(run_attempt="2")),
        ("summary.json", lambda s: s.update(status="failure")),
        ("summary.json", lambda s: s.update(board="ppu15")),
        ("effective-selection.json", lambda s: s["groups"][0]["files"].clear()),
        (
            "effective-selection.json",
            lambda s: s["groups"].append(copy.deepcopy(s["groups"][0])),
        ),
        ("timings.json", lambda s: s["files"].clear()),
        ("nodes/sample.json", lambda s: s["nodes"].clear()),
        ("environment.json", lambda s: s.update(runtime_image="wrong")),
        ("wheel-manifest.json", lambda s: s["commits"].update(vllm="e" * 40)),
    ],
)
def test_incomplete_or_stale_group_rejected(runner, evidence, name, mutation):
    config, raw, dirs = evidence
    path = dirs[0] / name
    document = json.loads(path.read_bytes())
    mutation(document)
    path.write_text(json.dumps(document))
    with pytest.raises((ValueError, KeyError)):
        runner.validate_results(config, raw, dirs, "123", "1")


def test_summary_cli_rejects_extra_partial_artifact(
    runner, evidence, tmp_path, monkeypatch
):
    config, raw, _ = evidence
    orphan = tmp_path / "unknown-artifact"
    orphan.mkdir()
    (orphan / "build-manifest.txt").write_text("only diagnostics")
    monkeypatch.setattr(
        runner, "read_inputs", lambda env: (raw, None, config, None, None)
    )
    monkeypatch.setenv("CI_RUN_ID", "123")
    monkeypatch.setenv("CI_RUN_ATTEMPT", "1")
    monkeypatch.setattr(
        runner.sys, "argv", ["runner", "summary", "--results", str(tmp_path)]
    )
    with pytest.raises((ValueError, OSError)):
        runner.main()


def test_manual_workflow_does_not_change_pr_gate():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/build-ppu-wheels.yaml").read_text()
    )
    jobs = workflow["jobs"]
    assert jobs["gate"]["needs"] == ["build", "pin-unit-tests", "smoke"]
    assert "needs" not in jobs["build"]
    assert "detect-device-changes" in jobs
    assert "workflow_dispatch" in jobs["detect-device-changes"]["if"]
    assert "existing_build_run_id" in jobs["detect-device-changes"]["if"]
    assert "schedule" not in workflow.get(True, workflow.get("on", {}))
    assert jobs["device-tests"]["uses"] == "./.github/workflows/ppu-device-tests.yaml"
    assert (
        jobs["device-build"]["container"]["image"]
        == "${{ needs.detect-device-changes.outputs.build_image }}"
    )
    assert (
        jobs["device-build"]["env"]["VLLM_SAIL_BUILD_IMAGE"]
        == jobs["device-build"]["container"]["image"]
    )


@pytest.mark.parametrize(
    "owner,run_id,attempt",
    [
        ("t-head", "12345678901", "1"),
        ("t-head", "9" * 19, "9"),
        ("Long_Organization_Name_123", "9" * 13, "9"),
    ],
)
def test_scheduler_names_retain_unique_group_suffix(runner, owner, run_id, attempt):
    groups = [{"group_id": "ppu10-0"}, {"group_id": "ppu15-0"}]
    names = runner.scheduler_names(groups, owner, run_id, attempt)
    normalized_owner = "t-head" if owner == "t-head" else "long-organization-na"
    assert names == [
        f"ppu-{normalized_owner}-{run_id}-{attempt}-dev-{g['group_id']}" for g in groups
    ]
    assert len(set(names)) == 2
    assert all(len(name) <= 52 for name in names)


def test_prepare_rejects_scheduler_name_truncation_before_nas(
    runner, evidence, monkeypatch
):
    config, raw, _ = evidence
    monkeypatch.setattr(
        runner, "read_inputs", lambda env: (raw, None, config, None, None)
    )
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "t-head")
    monkeypatch.setenv("CI_RUN_ID", "9" * 40)
    monkeypatch.setenv("CI_RUN_ATTEMPT", "1")
    monkeypatch.setenv("DEVICE_GROUP_ID", "ppu10-0")
    monkeypatch.setattr(runner.sys, "argv", ["runner", "prepare"])

    def unexpected_nas(_):
        pytest.fail("作业名不合法时不应准备 NAS 或派发设备")

    monkeypatch.setattr(runner, "require_nas", unexpected_nas)
    with pytest.raises(ValueError, match="作业名"):
        runner.main()


def test_reusable_workflow_contract():
    path = ROOT / ".github/workflows/ppu-device-tests.yaml"
    assert path.is_file(), "缺少 reusable workflow"
    workflow = yaml.safe_load(path.read_text())
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    jobs = workflow["jobs"]
    device = jobs["device-tests"]
    assert device["runs-on"] == "k8s-runner-group-cpu-thead"
    assert device["container"]["image"] == "${{ inputs.image }}"
    assert device["timeout-minutes"] == 90
    assert device["strategy"]["fail-fast"] is False
    assert device["strategy"]["max-parallel"] == 2
    scheduler = next(
        s for s in device["steps"] if "ppu-scheduler-action" in s.get("uses", "")
    )
    assert scheduler["uses"] == (
        "t-head/ppu-scheduler-action@a4e03cbbdb2624f4871fff30d7081b34cbc4b7d4"
    )
    assert scheduler["with"]["timeout_minutes"] == 60
    assert scheduler["with"]["job_suffix"] == "dev-${{ matrix.group_id }}"
    assert (
        scheduler["with"]["node_selector"]
        == "${{ steps.prepare.outputs.node_selector }}"
    )
    assert scheduler["with"]["cleanup_policy"] == "always"
    uploads = [s for s in device["steps"] if "upload-artifact" in s.get("uses", "")]
    assert all(
        s["if"] == "always()" and s["with"]["if-no-files-found"] == "error"
        for s in uploads
    )
    assert all(s["with"]["retention-days"] == 14 for s in uploads)
    assert all("/wl_nas/devops/" in s["with"]["path"] for s in uploads)
    assert jobs["device-summary"]["if"] == "always()"
    assert "continue-on-error" not in path.read_text()
