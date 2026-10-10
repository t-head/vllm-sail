# SPDX-License-Identifier: Apache-2.0
"""Device CI regressions for isolation, node evidence, and fail-closed behavior using fake CPU-installed packages."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "tests/support/ppu_ci.py"
RUNNER = ROOT / "scripts/ci/ppu_device_runner.py"


@pytest.fixture
def runner():
    assert RUNNER.is_file(), "缺少设备执行器"
    return load(RUNNER)


@pytest.mark.parametrize(
    "available,count",
    [(False, 0), (False, 2), (True, 1), (True, 16), ("error", 1), (True, "error")],
)
def test_device_observation_keeps_independent_results(available, count):
    from types import SimpleNamespace

    calls = []

    def query(name, value):
        calls.append(name)
        if value == "error":
            raise RuntimeError(name)
        return value

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: query("is_available", available),
            device_count=lambda: query("device_count", count),
        )
    )
    plugin = load(PLUGIN)
    assert hasattr(plugin, "observe_devices"), "缺少独立设备采样"
    observation = plugin.observe_devices(torch)
    assert calls == ["is_available", "device_count"]
    for name, value in (("is_available", available), ("device_count", count)):
        if value == "error":
            assert observation[name] == {"error": f"RuntimeError: {name}"}
        else:
            assert observation[name] == {"value": value}


def test_runtime_probe_preserves_strict_single_device_check(tmp_path):
    result, _ = run_case(tmp_path, "def test_ok(): pass\n")
    assert result.returncode == 0
    torch_file = tmp_path / "site-packages/torch/__init__.py"
    torch_file.write_text(
        torch_file.read_text().replace(
            "device_count=lambda:1", "device_count=lambda:16"
        )
    )
    command = f"""
import sys, json
sys.path.insert(0, {str(tmp_path / "site-packages")!r})
from tests.support.ppu_ci import inspect_runtime
observed = {{}}
try:
    inspect_runtime({str(tmp_path / "isolated")!r}, {str(ROOT)!r}, [], device_observation=observed)
except Exception as error:
    print(json.dumps(dict(observed=observed, error=str(error))))
"""
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path / "isolated",
        env={**os.environ, "VLLM_SAIL_USE_PLA": "1"},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(completed.stdout)
    assert data["observed"] == {
        "is_available": {"value": True},
        "device_count": {"value": 16},
    }
    assert data["error"] == "需要一张可用 PPU"


def test_runner_entrypoint():
    assert RUNNER.is_file(), "缺少设备执行器"
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts/ci/ppu_device_test.sh")], capture_output=True
    )
    assert result.returncode == 0


def test_run_cli_cannot_succeed_without_inputs():
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("CI_RUN", "DEVICE_", "EXPECTED_", "SELECTION_"))
    }
    result = subprocess.run(
        [sys.executable, str(RUNNER), "run"], env=env, capture_output=True
    )
    assert result.returncode != 0


def test_observe_environment_rejects_unidentified_driver(runner):
    runtime = {"capability": [8, 0], "runtime_env": {"VLLM_SAIL_USE_PLA": True}}
    versions = {"torch": "2", "triton": "3", "pla": "1"}
    profile = runner.runtime_profile(
        runtime, versions, "HGCC 2.2.0", "Driver Version: 1.2.3"
    )
    assert profile["driver_identity"] == "1.2.3"
    with pytest.raises(ValueError):
        runner.runtime_profile(runtime, versions, "HGCC 2.2.0", "no driver identity")
    with pytest.raises(ValueError):
        runner.runtime_profile(
            runtime,
            {**versions, "pla": "unknown"},
            "HGCC 2.2.0",
            "Driver Version: 1.2.3",
        )


def test_isolated_tree_and_environment(runner, tmp_path):
    isolated = tmp_path / "isolated"
    runner.copy_test_tree(ROOT, isolated)
    assert (isolated / "tests/support/ppu_ci.py").is_file()
    assert (isolated / "tests/__init__.py").is_file()
    assert (isolated / "pyproject.toml").read_bytes() == (
        ROOT / "pyproject.toml"
    ).read_bytes()
    assert not (isolated / "tests/ut").exists()
    assert not (isolated / "vllm_sail").exists()
    env = runner.clean_environment(
        {
            "PYTHONPATH": "bad",
            "PYTEST_ADDOPTS": "-k bad",
            "VLLM_VERSION": "bad",
            "VLLM_PPU_USE_PLA": "0",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_SAIL": "bad",
        },
        ROOT,
        isolated,
    )
    for name in (
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "VLLM_VERSION",
        "VLLM_PPU_USE_PLA",
        "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_SAIL",
    ):
        assert name not in env
    assert (
        env["VLLM_SAIL_USE_PLA"] == "1" and env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    )


@pytest.mark.parametrize(
    "run,attempt,group",
    [
        ("bad", "1", "ppu10-0"),
        ("123", "0", "ppu10-0"),
        ("123", "1", "../x"),
        ("123", "1", "ppu10-1"),
    ],
)
def test_nas_identity_rejects_invalid_values(runner, run, attempt, group):
    with pytest.raises(ValueError):
        runner.result_path(run, attempt, group, pod=True)


def test_nas_prefix_and_missing_mount(runner, monkeypatch):
    assert (
        str(runner.result_path("123", "2", "ppu10-0", pod=True))
        == "/mnt/wl_nas/devops/123-2/device-tests/ppu10-0"
    )
    assert (
        str(runner.result_path("123", "2", "ppu10-0", pod=False))
        == "/wl_nas/devops/123-2/device-tests/ppu10-0"
    )
    monkeypatch.setattr(os.path, "ismount", lambda p: False)
    with pytest.raises(ValueError, match="NAS"):
        runner.require_nas(Path("/mnt/wl_nas"))


def test_runtime_expansion_preserves_other_board(runner):
    config = json.loads((ROOT / "scripts/ci/ppu_device_tests.json").read_bytes())
    selector = load(ROOT / "scripts/ci/ppu_device_select.py")
    original = {
        "mode": "subset",
        "test_ids": ["kda"],
        "files": ["tests/e2e/fork_port/test_kda.py"],
        **selector.route_tests(config, ["kda"]),
    }
    effective = runner.effective_selection(config, original, b"original", "ppu10", True)
    assert effective["expanded_boards"] == ["ppu10"]
    assert len(effective["groups"][0]["files"]) == 12
    assert effective["groups"][1] == original["groups"][1]
    assert original["test_ids"] == ["kda"]


@pytest.mark.parametrize(
    "damage", ["xml", "json", "empty", "wrong-board", "call-missing", "xml-skip"]
)
def test_incomplete_file_evidence_rejected(runner, tmp_path, damage):
    result, report = run_case(tmp_path, "def test_ok(): pass\n")
    assert result.returncode == 0
    output = tmp_path / "results"
    for kind in ("logs", "nodes", "junit"):
        (output / kind).mkdir(parents=True)
    (output / "logs/sample.log").write_text(result.stdout)
    shutil.copyfile(tmp_path / "junit.xml", output / "junit/sample.xml")
    node_path = output / "nodes/sample.json"
    node_path.write_text(json.dumps(report))
    entry = json.loads((tmp_path / "config.json").read_bytes())["test_catalog"][0]
    runner.validate_file_evidence(output, entry, "ppu10")
    if damage == "xml":
        (output / "junit/sample.xml").rename(output / "junit/other.xml")
    elif damage == "json":
        node_path.rename(output / "nodes/other.json")
    elif damage == "empty":
        node_path.write_text("{}")
    elif damage == "xml-skip":
        path = output / "junit/sample.xml"
        path.write_text(
            path.read_text().replace("</testcase>", "<skipped/></testcase>")
            if "</testcase>" in path.read_text()
            else path.read_text().replace(" />", "><skipped/></testcase>", 1)
        )
    else:
        if damage == "wrong-board":
            report["board"] = "ppu15"
        else:
            report["nodes"][report["applicable"][0]].pop("call")
        node_path.write_text(json.dumps(report))
    with pytest.raises((ValueError, KeyError, OSError)):
        runner.validate_file_evidence(output, entry, "ppu10")


def test_file_failure_does_not_stop_remaining_files(runner, tmp_path):
    result, _ = run_case(tmp_path, "def test_ok(): pass\n")
    assert result.returncode == 0
    isolated = tmp_path / "isolated"
    (isolated / "tests/e2e/test_first.py").write_text("def test_fail(): assert False\n")
    (isolated / "tests/e2e/test_second.py").write_text("def test_pass(): pass\n")
    entries = [
        {
            "id": n,
            "test": f"tests/e2e/test_{n}.py",
            "boards": ["ppu10"],
            "requires": ["torch", "vllm", "vllm_sail"],
            "board_overrides": {},
        }
        for n in ("first", "second")
    ]
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"test_catalog": entries}))
    env = runner.clean_environment(os.environ, ROOT, isolated)
    # Only the fixture supplies fake installed packages.
    env["PYTHONPATH"] = str(tmp_path / "site-packages")
    ticks = iter(range(20))
    output = tmp_path / "results"
    output.mkdir()
    reports = runner.run_files(
        entries, "ppu10", config, isolated, output, env, clock=lambda: next(ticks)
    )
    assert [r["outcome"] for r in reports] == ["failure", "success"]
    assert all(r["duration_seconds"] == 1 for r in reports)
    assert (output / "nodes/second.json").is_file()


@pytest.fixture
def valid_inputs(runner, tmp_path):
    from tests.ut.test_ppu_wheel_manifest import make_wheels

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    make_wheels(wheels)
    manifest_tool = runner.select.load_sibling("ppu_wheel_manifest")
    manifest = {
        "schema": 1,
        "commits": {"vllm": "c" * 40, "vllm-sail": "d" * 40},
        "wheels": manifest_tool.wheel_records(wheels),
    }
    runner.select.write_json(wheels / "wheel-manifest.json", manifest)
    config_path = tmp_path / "ppu_device_tests.json"
    env_path = tmp_path / "ppu_device_environment.json"
    shutil.copyfile(ROOT / "scripts/ci/ppu_device_tests.json", config_path)
    shutil.copyfile(ROOT / "scripts/ci/ppu_device_environment.json", env_path)
    image = "registry/image@sha256:" + "a" * 64
    diagnostics = dict(
        sdk_sha256="a" * 64,
        torch_sha256="b" * 64,
        pytorch_sail_arch="ppu_15;ppu_10",
        build_image=image,
        vllm_commit="c" * 40,
        vllm_sail_commit="d" * 40,
        status="succeeded",
    )
    build = tmp_path / "build-manifest.txt"
    build.write_text("".join(f"{k}={v}\n" for k, v in diagnostics.items()))
    changes = runner.select.collect_changes(ROOT, None, None, "d" * 40)
    selection = runner.select.make_selection(
        config_bytes=config_path.read_bytes(),
        environment_bytes=env_path.read_bytes(),
        changes=changes,
        diagnostics=diagnostics,
        wheel_manifest=manifest,
        build_run_id="123",
        build_image=image,
        runtime_image=image,
        expected_vllm_commit="c" * 40,
        force_full=True,
    )
    selection_path = tmp_path / "selection.json"
    runner.select.write_json(selection_path, selection)
    return dict(
        WHEELS_DIR=str(wheels),
        BUILD_MANIFEST_FILE=str(build),
        SELECTION_FILE=str(selection_path),
        DEVICE_BOARD="ppu10",
        DEVICE_GROUP_ID="ppu10-0",
        CI_RUN_ID="123",
        CI_RUN_ATTEMPT="1",
        DEVICE_RESULTS_DIR=str(runner.result_path("123", "1", "ppu10-0", pod=True)),
        EXPECTED_SAIL_COMMIT="d" * 40,
        EXPECTED_VLLM_COMMIT="c" * 40,
        EXPECTED_BUILD_RUN_ID="123",
        EXPECTED_BUILD_IMAGE=image,
        RUNTIME_IMAGE=image,
        QUALIFICATION_MODE="true",
        DISPATCH_MATRIX=json.dumps(selection["matrix"]),
    )


def test_inputs_verify_real_wheel_archives(runner, valid_inputs):
    raw, _, config, _, manifest = runner.read_inputs(valid_inputs)
    assert len(json.loads(raw)["files"]) == len(config["test_catalog"]) == 14
    assert len(manifest["wheels"]) == 2


@pytest.mark.parametrize(
    "matrix",
    [
        {"include": []},
        {"include": [{"group_id": "ppu10-0", "board": "ppu10", "nproc_per_node": 2}]},
    ],
)
def test_dispatch_matrix_must_equal_selection(runner, valid_inputs, matrix):
    valid_inputs["DISPATCH_MATRIX"] = json.dumps(matrix)
    with pytest.raises(ValueError, match="矩阵"):
        runner.read_inputs(valid_inputs)


@pytest.mark.parametrize(
    "name,value",
    [
        ("CI_RUN_ATTEMPT", "0"),
        ("EXPECTED_BUILD_RUN_ID", "122"),
        ("DEVICE_GROUP_ID", "ppu15-0"),
        ("DEVICE_RESULTS_DIR", "/tmp/results"),
        ("RUNTIME_IMAGE", "registry/image@sha256:" + "b" * 64),
        ("EXPECTED_SAIL_COMMIT", "a" * 40),
        ("QUALIFICATION_MODE", "false"),
    ],
)
def test_invalid_execution_inputs_fail_before_install(
    runner, valid_inputs, name, value
):
    valid_inputs[name] = value
    with pytest.raises(ValueError):
        runner.read_inputs(valid_inputs)


def test_changed_wheel_is_rejected_before_install(runner, valid_inputs):
    wheel = next(Path(valid_inputs["WHEELS_DIR"]).rglob("vllm-*.whl"))
    with wheel.open("ab") as stream:
        stream.write(b"changed bytes")
    with pytest.raises(ValueError, match="hashes"):
        runner.read_inputs(valid_inputs)


def test_selection_cli_with_verified_wheels(runner, valid_inputs, tmp_path):
    changes_path = tmp_path / "changes.json"
    runner.select.write_json(
        changes_path, runner.select.collect_changes(ROOT, None, None, "d" * 40)
    )
    output, github_output = tmp_path / "selected.json", tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/ci/ppu_device_select.py"),
            "select",
            "--repo",
            str(ROOT),
            "--config",
            str(tmp_path / "ppu_device_tests.json"),
            "--environment",
            str(tmp_path / "ppu_device_environment.json"),
            "--changes",
            str(changes_path),
            "--build-manifest",
            valid_inputs["BUILD_MANIFEST_FILE"],
            "--wheel-manifest",
            str(Path(valid_inputs["WHEELS_DIR"]) / "wheel-manifest.json"),
            "--build-run-id",
            "123",
            "--build-image",
            valid_inputs["EXPECTED_BUILD_IMAGE"],
            "--runtime-image",
            valid_inputs["RUNTIME_IMAGE"],
            "--expected-vllm-commit",
            "c" * 40,
            "--scope",
            "full",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_OUTPUT": str(github_output)},
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(output.read_bytes())["files"]) == 14
    assert "has_tests=true" in github_output.read_text()


@pytest.mark.parametrize("pip_index", [None, "", "https://packages.example/simple/"])
@pytest.mark.parametrize(
    "failure", [None, "install", "pytest", "preflight", "missing-report"]
)
def test_execute_only_publishes_success_after_complete_evidence(
    runner, valid_inputs, tmp_path, monkeypatch, failure, pip_index
):
    # Replace only device/NAS/install boundaries; run real pytest subprocesses and all evidence checks.
    case = tmp_path / "case"
    case.mkdir()
    result, _ = run_case(case, "def test_ok(): pass\n")
    assert result.returncode == 0
    repo = case / "isolated"
    (repo / "tests/e2e/test_first.py").write_text(
        "def test_first(): "
        + ("assert False" if failure == "pytest" else "pass")
        + "\n"
    )
    (repo / "tests/e2e/test_second.py").write_text("def test_second(): pass\n")
    entries = [
        {
            "id": n,
            "test": f"tests/e2e/test_{n}.py",
            "boards": ["ppu10"],
            "requires": ["torch", "triton", "vllm", "vllm_sail"],
            "board_overrides": {},
        }
        for n in ("first", "second")
    ]
    raw, config_path, config, baseline, manifest = runner.read_inputs(valid_inputs)
    config["test_catalog"] = entries
    config_path.write_text(json.dumps(config))
    original = json.loads(raw)
    original.update(
        test_ids=[t["id"] for t in entries],
        files=[t["test"] for t in entries],
        **runner.select.route_tests(config, [t["id"] for t in entries]),
    )
    raw = json.dumps(original).encode()
    output = tmp_path / "nas"
    output.mkdir()
    probe = dict(run_id="123", run_attempt="1", group_id="ppu10-0", nonce="test")
    runner.select.write_json(output / "cpu-probe.json", probe)
    env = {
        **os.environ,
        **valid_inputs,
        "DEVICE_RESULTS_DIR": str(output),
        "PIP_INDEX_URL": "https://unreachable.example/simple/",
    }
    env.pop("SAIL_PIP_INDEX_URL", None)
    if pip_index is not None:
        env["SAIL_PIP_INDEX_URL"] = pip_index
    monkeypatch.setattr(runner, "ROOT", repo)
    monkeypatch.setattr(runner, "result_path", lambda *a, **kw: output)
    monkeypatch.setattr(runner, "require_nas", lambda mount: None)
    monkeypatch.setattr(
        runner,
        "read_inputs",
        lambda env: (raw, config_path, config, baseline, manifest),
    )
    original_clean = runner.clean_environment

    def child_environment(env, repo, isolated):
        child = original_clean(env, repo, isolated)
        child["PYTHONPATH"] = str(case / "site-packages")
        return child

    monkeypatch.setattr(runner, "clean_environment", child_environment)
    original_run = subprocess.run
    installed = []

    def device_boundary(command, **kwargs):
        if command[1:3] == ["-m", "pip"]:
            installed.append(command)
            if failure == "install" and "-r" in command:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        if command[1] == "-c":
            if failure == "preflight":
                raise subprocess.CalledProcessError(1, command)
            observed = dict(
                runtime={
                    "capability": [8, 0],
                    "runtime_env": {"VLLM_SAIL_USE_PLA": True},
                },
                packages=dict(torch="2", triton="3", pla="1"),
                sdk_output="HGCC 2.2",
                smi_output="Driver Version: 1.2.3",
            )
            runner.select.write_json(
                Path(kwargs["env"]["PPU_CI_OBSERVATION"]), observed
            )
            return subprocess.CompletedProcess(command, 0)
        completed = original_run(command, **kwargs)
        if failure == "missing-report" and "tests/e2e/test_first.py" in command:
            (output / "nodes/first.json").rename(output / "nodes/unexpected.json")
        return completed

    monkeypatch.setattr(subprocess, "run", device_boundary)
    code = runner.execute(env)
    summary = json.loads((output / "summary.json").read_bytes())
    assert code == (0 if failure is None else 1)
    assert summary["status"] == ("success" if failure is None else "failure")
    assert "--no-deps" in installed[0] and "--force-reinstall" in installed[0]
    assert len(installed[0][installed[0].index("--force-reinstall") + 1 :]) == 2
    assert installed[1] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--index-url",
        pip_index or "https://mirrors.aliyun.com/pypi/simple/",
        "-r",
        str(repo / "requirements/dev.txt"),
    ]
    assert len(installed) == 2  # No device-library installs or upgrades.
    if failure not in {"preflight", "install"}:
        assert (
            json.loads((output / "nodes/second.json").read_bytes())["status"]
            == "success"
        )
    else:
        assert summary["failure_stage"] == failure
        assert not (output / "nodes").exists()
        if failure == "install":
            assert not (output / "preflight.log").exists()


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_install(site):
    contents = {
        "torch/__init__.py": "from types import SimpleNamespace\n__version__='1'\ncuda=SimpleNamespace(is_available=lambda:True, device_count=lambda:1, get_device_capability=lambda i:(8,0))\n",
        "vllm/__init__.py": "",
        "vllm/platforms/__init__.py": "from types import SimpleNamespace\ncurrent_platform=SimpleNamespace(is_ppu=lambda:True)\n",
        "vllm_sail/__init__.py": "",
        "vllm_sail/envs.py": "import os\nVLLM_SAIL_USE_PLA=VLLM_PPU_USE_PLA=os.environ.get('VLLM_SAIL_USE_PLA')=='1'\n",
        "vllm_sail/native/__init__.py": "",
        "vllm_sail/native/extensions.py": "def import_kernels(*, strict=False):\n    assert strict\n    return ('fake-provider',)\n",
        "triton/__init__.py": "__version__='1'\n",
        "pla/__init__.py": "__version__='1'\n",
    }
    for name in ("_C", "_moe_C", "_upstream_C", "_upstream_moe_C"):
        contents[f"vllm_sail/{name}.py"] = ""
    for name, text in contents.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def run_case(
    tmp_path,
    text,
    *,
    overrides=None,
    board="ppu10",
    real_device=True,
    pla="1",
    shadow=False,
    extra_args=(),
):
    assert PLUGIN.is_file(), "缺少显式 pytest CI 插件"
    isolated, site = tmp_path / "isolated", tmp_path / "site-packages"
    for name in ("tests", "tests/support", "tests/e2e"):
        path = isolated / name
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").write_text("")
    shutil.copyfile(PLUGIN, isolated / "tests/support/ppu_ci.py")
    (isolated / "tests/conftest.py").write_text(f"HAS_REAL_DEVICE = {real_device!r}\n")
    file = "tests/e2e/test_sample.py"
    (isolated / file).write_text(text)
    (isolated / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath=["."]\n'
    )
    fake_install(isolated if shadow else site)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "test_catalog": [
                    {
                        "id": "sample",
                        "test": file,
                        "boards": ["ppu10", "ppu15"],
                        "board_overrides": overrides or {},
                        "requires": ["torch", "triton", "vllm", "vllm_sail"],
                    }
                ]
            }
        )
    )
    report = tmp_path / "nodes.json"
    args = [
        "-c",
        "pyproject.toml",
        "-p",
        "tests.support.ppu_ci",
        file,
        "--ppu-ci-config",
        str(config),
        "--ppu-ci-board",
        board,
        "--ppu-ci-report",
        str(report),
        "--junitxml",
        str(tmp_path / "junit.xml"),
        *extra_args,
    ]
    command = f"import sys; sys.path.insert(0, {str(site)!r}); import pytest; sys.exit(pytest.main({args!r}))"
    env = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "VLLM_SAIL_USE_PLA": pla,
        "PPU_CI_CHECKOUT": str(ROOT),
        "PPU_CI_ISOLATED_ROOT": str(isolated),
    }
    env.pop("PYTHONPATH", None)
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, "-c", command],
        cwd=isolated,
        env=env,
        capture_output=True,
        text=True,
    )
    assert report.is_file(), result.stdout + result.stderr
    return result, json.loads(report.read_text())


@pytest.mark.parametrize(
    "text,success",
    [
        ("def test_ok(): pass\n", True),
        ("def test_bad(): assert False\n", False),
        (
            "import pytest\ndef test_mock(): pass\ndef test_numeric(): pytest.skip('missing SDK')\n",
            False,
        ),
        ("import pytest\npytest.skip('module', allow_module_level=True)\n", False),
        ("import pytest\npytest.importorskip('not_a_real_sdk_package')\n", False),
        ("import pytest\n@pytest.mark.xfail\ndef test_bad(): assert False\n", False),
        ("import pytest\n@pytest.mark.xfail\ndef test_ok(): pass\n", False),
        (
            "import pytest\n@pytest.fixture\ndef bad(): assert False\ndef test_bad(bad): pass\n",
            False,
        ),
        (
            "import pytest\n@pytest.fixture\ndef bad():\n yield\n assert False\ndef test_bad(bad): pass\n",
            False,
        ),
        ("", False),
        ("raise RuntimeError('collection')\n", False),
    ],
)
def test_pytest_node_guard(tmp_path, text, success):
    result, report = run_case(tmp_path, text)
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert (report["status"] == "success") is success
    if success:
        assert report["applicable"] == ["tests/e2e/test_sample.py::test_ok"]
        assert report["nodes"][report["applicable"][0]]["call"]["outcome"] == "passed"
    if "missing SDK" in text:
        assert "missing SDK" in json.dumps(report)


@pytest.mark.parametrize("damage", ["missing", "source-path", "conftest-path", "pla"])
def test_node_evidence_rechecks_installation_paths(tmp_path, damage):
    result, report = run_case(tmp_path, "def test_ok(): pass\n")
    assert result.returncode == 0
    runtime = report["runtime"]
    if damage == "missing":
        runtime["paths"].pop("vllm_sail._C")
    elif damage == "source-path":
        runtime["paths"]["vllm"] = str(ROOT / "vllm/__init__.py")
    elif damage == "conftest-path":
        runtime["paths"]["conftest"] = "/other/tests/conftest.py"
    else:
        runtime["runtime_env"]["VLLM_SAIL_USE_PLA"] = False
    entry = json.loads((tmp_path / "config.json").read_bytes())["test_catalog"][0]
    with pytest.raises(ValueError):
        load(PLUGIN).validate_node_report(report, entry, "ppu10")


def test_hardware_exclusion_is_not_pass(tmp_path):
    result, report = run_case(
        tmp_path,
        "def test_general(): pass\ndef test_fp8(): assert False\n",
        overrides={"test_fp8": ["ppu15"]},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(report["collected"]) == 2
    assert len(report["hardware_excluded"]) == 1
    assert len(report["applicable"]) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"overrides": {"test_typo": ["ppu15"]}},
        {"real_device": False},
        {"pla": "0"},
        {"shadow": True},
        {"extra_args": ("-k", "general")},
    ],
)
def test_runtime_or_collection_violation(tmp_path, options):
    result, report = run_case(
        tmp_path, "def test_general(): pass\ndef test_other(): pass\n", **options
    )
    assert result.returncode != 0
    assert report["status"] == "failure"
    assert report["errors"]


def load_diagnostic():
    path = ROOT / "scripts/ci/ppu_device_probe.py"
    assert path.is_file(), "缺少独立设备诊断入口"
    return load(path)


@pytest.mark.parametrize("damage", [None, "wheel", "sha", "image", "run"])
def test_probe_verifies_historical_artifacts(valid_inputs, damage):
    diagnostic = load_diagnostic()
    env = {**valid_inputs, "CI_RUN_ID": "456", "PROBE_CODE_SHA": "e" * 40}
    if damage == "wheel":
        wheel = next(Path(env["WHEELS_DIR"]).rglob("vllm-*.whl"))
        with wheel.open("ab") as stream:
            stream.write(b"changed")
    elif damage == "sha":
        env["EXPECTED_SAIL_COMMIT"] = "f" * 40
    elif damage == "image":
        env["RUNTIME_IMAGE"] = "registry/other@sha256:" + "b" * 64
    elif damage == "run":
        env["EXPECTED_BUILD_RUN_ID"] = "122"
    if damage:
        with pytest.raises(ValueError):
            diagnostic.verify_inputs(env)
    else:
        identity, config, manifest = diagnostic.verify_inputs(env)
        assert identity["build_run_id"] == "123"
        assert identity["run_id"] == "456"
        assert identity["wheel_sail_commit"] == "d" * 40
        assert identity["probe_code_sha"] == "e" * 40
        assert identity["qualification"] is False
        assert config["test_catalog"] and len(manifest["wheels"]) == 2


@pytest.mark.parametrize("mode", ["preflight", "torch-only"])
@pytest.mark.parametrize("failure", [None, "multiple", "unavailable", "import"])
def test_probe_child_always_saves_observations(tmp_path, mode, failure):
    diagnostic = load_diagnostic()
    site, isolated = tmp_path / "site-packages", tmp_path / "isolated"
    fake_install(site)
    runner_module = load(RUNNER)
    runner_module.copy_test_tree(ROOT, isolated)
    torch_file = site / "torch/__init__.py"
    if failure == "multiple":
        torch_file.write_text(
            torch_file.read_text().replace(
                "device_count=lambda:1", "device_count=lambda:16"
            )
        )
    elif failure == "unavailable":
        torch_file.write_text(
            torch_file.read_text().replace(
                "is_available=lambda:True", "is_available=lambda:False"
            )
        )
    elif failure == "import":
        torch_file.write_text("raise ImportError('diagnostic import error')\n")
    output = tmp_path / "observed.json"
    env = runner_module.clean_environment(os.environ, ROOT, isolated)
    env.update(PPU_CI_REQUIRES="[]", PPU_PROBE_OUTPUT=str(output))
    command = f"""
import sys, importlib.util
sys.path.insert(0, {str(site)!r})
spec = importlib.util.spec_from_file_location('probe', {diagnostic.__file__!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.observe({mode!r}, dict(__import__('os').environ))
"""
    result = subprocess.run(
        [sys.executable, "-c", command],
        env=env,
        cwd=isolated,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(output.read_bytes())
    assert observed["mode"] == mode and observed["qualification"] is False
    if failure == "import":
        assert "diagnostic import error" in observed["error"]
    else:
        assert observed["devices"]["device_count"]["value"] == (
            16 if failure == "multiple" else 1
        )
        assert observed["devices"]["is_available"]["value"] is (
            failure != "unavailable"
        )
        if mode == "preflight" and failure:
            assert observed["error"] == "ValueError: 需要一张可用 PPU"
    assert not (tmp_path / "nodes").exists()


def test_probe_pod_metadata_excludes_secrets():
    diagnostic = load_diagnostic()
    pod = {
        "metadata": {"name": "pod", "uid": "uid", "annotations": {"secret": "token"}},
        "spec": {
            "nodeName": "node",
            "containers": [
                {
                    "name": "worker",
                    "image": "image",
                    "resources": {"limits": {"alibabacloud.com/ppu": "1"}},
                    "securityContext": {"privileged": True},
                    "env": [
                        {"name": "CUDA_VISIBLE_DEVICES", "value": "5"},
                        {"name": "GITHUB_TOKEN", "value": "secret-token"},
                    ],
                }
            ],
        },
        "status": {
            "phase": "Running",
            "containerStatuses": [
                {
                    "name": "worker",
                    "imageID": "sha256:image",
                    "containerID": "containerd://id",
                }
            ],
        },
    }
    result = diagnostic.pod_snapshot(pod)
    assert result["uid"] == "uid" and result["node"] == "node"
    assert result["worker"]["env"] == {"CUDA_VISIBLE_DEVICES": "5"}
    assert result["worker_status"]["imageID"] == "sha256:image"
    assert "token" not in json.dumps(result)


@pytest.mark.parametrize("child_failure", [None, "missing", "timeout"])
def test_probe_execution_never_runs_test_files(
    valid_inputs, tmp_path, monkeypatch, child_failure
):
    diagnostic = load_diagnostic()
    output = tmp_path / "probe-results"
    output.mkdir()
    env = {**os.environ, **valid_inputs, "CI_RUN_ID": "456", "PROBE_CODE_SHA": "e" * 40}
    identity, _, _ = diagnostic.verify_inputs(env)
    (output / "cpu-identity.json").write_text(json.dumps(identity))
    monkeypatch.setattr(diagnostic, "result_directory", lambda *a, **kw: output)
    monkeypatch.setattr(diagnostic.runner, "require_nas", lambda *a: None)
    commands = []

    def boundary(command, **kwargs):
        commands.append(command)
        if "-c" in command:
            if child_failure == "timeout":
                raise subprocess.TimeoutExpired(command, 180)
            if child_failure == "missing":
                return subprocess.CompletedProcess(command, 0)
            Path(kwargs["env"]["PPU_PROBE_OUTPUT"]).write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "qualification": False,
                        "mode": kwargs["env"]["PPU_PROBE_MODE"],
                        "devices": {
                            "is_available": {"value": True},
                            "device_count": {"value": 16},
                        },
                        "error": "ValueError: 需要一张可用 PPU",
                    }
                )
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", boundary)
    assert diagnostic.execute(env) == (1 if child_failure else 0)
    summary = json.loads((output / "probe-summary.json").read_bytes())
    assert summary["qualification"] is False
    assert summary["status"] == (
        "diagnostic-incomplete" if child_failure else "diagnostic-complete"
    )
    installs = [c for c in commands if c[1:4] == ["-m", "pip", "install"]]
    assert len(installs) == 2
    assert "--no-deps" in installs[0] and "--force-reinstall" in installs[0]
    assert installs[1][-3:] == [
        "https://mirrors.aliyun.com/pypi/simple/",
        "-r",
        str(ROOT / "requirements/dev.txt"),
    ]
    assert len([c for c in commands if "-c" in c]) == 2
    assert not any("pytest" in c or "serve" in c for c in commands)
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("board", ["ppu10", "ppu15"])
@pytest.mark.parametrize("suite", ["visibility", "gdn", "invalid"])
def test_probe_suite_is_explicit_and_binds_test_sources(valid_inputs, board, suite):
    diagnostic = load_diagnostic()
    env = {
        **valid_inputs,
        "PROBE_CODE_SHA": "e" * 40,
        "PROBE_SUITE": suite,
        "DEVICE_BOARD": board,
    }
    if suite == "invalid":
        with pytest.raises(ValueError, match="专项"):
            diagnostic.verify_inputs(env)
        return
    identity, _, _ = diagnostic.verify_inputs(env)
    assert identity["suite"] == suite
    assert identity["qualification"] is False
    expected = (
        {
            f"tests/e2e/fork_port/test_gdn_pla_{kind}.py": diagnostic.select.checksum(
                (ROOT / f"tests/e2e/fork_port/test_gdn_pla_{kind}.py").read_bytes()
            )
            for kind in ("decode", "prefill")
        }
        if suite == "gdn"
        else {}
    )
    assert identity["test_sources"] == expected


def _write_gdn_probe_evidence(command, env, damage):
    """Build only control-plane contract fixtures, not evidence of device numerical execution."""
    plugin = load(PLUGIN)
    test = command[command.index("tests.support.ppu_ci") + 1]
    node = test + "::test_numerical"
    isolated = env["PPU_CI_ISOLATED_ROOT"]
    config = json.loads(
        Path(command[command.index("--ppu-ci-config") + 1]).read_bytes()
    )
    entry = next(t for t in config["test_catalog"] if t["test"] == test)
    report = {
        "schema": 1,
        "board": env["DEVICE_BOARD"],
        "file": test,
        "status": "success",
        "exitstatus": 0,
        "errors": [],
        "collected": [node],
        "applicable": [node],
        "hardware_excluded": [],
        "nodes": {
            node: {
                phase: {"outcome": "passed", "wasxfail": None, "duration_seconds": 0.1}
                for phase in ("setup", "call", "teardown")
            }
        },
        "runtime": {
            "isolated_root": isolated,
            "checkout_root": env["PPU_CI_CHECKOUT"],
            "runtime_env": {"VLLM_SAIL_USE_PLA": True},
            "paths": {
                name: "/opt/site-packages/" + name.replace(".", "/") + ".py"
                for name in set(entry["requires"]) | plugin.RUNTIME_MODULES
            },
        },
    }
    report["runtime"]["paths"].update(
        plugin=isolated + "/tests/support/ppu_ci.py",
        conftest=isolated + "/tests/conftest.py",
    )
    if damage == "node-skip" and entry["id"] == "gdn-decode":
        report["nodes"][node]["call"]["outcome"] = "skipped"
    Path(command[command.index("--ppu-ci-report") + 1]).write_text(json.dumps(report))
    junit = Path(command[command.index("--junitxml") + 1])
    if damage != "missing-junit" or entry["id"] != "gdn-decode":
        junit.write_text(
            f'<testsuites><testsuite><testcase classname="{test[:-3].replace("/", ".")}" name="test_numerical"/></testsuite></testsuites>'
        )


@pytest.mark.parametrize("board", ["ppu10", "ppu15"])
@pytest.mark.parametrize(
    "damage",
    [
        None,
        "preflight",
        "multiple",
        "missing",
        "timeout",
        "pytest",
        "node-skip",
        "missing-junit",
    ],
)
def test_gdn_probe_runs_only_two_files_and_fails_closed(
    valid_inputs, tmp_path, monkeypatch, board, damage
):
    diagnostic = load_diagnostic()
    output = tmp_path / "gdn-probe"
    env = {
        **os.environ,
        **valid_inputs,
        "CI_RUN_ID": "456",
        "PROBE_CODE_SHA": "e" * 40,
        "PROBE_SUITE": "gdn",
        "DEVICE_BOARD": board,
    }
    monkeypatch.setattr(diagnostic, "result_directory", lambda *a, **kw: output)
    monkeypatch.setattr(diagnostic.runner, "require_nas", lambda *a: None)
    diagnostic.prepare(env)
    original = Path(env["SELECTION_FILE"]).read_bytes()
    commands = []

    def boundary(command, **kwargs):
        commands.append(command)
        child = kwargs.get("env", {})
        if "-c" in command:
            if damage == "timeout":
                raise subprocess.TimeoutExpired(command, 180)
            if damage == "missing":
                return subprocess.CompletedProcess(command, 0)
            Path(child["PPU_PROBE_OUTPUT"]).write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "qualification": False,
                        "mode": child["PPU_PROBE_MODE"],
                        "devices": {
                            "is_available": {"value": True},
                            "device_count": {
                                "value": 16 if damage == "multiple" else 1
                            },
                        },
                        "error": "preflight failure" if damage == "preflight" else None,
                        "runtime": {"runtime_env": {"VLLM_SAIL_USE_PLA": True}},
                    }
                )
            )
        if "pytest" in command:
            assert Path(kwargs["cwd"]) != ROOT
            assert "PYTHONPATH" not in child and child["VLLM_SAIL_USE_PLA"] == "1"
            _write_gdn_probe_evidence(command, child, damage)
            kwargs["stdout"].write("模拟 pytest 控制面输出\n")
            if damage == "pytest" and any(
                "test_gdn_pla_decode.py" == Path(c).name for c in command
            ):
                return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", boundary)
    assert diagnostic.execute(env) == (1 if damage else 0)
    tests = [c[c.index("tests.support.ppu_ci") + 1] for c in commands if "pytest" in c]
    blocked = damage in {"preflight", "multiple", "missing", "timeout"}
    assert tests == (
        []
        if blocked
        else [
            "tests/e2e/fork_port/test_gdn_pla_decode.py",
            "tests/e2e/fork_port/test_gdn_pla_prefill.py",
        ]
    )
    summary = json.loads((output / "probe-summary.json").read_bytes())
    assert summary["suite"] == "gdn" and summary["qualification"] is False
    assert summary["status"] == ("gdn-failed" if damage else "gdn-passed")
    assert not (output / "summary.json").exists()
    assert Path(env["SELECTION_FILE"]).read_bytes() == original
    if damage:
        with pytest.raises((ValueError, OSError)):
            diagnostic.finish(env)
    else:
        diagnostic.finish(env)
        # CPU finalization must recheck identity and evidence, not just trust a success status.
        summary["probe_code_sha"] = "f" * 40
        (output / "probe-summary.json").write_text(json.dumps(summary))
        with pytest.raises(ValueError, match="身份"):
            diagnostic.finish(env)
        summary["probe_code_sha"] = env["PROBE_CODE_SHA"]
        (output / "probe-summary.json").write_text(json.dumps(summary))
        (output / "junit/gdn-decode.xml").write_text("<broken>")
        with pytest.raises(ValueError, match="JUnit"):
            diagnostic.finish(env)


def test_probe_prepare_creates_distinct_cpu_identity(
    valid_inputs, tmp_path, monkeypatch
):
    diagnostic = load_diagnostic()
    assert hasattr(diagnostic, "prepare"), "缺少诊断准备"
    output = tmp_path / "probe"
    env = {**valid_inputs, "CI_RUN_ID": "456", "PROBE_CODE_SHA": "e" * 40}
    monkeypatch.setattr(diagnostic, "result_directory", lambda *a, **kw: output)
    monkeypatch.setattr(diagnostic.runner, "require_nas", lambda *a: None)
    diagnostic.prepare(env)
    assert json.loads((output / "cpu-identity.json").read_bytes())["run_id"] == "456"
    assert (output / "wheel-manifest.json").is_file()
    with pytest.raises(FileExistsError):
        diagnostic.prepare(env)


@pytest.mark.parametrize("found", [True, False])
def test_probe_watcher_only_reads_current_pod(
    valid_inputs, tmp_path, monkeypatch, found
):
    diagnostic = load_diagnostic()
    assert hasattr(diagnostic, "watch"), "缺少清理前 Pod 采集"
    monkeypatch.setattr(diagnostic, "result_directory", lambda *a, **kw: tmp_path)
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        pod = {
            "metadata": {"name": args[5], "uid": "uid"},
            "status": {"containerStatuses": [{"name": "worker", "imageID": "digest"}]},
        }
        return subprocess.CompletedProcess(
            args,
            0 if found else 1,
            stdout=json.dumps(pod) if found else "",
            stderr="" if found else "Forbidden",
        )

    monkeypatch.setattr(subprocess, "run", command)
    env = {**valid_inputs, "GITHUB_REPOSITORY_OWNER": "t-head"}
    diagnostic.watch(env, timeout=0.01, interval=0)
    assert calls and calls[0][1:] == [
        "get",
        "pod",
        "-n",
        "ppu-sched",
        "ppu-t-head-123-1-probe-ppu10-worker-0",
        "-o",
        "json",
    ]
    report = json.loads((tmp_path / "pod-observation.json").read_bytes())
    if found:
        assert report["pod"]["uid"] == "uid"
    else:
        assert report["error"] == "Forbidden"
        assert report["status"] == "unavailable"


def test_probe_paths_cannot_overlap_qualification(valid_inputs):
    diagnostic = load_diagnostic()
    assert (
        str(diagnostic.result_directory(valid_inputs, pod=True))
        == "/mnt/wl_nas/devops/123-1/device-probe/ppu10"
    )
    assert (
        str(diagnostic.result_directory(valid_inputs, pod=False))
        == "/wl_nas/devops/123-1/device-probe/ppu10"
    )
    with pytest.raises(ValueError):
        diagnostic.result_directory(
            {**valid_inputs, "DEVICE_BOARD": "../bad"}, pod=True
        )


def test_test_local_pla_monkeypatch_does_not_invalidate_startup(tmp_path):
    result, report = run_case(
        tmp_path,
        "def test_toggle(monkeypatch):\n import vllm_sail.envs as envs\n monkeypatch.setattr(envs, 'VLLM_SAIL_USE_PLA', False)\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert report["status"] == "success"
