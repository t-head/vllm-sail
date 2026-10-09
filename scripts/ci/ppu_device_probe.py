# SPDX-License-Identifier: Apache-2.0
"""独立的手动设备诊断；复用历史 wheel，但不生成资格通过证据。"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ppu_device_runner", Path(__file__).with_name("ppu_device_runner.py")
)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)
select = runner.select
SAFE_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "PPU_VISIBLE_DEVICES",
    "TPU_VISIBLE_DEVICES",
    "HGGC_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "PPU_SDK",
    "UMD_PLATFORM_TYPE",
    "HGGC_DRIVER_CANDIDATE",
    "PATH",
    "LD_LIBRARY_PATH",
    "BASH_ENV",
    "NODE_NAME",
    "NPROC_PER_NODE",
    "WORLD_SIZE",
    "VLLM_SAIL_USE_PLA",
    "VLLM_PLUGINS",
)


def verify_inputs(env):
    # 不调用或放宽资格执行器的同 run 限制；这里明确校验历史产物身份。
    path = Path(env["SELECTION_FILE"])
    raw = path.read_bytes()
    original = json.loads(raw)
    config_raw = path.with_name("ppu_device_tests.json").read_bytes()
    environment_raw = path.with_name("ppu_device_environment.json").read_bytes()
    for name, content in (
        ("ppu_device_tests.json", config_raw),
        ("ppu_device_environment.json", environment_raw),
    ):
        runner.require(
            content == (ROOT / "scripts/ci" / name).read_bytes(),
            "诊断配置与原批产物不同",
        )
    config = json.loads(config_raw)
    select.validate_config(config, ROOT)
    select.validate_selection(
        original,
        config_raw,
        environment_raw,
        tested_sha=env["EXPECTED_SAIL_COMMIT"],
        build_run_id=env["EXPECTED_BUILD_RUN_ID"],
        build_image=env["EXPECTED_BUILD_IMAGE"],
        runtime_image=env["RUNTIME_IMAGE"],
        expected_vllm_commit=env["EXPECTED_VLLM_COMMIT"],
    )
    manifest = select.load_sibling("ppu_wheel_manifest").verify(
        Path(env["WHEELS_DIR"]),
        vllm_commit=env["EXPECTED_VLLM_COMMIT"],
        sail_commit=env["EXPECTED_SAIL_COMMIT"],
    )
    fingerprint = select.build_fingerprint(
        select.read_diagnostics(env["BUILD_MANIFEST_FILE"]),
        manifest,
        tested_sha=env["EXPECTED_SAIL_COMMIT"],
        expected_vllm_commit=env["EXPECTED_VLLM_COMMIT"],
        build_image=env["EXPECTED_BUILD_IMAGE"],
    )
    runner.require(fingerprint == original["build_fingerprint"], "诊断构建指纹不符")
    runner.require(env["DEVICE_BOARD"] in select.BOARDS, "非法诊断板型")
    return (
        {
            "schema": 1,
            "qualification": False,
            "run_id": select.decimal(env["CI_RUN_ID"]),
            "run_attempt": select.decimal(env["CI_RUN_ATTEMPT"]),
            "probe_code_sha": select.sha(env["PROBE_CODE_SHA"]),
            "build_run_id": env["EXPECTED_BUILD_RUN_ID"],
            "wheel_sail_commit": env["EXPECTED_SAIL_COMMIT"],
            "wheel_vllm_commit": env["EXPECTED_VLLM_COMMIT"],
            "board": env["DEVICE_BOARD"],
            "runtime_image": env["RUNTIME_IMAGE"],
            "selection_sha256": select.checksum(raw),
            "build_fingerprint": fingerprint,
        },
        config,
        manifest,
    )


def result_directory(env, *, pod):
    run = select.decimal(env["CI_RUN_ID"])
    attempt = select.decimal(env["CI_RUN_ATTEMPT"])
    board = env["DEVICE_BOARD"]
    runner.require(board in select.BOARDS, "非法诊断板型")
    return (
        Path("/mnt/wl_nas" if pod else "/wl_nas")
        / "devops"
        / f"{run}-{attempt}"
        / "device-probe"
        / board
    )


def observe(mode, env):
    """每种导入路径使用独立进程；严格检查失败也先落盘。"""
    report = {
        "schema": 1,
        "qualification": False,
        "mode": mode,
        "devices": {},
        "error": None,
    }
    destination = Path(env["PPU_PROBE_OUTPUT"])
    select.write_json(destination, report)
    try:
        plugin = importlib.import_module("tests.support.ppu_ci")
        if mode == "preflight":
            report["runtime"] = plugin.inspect_runtime(
                env["PPU_CI_ISOLATED_ROOT"],
                env["PPU_CI_CHECKOUT"],
                json.loads(env["PPU_CI_REQUIRES"]),
                device_observation=report["devices"],
            )
        else:
            runner.require(mode == "torch-only", "未知诊断模式")
            report["devices"] = plugin.observe_devices(importlib.import_module("torch"))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        select.write_json(destination, report)
    # 这些辅助观测在原始设备查询之后进行，不改变首次导入/初始化条件。
    report["modules"] = {}
    for name in ("torch", "torch._C", "triton", "pla", "vllm", "vllm_sail"):
        module = sys.modules.get(name)
        if module is not None:
            report["modules"][name] = {
                "path": str(getattr(module, "__file__", None)),
                "version": str(getattr(module, "__version__", None)),
            }
    report["packages"] = {}
    for name in ("torch", "triton", "pla", "packaging", "nvidia-dali-cuda130"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    select.write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[:65536],
            "stderr": result.stderr[:65536],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def execute(env):
    output = result_directory(env, pod=True)
    runner.require_nas(Path("/mnt/wl_nas"))
    identity, config, manifest = verify_inputs(env)
    runner.require(
        json.loads((output / "cpu-identity.json").read_bytes()) == identity,
        "CPU/Pod 诊断身份不同",
    )
    report = {**identity, "status": "diagnostic-incomplete", "error": None}
    select.write_json(output / "probe-summary.json", report)
    select.write_json(
        output / "worker-environment.json",
        {
            "hostname": socket.gethostname(),
            "env": {key: env.get(key) for key in SAFE_ENV},
            "device_paths": sorted(
                str(p)
                for p in Path("/dev").iterdir()
                if p.name.startswith(("ppu", "tpu", "nvidia", "hggc"))
            ),
        },
    )
    try:
        with tempfile.TemporaryDirectory(prefix="ppu-probe-", dir="/tmp") as temporary:
            isolated = Path(temporary).resolve() / "isolated"
            runner.copy_test_tree(ROOT, isolated)
            child_env = runner.clean_environment(env, ROOT, isolated)
            wheels = [
                str(Path(env["WHEELS_DIR"]) / item["file"])
                for item in manifest["wheels"].values()
            ]
            with (output / "install.log").open("w") as log:
                for command in (
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-deps",
                        "--force-reinstall",
                        *wheels,
                    ],
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--index-url",
                        env.get("SAIL_PIP_INDEX_URL")
                        or "https://mirrors.aliyun.com/pypi/simple/",
                        "-r",
                        str(ROOT / "requirements/dev.txt"),
                    ],
                ):
                    subprocess.run(
                        command,
                        check=True,
                        cwd=isolated,
                        env=child_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                    )
            child_env["PPU_CI_REQUIRES"] = json.dumps(
                sorted(
                    {
                        r
                        for t in config["test_catalog"]
                        if identity["board"] in t["boards"]
                        for r in t["requires"]
                    }
                )
            )
            child_env["PPU_PROBE_SCRIPT"] = str(Path(__file__).resolve())
            report["children"] = {}
            for mode in ("preflight", "torch-only"):
                child_env.update(
                    PPU_PROBE_MODE=mode,
                    PPU_PROBE_OUTPUT=str(output / f"{mode}-observation.json"),
                )
                with (output / f"{mode}.log").open("w") as log:
                    try:
                        result = subprocess.run(
                            [
                                sys.executable,
                                "-c",
                                "import os, runpy; runpy.run_path(os.environ['PPU_PROBE_SCRIPT'], run_name='__main__')",
                                "child",
                            ],
                            cwd=isolated,
                            env=child_env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=180,
                        )
                        observed = json.loads(
                            Path(child_env["PPU_PROBE_OUTPUT"]).read_bytes()
                        )
                        runner.require(
                            observed.get("schema") == 1
                            and observed.get("qualification") is False
                            and observed.get("mode") == mode,
                            "诊断子进程报告身份不符",
                        )
                        report["children"][mode] = {"returncode": result.returncode}
                    except subprocess.TimeoutExpired:
                        report["children"][mode] = {"error": "诊断子进程超时"}
                    except (OSError, ValueError) as error:
                        report["children"][mode] = {
                            "error": f"{type(error).__name__}: {error}"
                        }
                select.write_json(output / "probe-summary.json", report)
            report["status"] = (
                "diagnostic-complete"
                if all(
                    item.get("returncode") == 0 for item in report["children"].values()
                )
                else "diagnostic-incomplete"
            )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        select.write_json(output / "probe-summary.json", report)
        select.write_json(
            output / "system-observation.json",
            {
                "sdk": capture(["hgcc", "--version"]),
                "smi": capture([shutil.which("ppu-smi") or "nvidia-smi"]),
            },
        )
    return 0 if report["status"] == "diagnostic-complete" else 1


def pod_snapshot(pod):
    metadata, spec, status = (
        pod.get(key, {}) for key in ("metadata", "spec", "status")
    )
    worker = next((c for c in spec.get("containers", []) if c["name"] == "worker"), {})
    state = next(
        (c for c in status.get("containerStatuses", []) if c["name"] == "worker"), {}
    )
    return {
        "name": metadata.get("name"),
        "uid": metadata.get("uid"),
        "node": spec.get("nodeName"),
        "phase": status.get("phase"),
        "node_selector": spec.get("nodeSelector"),
        "worker": {
            "image": worker.get("image"),
            "resources": worker.get("resources"),
            "security_context": worker.get("securityContext"),
            "env": {
                item["name"]: item.get("value")
                for item in worker.get("env", [])
                if item["name"] in SAFE_ENV
            },
        },
        "worker_status": {
            key: state.get(key) for key in ("imageID", "containerID", "restartCount")
        },
    }


def prepare(env):
    identity, _, manifest = verify_inputs(env)
    runner.require_nas(Path("/wl_nas"))
    output = result_directory(env, pod=False)
    output.mkdir(parents=True, exist_ok=False)
    select.write_json(output / "cpu-identity.json", identity)
    select.write_json(output / "wheel-manifest.json", manifest)
    shutil.copyfile(env["BUILD_MANIFEST_FILE"], output / "build-manifest.txt")
    shutil.copyfile(env["SELECTION_FILE"], output / "selection.json")


def watch(env, *, timeout=1500, interval=2):
    # 独立 CPU 进程在 action 无条件清理前保存快照；不保留或修改 Pod。
    output = result_directory(env, pod=False)
    name = f"ppu-{env['GITHUB_REPOSITORY_OWNER'].lower()}-{env['CI_RUN_ID']}-{env['CI_RUN_ATTEMPT']}-probe-{env['DEVICE_BOARD']}-worker-0"
    report = {"status": "unavailable", "pod_name": name, "error": "尚未取得 Pod"}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not (output / "watch.stop").exists():
        binary = (
            shutil.which(
                "kubectl",
                path="/opt/bin:"
                + env.get("PATH", "")
                + ":"
                + str(Path.home() / ".local/bin"),
            )
            or "kubectl"
        )
        try:
            result = subprocess.run(
                [binary, "get", "pod", "-n", "ppu-sched", name, "-o", "json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                snapshot = pod_snapshot(json.loads(result.stdout))
                runner.require(snapshot["name"] == name, "Pod 名称不符")
                report.update(pod=snapshot, status="observed", error=None)
            else:
                report["error"] = result.stderr[:4096]
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            report["error"] = f"{type(error).__name__}: {error}"
        select.write_json(output / "pod-observation.json", report)
        if report.get("pod", {}).get("worker_status", {}).get("imageID"):
            break
        time.sleep(interval)
    select.write_json(output / "pod-observation.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "child", "prepare", "watch"))
    args = parser.parse_args()
    env = dict(os.environ)
    if args.command == "child":
        observe(env["PPU_PROBE_MODE"], env)
    elif args.command == "prepare":
        prepare(env)
    elif args.command == "watch":
        watch(env)
    else:
        sys.exit(execute(env))
