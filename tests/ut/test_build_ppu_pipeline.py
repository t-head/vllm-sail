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
        "ppu_image_stack.py",
        "status=running",
        'export PIP_CONSTRAINT="${WORK_DIR}/image-constraints.txt"',
        ".github/vllm-main-verified.commit",
        "unset TORCH_CUDA_ARCH_LIST",
        "command -v hgcc",
        "compatible_wrapper.h",
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
    for job in ("build", "ops-build"):
        uploads = {
            step["with"]["name"]: step["with"]
            for step in jobs[job]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        }
        assert set(uploads) == {
            "ppu-wheels-${{ github.run_id }}",
            "ppu-build-diagnostics-${{ github.run_id }}-${{ github.run_attempt }}"
            if job == "build"
            else "ppu-build-diagnostics-${{ github.run_id }}",
        }
        assert all(config["retention-days"] == 14 for config in uploads.values())
        assert (
            uploads["ppu-wheels-${{ github.run_id }}"]["if-no-files-found"] == "error"
        )
    for job, name in (
        ("detect-ops-changes", "changes"),
        ("select-ops-tests", "selection"),
    ):
        assert any(
            step.get("with", {}).get("name")
            == f"ppu-ops-{name}-${{{{ github.run_id }}}}-${{{{ github.run_attempt }}}}"
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


def test_image_build_only_replaces_torch_before_capturing_the_stack():
    text = SCRIPT.read_text()
    for forbidden in (
        "SDK_URL",
        "TORCH_URL",
        "SDK_INSTALLER",
        "TORCH_ARCHIVE",
        'mv "${path}"',
        "torch_wheels",
        "download()",
    ):
        assert forbidden not in text
    prepare = text.index('ppu_image_stack.py" prepare')
    verify = text.index('ppu_image_stack.py" verify')
    assert prepare < text.index("compatible_wrapper.h")
    assert prepare < text.index("python -m pip install") < verify
    assert verify < text.index("status=succeeded")


def test_all_ci_paths_use_the_same_pinned_image_stack():
    import yaml

    image = "reg.docker.alibaba-inc.com/ai-tmp/llm@sha256:24872709b1a84ec0e91260eeeccf6ae7ba52a6c70fb90f55a7aae1c1df503bcb"
    workflow = yaml.safe_load(WORKFLOW.read_text())
    inputs = workflow[True]["workflow_dispatch"]["inputs"]
    assert not {"sdk_url", "torch_url"} & inputs.keys()
    assert inputs["build_image"]["default"] == image
    assert inputs["ops_image"]["default"] == image
    build = workflow["jobs"]["build"]
    assert image in build["container"]["image"]
    assert build["env"]["VLLM_SAIL_BUILD_IMAGE"] == build["container"]["image"]
    for job in ("pin-unit-tests", "smoke"):
        assert "needs.build.outputs.image" in workflow["jobs"][job]["with"]["image"]
    for name in ("ppu-pin-ut.yaml", "e2e-ppu.yaml"):
        triggers = yaml.safe_load((WORKFLOW.parent / name).read_text())[True]
        for trigger in triggers.values():
            assert trigger["inputs"]["image"]["default"] == image


@pytest.fixture
def image_stack():
    import importlib.util

    path = ROOT / "scripts/ci/ppu_image_stack.py"
    assert path.is_file(), "image stack fingerprint helper is missing"
    spec = importlib.util.spec_from_file_location("image_stack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def stack_trees(tmp_path):
    sdk = tmp_path / "sdk"
    torch = tmp_path / "torch"
    sdk.mkdir()
    torch.mkdir()
    (sdk / "hgcc").write_bytes(b"compiler")
    (torch / "_C.so").write_bytes(b"torch")
    return sdk, torch


def test_image_stack_hashes_installed_contents_and_ignores_bytecode(
    image_stack, stack_trees
):
    sdk, torch = stack_trees
    before = image_stack.snapshot(sdk, torch, "2.13.0", {"torch": "2.13.0"})
    (torch / "__pycache__").mkdir()
    (torch / "__pycache__/generated.pyc").write_bytes(b"cache")
    after = image_stack.snapshot(sdk, torch, "2.13.0", {"torch": "2.13.0"})
    image_stack.verify_unchanged(before, after)
    assert before["stack_source"] == "image"
    assert len(before["sdk_sha256"]) == len(before["torch_sha256"]) == 64
    assert before["sdk_sha256"] != before["torch_sha256"]


@pytest.mark.parametrize("component", ["sdk", "torch", "packages", "version"])
def test_image_stack_rejects_changes(image_stack, stack_trees, component):
    sdk, torch = stack_trees
    versions = {"torch": "2.13.0", "triton": "3.5.0"}
    before = image_stack.snapshot(sdk, torch, "2.13.0", versions)
    version = "2.13.0"
    if component == "sdk":
        (sdk / "hgcc").write_bytes(b"replaced")
    elif component == "torch":
        (torch / "_C.so").write_bytes(b"replaced")
    elif component == "packages":
        versions = {**versions, "triton": "3.6.0"}
    else:
        version = "2.14.0"
    after = image_stack.snapshot(sdk, torch, version, versions)
    with pytest.raises(ValueError, match="image stack changed"):
        image_stack.verify_unchanged(before, after)


def test_image_stack_constraints_protect_installed_and_absent_device_packages(
    image_stack,
):
    from packaging.requirements import Requirement

    constraints = image_stack.constraints_text(
        {"torch": "2.13.0+ppu", "triton": "3.5.0"}
    )
    requirements = {r.name: r for r in map(Requirement, constraints.splitlines())}
    assert requirements["torch"].specifier.contains("2.13.0+ppu")
    assert not requirements["torch"].specifier.contains("2.13.0")
    assert not requirements["triton"].specifier.contains("3.6.0")
    assert not requirements["flash-attn"].specifier.contains("2.8.2")
    assert "vllm" not in requirements


def test_image_stack_rejects_missing_sdk(image_stack, tmp_path):
    with pytest.raises(ValueError, match="missing or empty"):
        image_stack.tree_digest(tmp_path / "missing")


def test_image_stack_hashes_symlinked_library_directories(
    image_stack, stack_trees, tmp_path
):
    sdk, _ = stack_trees
    libraries = tmp_path / "libraries"
    libraries.mkdir()
    (libraries / "runtime.so").write_bytes(b"runtime")
    (sdk / "lib").symlink_to(libraries, target_is_directory=True)
    before = image_stack.tree_digest(sdk)
    (libraries / "runtime.so").write_bytes(b"replacement")
    assert image_stack.tree_digest(sdk) != before


def test_image_stack_handles_directory_symlink_cycles(image_stack, stack_trees):
    sdk, _ = stack_trees
    (sdk / "back").symlink_to(sdk, target_is_directory=True)
    before = image_stack.tree_digest(sdk)
    assert image_stack.tree_digest(sdk) == before
    (sdk / "hgcc").write_bytes(b"changed")
    assert image_stack.tree_digest(sdk) != before


def test_image_stack_hashes_symlinked_files(image_stack, stack_trees, tmp_path):
    sdk, _ = stack_trees
    library = tmp_path / "runtime.so"
    library.write_bytes(b"runtime")
    (sdk / "runtime.so").symlink_to(library)
    before = image_stack.tree_digest(sdk)
    library.write_bytes(b"replacement")
    assert image_stack.tree_digest(sdk) != before


def test_image_stack_rejects_empty_sdk(image_stack, tmp_path):
    with pytest.raises(ValueError, match="missing or empty"):
        image_stack.tree_digest(tmp_path)


def test_image_stack_cli_capture_verify_and_refuse_mutation(tmp_path):
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "hgcc").write_bytes(b"compiler")
    site = tmp_path / "site"
    torch = site / "torch"
    torch.mkdir(parents=True)
    (torch / "__init__.py").write_text('__version__ = "2.13.0"\n')
    dist = site / "torch-2.13.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: torch\nVersion: 2.13.0\n")
    output = tmp_path / "stack.json"
    constraints = tmp_path / "constraints.txt"
    script = ROOT / "scripts/ci/ppu_image_stack.py"

    def run(command, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(script),
                command,
                "--sdk",
                str(sdk),
                "--output",
                str(output),
                *extra,
            ],
            env={**os.environ, "PYTHONPATH": str(site)},
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )

    captured = run("capture", "--constraints", str(constraints))
    assert captured.returncode == 0, captured.stderr
    assert "stack_source=image" in captured.stdout
    assert "torch===2.13.0" in constraints.read_text()
    verified = run("verify")
    assert verified.returncode == 0, verified.stderr
    assert "stack_unchanged=true" in verified.stdout
    (sdk / "hgcc").write_bytes(b"changed")
    refused = run("verify")
    assert refused.returncode != 0
    assert "image stack changed: sdk_sha256" in refused.stderr


def test_torch_override_is_content_pinned(image_stack):
    assert image_stack.TORCH_OVERRIDE == {
        "torch_archive_url": "https://art-pub.eng.t-head.cn/artifactory/apackage/daily/ppu.2v2_release/latest/PPU_SDK_PYPI/pytorch/ppu_sdk_hggcrt3-pytorch2.13.0-ubuntu2404-py312.tar.gz",
        "torch_archive_sha256": "63e23fedca52f1039b2deac29aa37d7b88d8ba7413f9b5600d223ffb095ec1f7",
        "torch_wheel_sha256": "24a95ab1e6bbc9d8dd61a4347020dd1975e7bdfb06e2110bbc618874a8675760",
    }


@pytest.fixture
def torch_archive(image_stack, tmp_path, monkeypatch):
    import hashlib
    import io
    import tarfile

    archive = tmp_path / "torch.tar.gz"
    wheel_data = b"verified torch wheel"

    def create(names):
        with tarfile.open(archive, "w:gz") as target:
            for name in names:
                info = tarfile.TarInfo(name)
                info.size = len(wheel_data)
                target.addfile(info, io.BytesIO(wheel_data))
        monkeypatch.setattr(
            image_stack,
            "TORCH_OVERRIDE",
            {
                "torch_archive_url": archive.as_uri(),
                "torch_archive_sha256": hashlib.sha256(
                    archive.read_bytes()
                ).hexdigest(),
                "torch_wheel_sha256": hashlib.sha256(wheel_data).hexdigest(),
            },
            raising=False,
        )
        return archive

    create(
        [
            "package/torch-2.13.0-cp312-cp312-linux_x86_64.whl",
            "package/torchvision-0.28.0-cp312-cp312-linux_x86_64.whl",
        ]
    )
    return create


def test_prepare_installs_only_verified_torch_and_keeps_other_packages(
    image_stack, stack_trees, torch_archive, monkeypatch
):
    sdk, _ = stack_trees
    versions = {"torch": "2.13.0", "triton": "3.7.0", "torchvision": "0.28.0"}
    monkeypatch.setattr(image_stack, "package_versions", lambda: versions)
    monkeypatch.setenv("PIP_CONSTRAINT", "/not-yet-created/constraints.txt")
    commands = []

    def install(command, **kwargs):
        commands.append(command)
        assert command[:-1] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--force-reinstall",
        ]
        assert Path(command[-1]).name == "torch-2.13.0-cp312-cp312-linux_x86_64.whl"
        assert Path(command[-1]).read_bytes() == b"verified torch wheel"
        assert "PIP_CONSTRAINT" not in kwargs["env"]
        assert kwargs["check"] is True

    monkeypatch.setattr(subprocess, "run", install)
    before = image_stack.tree_digest(sdk)
    image_stack.install_torch(sdk)
    assert len(commands) == 1
    assert image_stack.tree_digest(sdk) == before


@pytest.mark.parametrize("failure", ["timeout", "connection", "read", "http503"])
def test_prepare_retries_transient_download_before_installing(
    image_stack, stack_trees, torch_archive, monkeypatch, failure
):
    import io
    import time
    import urllib.error

    original_open = image_stack.urllib.request.urlopen
    attempts, delays, installs, responses = [], [], [], []
    errors = {
        "timeout": TimeoutError("download timed out"),
        "connection": urllib.error.URLError(ConnectionResetError("connection reset")),
        "http503": urllib.error.HTTPError(
            "https://example.invalid", 503, "busy", {}, None
        ),
    }

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionResetError("download interrupted")
            return super().read(5)

    def open_archive(url, *, timeout):
        attempts.append((url, timeout))
        if len(attempts) <= 2:
            if failure != "read":
                raise errors[failure]
            response = InterruptedResponse(b"partial archive")
            responses.append(response)
            return response
        return original_open(url, timeout=timeout)

    def install(command, **kwargs):
        assert Path(command[-1]).read_bytes() == b"verified torch wheel"
        installs.append(command)

    monkeypatch.setattr(image_stack, "package_versions", lambda: {})
    monkeypatch.setattr(image_stack.urllib.request, "urlopen", open_archive)
    monkeypatch.setattr(time, "sleep", delays.append)
    monkeypatch.setattr(subprocess, "run", install)
    image_stack.install_torch(stack_trees[0])
    assert attempts == [(image_stack.TORCH_OVERRIDE["torch_archive_url"], 90)] * 3
    assert delays == [1, 2]
    assert len(installs) == 1
    assert all(response.closed for response in responses)


def test_prepare_download_stops_after_three_attempts(
    image_stack, stack_trees, monkeypatch
):
    import time

    failure = TimeoutError("download timed out")
    attempts, delays = [], []

    def open_archive(*args, **kwargs):
        attempts.append(args)
        raise failure

    monkeypatch.setattr(image_stack, "package_versions", lambda: {})
    monkeypatch.setattr(image_stack.urllib.request, "urlopen", open_archive)
    monkeypatch.setattr(time, "sleep", delays.append)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: pytest.fail("failed downloads must not reach installation"),
    )
    with pytest.raises(TimeoutError) as caught:
        image_stack.install_torch(stack_trees[0])
    assert caught.value is failure
    assert len(attempts) == 3
    assert delays == [1, 2]


def test_prepare_does_not_retry_http_not_found(image_stack, stack_trees, monkeypatch):
    import time
    import urllib.error

    failure = urllib.error.HTTPError(
        "https://example.invalid", 404, "missing", {}, None
    )
    attempts = []

    def open_archive(*args, **kwargs):
        attempts.append(args)
        raise failure

    monkeypatch.setattr(image_stack, "package_versions", lambda: {})
    monkeypatch.setattr(image_stack.urllib.request, "urlopen", open_archive)
    monkeypatch.setattr(
        time, "sleep", lambda _: pytest.fail("permanent errors must not be retried")
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: pytest.fail("failed downloads must not reach installation"),
    )
    with pytest.raises(urllib.error.HTTPError) as caught:
        image_stack.install_torch(stack_trees[0])
    assert caught.value is failure
    assert len(attempts) == 1


@pytest.mark.parametrize("damage", ["archive", "wheel", "missing", "duplicate"])
def test_prepare_rejects_invalid_torch_before_pip(
    image_stack, stack_trees, torch_archive, monkeypatch, damage
):
    import time

    monkeypatch.setattr(
        time, "sleep", lambda _: pytest.fail("validation errors must not be retried")
    )
    if damage in ("archive", "wheel"):
        image_stack.TORCH_OVERRIDE[f"torch_{damage}_sha256"] = "0" * 64
    elif damage == "missing":
        torch_archive(["package/torchvision-0.28.0-cp312-cp312-linux_x86_64.whl"])
    else:
        torch_archive(
            [
                "first/torch-2.13.0-cp312-cp312-linux_x86_64.whl",
                "second/torch-2.13.0-cp312-cp312-linux_x86_64.whl",
            ]
        )
    monkeypatch.setattr(image_stack, "package_versions", lambda: {})
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: pytest.fail("unverified packages must not be installed"),
    )
    with pytest.raises(ValueError, match="SHA256|exactly one"):
        image_stack.install_torch(stack_trees[0])


@pytest.mark.parametrize("damage", ["sdk", "packages"])
def test_prepare_rejects_changes_outside_torch(
    image_stack, stack_trees, torch_archive, monkeypatch, damage
):
    sdk, _ = stack_trees
    versions = {"torch": "2.13.0", "triton": "3.7.0"}
    monkeypatch.setattr(image_stack, "package_versions", lambda: dict(versions))

    def install(*args, **kwargs):
        if damage == "sdk":
            (sdk / "hgcc").write_bytes(b"changed")
        else:
            versions["triton"] = "different"

    monkeypatch.setattr(subprocess, "run", install)
    with pytest.raises(ValueError, match="image stack changed"):
        image_stack.install_torch(sdk)


@pytest.mark.parametrize("header", [True, False])
def test_prepare_records_replacement_provenance_and_checks_header(
    image_stack, stack_trees, torch_archive, tmp_path, monkeypatch, header, capsys
):
    from types import SimpleNamespace

    original_run = subprocess.run
    sdk, torch_root = stack_trees
    if header:
        (torch_root / ".ppu_compat").mkdir()
        (torch_root / ".ppu_compat/compatible_wrapper.h").write_text("header")
    output = tmp_path / "stack.json"
    constraints = tmp_path / "constraints.txt"
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(image_stack, "package_versions", lambda: {"torch": "2.13.0"})

    def install(*args, **kwargs):
        assert "torch" not in sys.modules
        print("pip installation progress", file=kwargs.get("stdout"))
        monkeypatch.setitem(
            sys.modules,
            "torch",
            SimpleNamespace(
                __file__=str(torch_root / "__init__.py"), __version__="2.13.0"
            ),
        )

    monkeypatch.setattr(subprocess, "run", install)
    argv = ["stack", "prepare", "--sdk", str(sdk), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", [*argv, "--constraints", str(constraints)])
    if not header:
        with pytest.raises(ValueError, match="compatible_wrapper.h"):
            image_stack.main()
        assert not output.exists()
        return
    image_stack.main()
    captured = capsys.readouterr()
    from scripts.ci.ppu_ops_select import read_diagnostics

    manifest = tmp_path / "build-manifest.txt"
    text = SCRIPT.read_text()
    start = text.index("{\n    printf 'status=running")
    end = text.index('} >>"${MANIFEST}"', start) + len('} >>"${MANIFEST}"')
    original_run(
        ["bash", "-c", text[start:end]],
        env={
            **os.environ,
            "MANIFEST": str(manifest),
            "VLLM_REF": "a" * 40,
            "PYTORCH_SAIL_ARCH": "ppu_15;ppu_10",
        },
        check=True,
    )
    manifest.write_text(manifest.read_text() + captured.out)
    assert read_diagnostics(manifest)["stack_source"] == "image+torch"
    observed = json.loads(output.read_text())
    assert observed["stack_source"] == "image+torch"
    assert all(observed[k] == v for k, v in image_stack.TORCH_OVERRIDE.items())
    assert "torch===2.13.0" in constraints.read_text()
    monkeypatch.setattr(sys, "argv", ["stack", "verify", *argv[2:]])
    image_stack.main()
    (torch_root / "_C.so").write_bytes(b"modified after preparation")
    with pytest.raises(ValueError, match="torch_sha256"):
        image_stack.main()


@pytest.mark.parametrize("script", ["ppu_pin_ut.sh", "ppu_e2e_smoke.sh"])
def test_runtime_scripts_prepare_same_torch_before_installing_wheels(script):
    text = (ROOT / "scripts/ci" / script).read_text()
    prepare = text.index('ppu_image_stack.py" prepare')
    install = text.index("python -m pip install")
    verify = text.index('ppu_image_stack.py" verify')
    assert prepare < install < verify
    assert "--constraints" in text[prepare:install]


def load_workflow(name):
    import yaml

    path = WORKFLOW.parent / name
    assert path.exists(), f"Missing workflow: {name}"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize(
    "name", ["run-ci-command.yml", "notify-ci-authorized.yml", "ppu-ci-report.yml"]
)
def test_control_workflows_only_execute_immutable_trusted_code(name):
    workflow = load_workflow(name)
    assert workflow["permissions"] == {}
    assert workflow["cache-mode"] == "none"
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest"
        for step in job["steps"]:
            if step.get("uses", "").startswith("actions/checkout@"):
                expected_ref = (
                    "${{ steps.default-revision.outputs.result }}"
                    if name == "notify-ci-authorized.yml"
                    else "${{ github.workflow_sha }}"
                )
                assert step["with"]["ref"] == expected_ref
                assert step["with"]["persist-credentials"] is False
            assert "download-artifact" not in step.get("uses", "")
            assert "${{ github.event" not in step.get("run", "")
    if name == "run-ci-command.yml":
        assert workflow[True] == {"issue_comment": {"types": ["created"]}}
    elif name == "notify-ci-authorized.yml":
        assert workflow[True] == {"pull_request_target": {"types": ["labeled"]}}
        assert "actions" not in workflow["jobs"]["notify"]["permissions"]
    else:
        assert workflow[True]["workflow_run"]["workflows"] == ["PPU PR CI"]
        assert set(workflow[True]["workflow_run"]["types"]) == {
            "in_progress",
            "completed",
            "requested",
        }


@pytest.fixture
def without_system_node(monkeypatch):
    # Exercise the same missing-PATH condition as a fresh PPU worker.
    monkeypatch.setenv("PATH", "")


def test_node_runtime_is_a_pinned_dev_dependency():
    from packaging.requirements import Requirement

    requirements = (ROOT / "requirements/dev.txt").read_text().splitlines()
    # This file also feeds setuptools' optional-dependencies.dev metadata.
    parsed = [
        Requirement(line) for line in requirements if line and not line.startswith("#")
    ]
    node = next(
        requirement
        for requirement in parsed
        if requirement.name == "nodejs-wheel-binaries"
    )
    assert str(node.specifier) == "==24.19.0"


def test_pin_ut_checks_node_after_installing_dev_dependencies():
    text = (ROOT / "scripts/ci/ppu_pin_ut.sh").read_text()
    install = text.index(
        'python -m pip install --only-binary=nodejs-wheel-binaries -r "${ROOT_DIR}/requirements/dev.txt"'
    )
    preflight = text.index("python -m nodejs_wheel --version")
    assert install < preflight < text.index("python -m pytest tests/ut")
    result = subprocess.run(
        [sys.executable, "-m", "nodejs_wheel", "--version"],
        env={**os.environ, "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v24.19.0"


@pytest.mark.usefixtures("without_system_node")
def test_node_module_preserves_arguments_environment_and_exit_status():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nodejs_wheel",
            "-e",
            "console.log(process.argv[1]); console.error(process.env.PPU_NODE_TEST); process.exit(23);",
            "argument with spaces",
        ],
        env={**os.environ, "PPU_NODE_TEST": "diagnostic"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 23, result.stderr
    assert result.stdout.strip() == "argument with spaces"
    assert result.stderr.strip() == "diagnostic"


def workflow_condition(
    expression, *, pr_mode=True, attempt=1, plan=None, build="success", artifact="500"
):
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    context = {
        "github": {
            "run_attempt": str(attempt),
            "run_id": "200",
            "repository": "t-head/vllm-sail",
            "ref": "refs/heads/main",
            "event_name": "workflow_dispatch" if pr_mode else "pull_request",
            "event": {"repository": {"default_branch": "main"}},
        },
        "inputs": {
            "pr_mode": pr_mode,
            "existing_build_run_id": "",
            "build_artifact_id": artifact,
            "execution_plan": json.dumps(
                {
                    "run_id": 200,
                    "run_attempt": 1,
                    "plan": plan
                    or {s: {"mode": "run"} for s in ("build", "pin", "smoke")},
                }
            ),
        },
        "needs": {"build": {"result": build}, "validate": {"result": "success"}},
    }
    code = (
        "const {github, inputs, needs} = JSON.parse(process.argv[1]); const fromJSON=JSON.parse; const fromJson=JSON.parse; const always=()=>true; const cancelled=()=>false; const success=()=>true; const format=(s,v)=>s.replace('{0}',v); console.log(Boolean("
        + expression
        + "));"
    )
    result = subprocess.run(
        [sys.executable, "-m", "nodejs_wheel", "-e", code, json.dumps(context)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() == "true"


@pytest.mark.parametrize(
    "filename,job",
    [
        ("ppu-pr-ci.yaml", "validate"),
        ("ppu-pr-ci.yaml", "ppu"),
        ("build-ppu-wheels.yaml", "build"),
        ("build-ppu-wheels.yaml", "pin-unit-tests"),
        ("build-ppu-wheels.yaml", "smoke"),
        ("build-ppu-wheels.yaml", "gate"),
        ("ppu-pin-ut.yaml", "pin-unit-tests"),
        ("e2e-ppu.yaml", "prepare"),
        ("e2e-ppu.yaml", "e2e"),
    ],
)
@pytest.mark.usefixtures("without_system_node")
def test_native_failed_rerun_cannot_schedule_any_on_demand_job(filename, job):
    expression = load_workflow(filename)["jobs"][job].get("if", "true")
    # 模拟成功 validate 被平台复用：所有后继仍必须独立拒绝 attempt 2。
    assert not workflow_condition(expression, attempt=2)
    assert workflow_condition(expression, attempt=1)
    if filename != "ppu-pr-ci.yaml":
        assert workflow_condition(expression, pr_mode=False, attempt=2)


@pytest.mark.usefixtures("without_system_node")
def test_native_rerun_has_independent_hosted_rejection():
    job = load_workflow("ppu-pr-ci.yaml")["jobs"]["reject-native-rerun"]
    assert "needs" not in job and job["runs-on"] == "ubuntu-latest"
    assert workflow_condition(job["if"], attempt=2)
    assert not workflow_condition(job["if"], attempt=1)
    assert "/ci retry" in job["steps"][0]["run"]


@pytest.mark.parametrize("stage,job", [("pin", "pin-unit-tests"), ("smoke", "smoke")])
@pytest.mark.usefixtures("without_system_node")
def test_retry_test_scheduling_requires_proven_build_reuse(stage, job):
    expr = load_workflow("build-ppu-wheels.yaml")["jobs"][job]["if"]
    plan = {s: {"mode": "run"} for s in ("build", "pin", "smoke")}
    assert not workflow_condition(expr, plan=plan, build="skipped")
    plan["build"] = {
        "mode": "reuse",
        "source": {"run_id": 199, "artifact": {"id": 900}},
    }
    assert workflow_condition(expr, plan=plan, build="skipped")
    plan[stage] = {"mode": "reuse"}
    assert not workflow_condition(expr, plan=plan, build="skipped")


def test_dispatch_entry_validates_before_reusable_ppu_chain():
    workflow = load_workflow("ppu-pr-ci.yaml")
    assert set(workflow[True]) == {"workflow_dispatch"}
    assert set(workflow[True]["workflow_dispatch"]["inputs"]) == {
        "pr_number",
        "head_sha",
        "base_sha",
        "request_check_id",
    }
    assert workflow["cache-mode"] == "none"
    validate = workflow["jobs"]["validate"]
    assert validate["runs-on"] == "ubuntu-latest"
    assert any("ppu_ci_command validate" in s.get("run", "") for s in validate["steps"])
    pipeline = workflow["jobs"]["ppu"]
    assert pipeline["name"] == "PPU"
    assert pipeline["needs"] == "validate"
    assert pipeline["permissions"] == {"contents": "read", "actions": "read"}
    assert "secrets" not in pipeline
    assert pipeline["uses"] == "./.github/workflows/build-ppu-wheels.yaml"
    assert pipeline["with"] == {
        "execution_plan": "${{ needs.validate.outputs.execution_plan }}",
        "pr_mode": True,
        "pr_number": "${{ needs.validate.outputs.pr_number }}",
        "pr_head_sha": "${{ needs.validate.outputs.head_sha }}",
    }
    assert "inputs.pr_number" in workflow["concurrency"]["group"]
    assert workflow["concurrency"]["cancel-in-progress"] is False


def test_pr_mode_cannot_fall_into_manual_operator_qualification():
    workflow = load_workflow("build-ppu-wheels.yaml")
    assert "pull_request" in workflow[True]
    assert "workflow_call" in workflow[True]
    assert "inputs.pr_mode" in workflow["jobs"]["build"]["if"]
    assert "!inputs.pr_mode" in workflow["jobs"]["detect-ops-changes"]["if"]
    for job in ("smoke", "gate"):
        assert "inputs.pr_mode" in workflow["jobs"][job]["if"]
    for job in ("pin-unit-tests", "smoke"):
        assert (
            workflow["jobs"][job]["with"]["pr_head_sha"] == "${{ inputs.pr_head_sha }}"
        )
        assert (
            workflow["jobs"][job]["with"]["pr_mode"] == "${{ inputs.pr_mode == true }}"
        )
    build = workflow["jobs"]["build"]
    upload = next(s for s in build["steps"] if s["name"] == "Upload wheels")
    assert upload["with"]["overwrite"] is True
    assert upload["if"] == "success()"
    assert "inputs.pr_number" in workflow["concurrency"]["group"]


@pytest.mark.parametrize(
    "name", ["build-ppu-wheels.yaml", "ppu-pin-ut.yaml", "e2e-ppu.yaml"]
)
def test_pr_test_jobs_disable_cache_and_checkout_bound_sha(name):
    workflow = load_workflow(name)
    assert workflow["cache-mode"] == "none"
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    selected = {
        "build-ppu-wheels.yaml": ["build"],
        "ppu-pin-ut.yaml": ["pin-unit-tests"],
        "e2e-ppu.yaml": ["prepare", "e2e"],
    }[name]
    for job_id in selected:
        steps = workflow["jobs"][job_id]["steps"]
        checkout = next(s for s in steps if s.get("uses") == "actions/checkout@v4")
        assert checkout["with"]["ref"] == "${{ inputs.pr_head_sha || github.sha }}"
        assert checkout["with"]["persist-credentials"] is False
        guard = next(
            s for s in steps if s.get("name") == "Require disabled cache access"
        )
        assert guard.get("uses") == "actions/github-script@v7"
        assert "ACTIONS_CACHE_MODE" in guard["with"]["script"]
        assert steps.index(guard) < steps.index(checkout)
        assert "ACTIONS_CACHE_MODE" not in workflow.get("env", {})
        assert "ACTIONS_CACHE_MODE" not in workflow["jobs"][job_id].get("env", {})
        assert "env" not in guard
        revision = next(
            s for s in steps if s.get("name") == "Verify the checked-out revision"
        )
        assert "rev-parse HEAD" in revision["run"]
        assert steps.index(revision) == steps.index(checkout) + 1
    text = (WORKFLOW.parent / name).read_text()
    assert "secrets: inherit" not in text and "actions/cache" not in text
    if name != "build-ppu-wheels.yaml":
        assert "inputs.pr_mode" in text
        assert "EXPECTED_SAIL_SHA" in text


@pytest.mark.parametrize(
    "filename,job_id",
    [
        ("build-ppu-wheels.yaml", "build"),
        ("ppu-pin-ut.yaml", "pin-unit-tests"),
        ("e2e-ppu.yaml", "prepare"),
        ("e2e-ppu.yaml", "e2e"),
    ],
)
@pytest.mark.parametrize("mode", [None, "", "read", "write", "none"])
@pytest.mark.usefixtures("without_system_node")
def test_cache_guard_checks_runner_injected_node_environment(filename, job_id, mode):
    steps = load_workflow(filename)["jobs"][job_id]["steps"]
    guard = next(s for s in steps if s.get("name") == "Require disabled cache access")
    assert guard.get("uses") == "actions/github-script@v7"
    env = {k: v for k, v in os.environ.items() if k != "ACTIONS_CACHE_MODE"}
    if mode is not None:
        env["ACTIONS_CACHE_MODE"] = mode
    result = subprocess.run(
        [sys.executable, "-m", "nodejs_wheel", "-e", guard["with"]["script"]],
        env=env,
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (mode == "none"), result.stderr
    if mode == "none":
        assert "Cache access verified: none" in result.stdout
    else:
        assert "Actions cache access must be disabled" in result.stderr


@pytest.mark.parametrize(
    "filename,job_id",
    [
        ("build-ppu-wheels.yaml", "build"),
        ("ppu-pin-ut.yaml", "pin-unit-tests"),
        ("e2e-ppu.yaml", "prepare"),
        ("e2e-ppu.yaml", "e2e"),
    ],
)
@pytest.mark.parametrize("different_owner", [False, True])
@pytest.mark.parametrize("expected", ["current", "wrong", "empty"])
def test_revision_guard_handles_checkout_ownership(
    tmp_path, filename, job_id, different_owner, expected
):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        HOME=str(tmp_path),
        LC_ALL="C",
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="CI test",
        GIT_AUTHOR_EMAIL="ci@example.invalid",
        GIT_COMMITTER_NAME="CI test",
        GIT_COMMITTER_EMAIL="ci@example.invalid",
    )

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=checkout, env=env, capture_output=True, text=True
        )

    assert git("init", "-q").returncode == 0
    created = git("commit", "--allow-empty", "-qm", "fixture")
    assert created.returncode == 0, created.stderr
    head = git("rev-parse", "HEAD").stdout.strip()
    assert len(head) == 40
    env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = str(int(different_owner))
    if different_owner:
        assert "dubious ownership" in git("rev-parse", "HEAD").stderr
    env["EXPECTED_SAIL_SHA"] = {"current": head, "wrong": "0" * 40, "empty": ""}[
        expected
    ]
    steps = load_workflow(filename)["jobs"][job_id]["steps"]
    guard = next(s for s in steps if s.get("name") == "Verify the checked-out revision")
    assert (
        guard["env"]["EXPECTED_SAIL_SHA"] == "${{ inputs.pr_head_sha || github.sha }}"
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-euo", "pipefail", "-c", guard["run"]],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "dubious ownership" not in result.stderr, result.stderr
    assert (result.returncode == 0) == (expected == "current"), result.stderr
    assert "safe.directory=*" not in guard["run"]
    assert "git config" not in guard["run"]
    if different_owner:
        assert "dubious ownership" in git("rev-parse", "HEAD").stderr
    assert not (tmp_path / ".gitconfig").exists()


@pytest.mark.parametrize("filename", ["ppu-pin-ut.yaml", "e2e-ppu.yaml"])
@pytest.mark.parametrize("damage", [None, "vllm", "vllm-sail", "missing"])
def test_actual_workflow_manifest_steps_bind_pr_sha(tmp_path, filename, damage):
    from test_ppu_wheel_manifest import make_wheels, record

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    make_wheels(wheels)
    data = record(wheels)
    data["commits"] = {"vllm": "a" * 40, "vllm-sail": "b" * 40}
    if damage == "missing":
        del data["commits"]
    elif damage:
        data["commits"][damage] = "c" * 40
    (wheels / "wheel-manifest.json").write_text(json.dumps(data))
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github/vllm-main-verified.commit").write_text("a" * 40)
    (tmp_path / "scripts/ci").mkdir(parents=True)
    (tmp_path / "scripts/ci/ppu_wheel_manifest.py").symlink_to(
        ROOT / "scripts/ci/ppu_wheel_manifest.py"
    )
    workflow = load_workflow(filename)
    steps = [s for j in workflow["jobs"].values() for s in j["steps"]]
    step = next(
        s
        for s in steps
        if s.get("name")
        in {
            "Verify wheel manifest and pin",
            "Require the current PR artifact and verified upstream pin",
        }
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "REQUIRE_PR_ARTIFACT": "true",
            "WHEELS_DIR": str(wheels),
            "EXPECTED_SAIL_SHA": "b" * 40,
            "GITHUB_SHA": "c" * 40,
            "GITHUB_OUTPUT": str(tmp_path / "output"),
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (damage is None), result.stderr


def test_cache_restriction_propagates_without_nested_overrides():
    pending = ["ppu-pr-ci.yaml"]
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        workflow = load_workflow(name)
        assert workflow.get("cache-mode", "none") == "none"
        for job in workflow["jobs"].values():
            assert job.get("cache-mode", "none") == "none"
            assert "secrets" not in job
            if job.get("uses", "").startswith("./.github/workflows/"):
                pending.append(job["uses"].split("/")[-1])
    assert {
        "ppu-pr-ci.yaml",
        "build-ppu-wheels.yaml",
        "ppu-pin-ut.yaml",
        "e2e-ppu.yaml",
    } <= visited


@pytest.mark.parametrize("mode", ["run", "reuse"])
@pytest.mark.parametrize(
    "state", ["success", "skipped", "failure", "cancelled", "missing"]
)
def test_gate_distinguishes_verified_reuse_from_skipped(tmp_path, mode, state):
    stage = (
        {"mode": "run"}
        if mode == "run"
        else {
            "mode": "reuse",
            "source": {
                "request_id": 1,
                "run_id": 199,
                "run_attempt": 1,
                "job_id": 900,
                "artifact": {"id": 500, "run_id": 199, "digest": "sha256:" + "e" * 64},
            },
        }
    )
    plan = {
        "run_id": 200,
        "run_attempt": 1,
        "plan": {s: stage for s in ("build", "pin", "smoke")},
    }
    results = {j: {"result": state} for j in ("build", "pin-unit-tests", "smoke")}
    if state == "missing":
        results.pop("build")
    step = load_workflow("build-ppu-wheels.yaml")["jobs"]["gate"]["steps"][0]
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", step["run"]],
        env={
            **os.environ,
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
            "JOB_RESULTS": json.dumps(results),
            "EXECUTION_PLAN": json.dumps(plan),
            "PR_MODE": "true",
            "GITHUB_RUN_ID": "200",
            "GITHUB_RUN_ATTEMPT": "1",
        },
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert (result.returncode == 0) == (
        state == ("success" if mode == "run" else "skipped")
    ), result.stderr


@pytest.mark.parametrize(
    "filename,job",
    [
        ("ppu-pin-ut.yaml", "pin-unit-tests"),
        ("e2e-ppu.yaml", "prepare"),
        ("e2e-ppu.yaml", "e2e"),
    ],
)
@pytest.mark.parametrize("artifact", ["", "0", "1,2", "invalid"])
@pytest.mark.usefixtures("without_system_node")
def test_missing_artifact_cannot_schedule_on_demand_download(filename, job, artifact):
    expr = load_workflow(filename)["jobs"][job]["if"]
    assert not workflow_condition(expr, artifact=artifact)
    assert workflow_condition(expr, artifact=artifact, pr_mode=False)


def test_immutable_artifact_handoff_contract():
    workflow = load_workflow("build-ppu-wheels.yaml")
    assert "artifact_id" in workflow["jobs"]["build"]["outputs"]
    for job in ("pin-unit-tests", "smoke"):
        call = workflow["jobs"][job]["with"]
        assert "source.artifact.id" in call["build_artifact_id"]
        assert "source.run_id" in call["build_run_id"]
        assert call["execution_plan"] == "${{ inputs.execution_plan }}"
    for filename, job in (
        ("ppu-pin-ut.yaml", "pin-unit-tests"),
        ("e2e-ppu.yaml", "e2e"),
    ):
        steps = load_workflow(filename)["jobs"][job]["steps"]
        download = next(
            s["with"] for s in steps if s.get("uses") == "actions/download-artifact@v4"
        )
        assert "inputs.build_artifact_id" in download["artifact-ids"]
        assert "!inputs.pr_mode" in download["name"]
        assert download["merge-multiple"] is True


def test_shared_queue_preserves_three_pending_control_writes():
    locks = [
        load_workflow(file)["jobs"][job]["concurrency"]
        for file, job in (
            ("run-ci-command.yml", "command"),
            ("ppu-pr-ci.yaml", "validate"),
            ("ppu-ci-report.yml", "report"),
        )
    ]
    for lock in locks:
        assert lock["group"].startswith("ppu-control-pr-")
        assert lock["cancel-in-progress"] is False
        pending = []
        for task in ("completed-report", "command", "validate"):
            if lock.get("queue", "single") == "single":
                pending.clear()
            pending.append(task)
        assert pending == ["completed-report", "command", "validate"]
        assert lock["queue"] == "max"
    entry = load_workflow("ppu-pr-ci.yaml")["concurrency"]
    assert entry["queue"] == "max" and entry["cancel-in-progress"] is False
    notify = load_workflow("notify-ci-authorized.yml")["jobs"]["notify"]
    assert notify["concurrency"]["group"].startswith("ppu-notification-pr-")


def test_classification_precedes_write_queue_with_separate_help():
    workflow = load_workflow("run-ci-command.yml")
    classify = workflow["jobs"]["classify"]
    assert "concurrency" not in classify
    assert classify["permissions"] == {"contents": "read"}
    assert any("ppu_ci_command classify" in s.get("run", "") for s in classify["steps"])
    command = workflow["jobs"]["command"]
    assert command["needs"] == "classify"
    assert command["if"] == "needs.classify.outputs.kind == 'command'"
    help_job = workflow["jobs"]["help"]
    assert help_job["if"] == "needs.classify.outputs.kind == 'help'"
    assert "concurrency" not in help_job
    assert help_job["permissions"] == {"contents": "read", "issues": "write"}


def test_report_schedule_and_manual_reconcile_never_dispatch_ppu():
    workflow = load_workflow("ppu-ci-report.yml")
    assert workflow[True]["schedule"] == [{"cron": "*/10 * * * *"}]
    assert set(workflow[True]["workflow_dispatch"]["inputs"]) == {"run_id"}
    discover = workflow["jobs"]["identify"]
    assert all(v == "read" for v in discover["permissions"].values())
    assert discover["concurrency"]["cancel-in-progress"] is False
    assert discover["concurrency"]["queue"] == "max"
    group = discover["concurrency"]["group"]
    assert "github.event_name == 'schedule'" in group
    assert "inputs.run_id == ''" in group and "github.run_id" in group
    identity = next(s for s in discover["steps"] if s.get("id") == "identity")
    assert identity["env"]["GITHUB_WORKFLOW_SHA"] == "${{ github.workflow_sha }}"
    assert (
        identity["env"]["CURSOR_FILE"]
        == "${{ runner.temp }}/ppu-report-cursor/cursor.json"
    )
    upload = next(
        s for s in discover["steps"] if "upload-artifact@" in s.get("uses", "")
    )
    assert upload["if"] == "steps.identity.outputs.cursor_written == 'true'"
    assert upload["with"]["name"] == "ppu-report-cursor-v1"
    assert upload["with"]["path"] == identity["env"]["CURSOR_FILE"]
    assert upload["with"]["if-no-files-found"] == "error"
    assert "overwrite" not in upload["with"]
    report = workflow["jobs"]["report"]
    assert report["strategy"]["fail-fast"] is False
    assert "matrix.pr_number" in report["concurrency"]["group"]
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest"
        assert "uses" not in job
        assert job["permissions"].get("actions") == "read"
        assert "github.event.repository.default_branch" in job.get("if", discover["if"])


def test_cpu_events_unchanged_and_token_is_read_only():
    workflow = load_workflow("ut.yaml")
    assert workflow[True] == {
        "pull_request": None,
        "push": {"branches": ["main", "master"]},
        "workflow_dispatch": None,
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["cpu-unit-tests"]["strategy"]["matrix"][
        "python-version"
    ] == ["3.10", "3.11", "3.12", "3.13"]
