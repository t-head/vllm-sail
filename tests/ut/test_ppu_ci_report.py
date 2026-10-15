# SPDX-License-Identifier: Apache-2.0
"""Exercise the real controller and reporter at the GitHub API boundary."""

import copy
import importlib
import json
from pathlib import Path

import pytest
from test_ppu_ci_command import (
    CONTROL,
    HEAD,
    FakeGitHub,
    complete_jobs,
    event,
    executor,
    latest_request,
    launch,
    next_comment,
    validate,
)

NAMES = (
    "PPU wheel build",
    "PPU pin UT / main-verified",
    "PPU E2E smoke / E2E qwen3.8-27b",
    "PPU CI gate",
)


@pytest.fixture
def reporter():
    class Reporter:
        def __getattr__(self, name):
            assert (
                Path(__file__).resolve().parents[2] / "scripts/ci/ppu_ci_report.py"
            ).exists(), "PPU reporter is missing"
            return getattr(importlib.import_module("scripts.ci.ppu_ci_report"), name)

    return Reporter()


def jobs(attempt=1, run_id=200):
    return [
        {
            "id": run_id * 10 + i + (attempt - 1) * 100,
            "run_id": run_id,
            "name": name,
            "status": "completed",
            "conclusion": "success",
            "run_attempt": attempt,
        }
        for i, name in enumerate([*("PPU / " + n for n in NAMES), "validate"])
    ]


class ReportAPI(FakeGitHub):
    def __init__(self):
        super().__init__()
        self.attempts = {1: jobs()}
        complete_jobs(self, {"id": 200})
        self.reads = []

    def get(self, path):
        from urllib.parse import parse_qs

        self.reads.append(path)
        if "per_page=" in path:
            route, query = path.split("?", 1)
            params = parse_qs(query)
            size = int(params["per_page"][0])
            page = int(params.get("page", ["1"])[0])
            if route.endswith("/runs"):
                values = sorted(super().pages(route), key=lambda run: -run["id"])
                key = "workflow_runs"
            elif route.startswith("commits/"):
                values = super().pages(route)
                key = "check_runs"
            else:
                raise AssertionError(f"Unexpected paginated GET {path}")
            return {
                "total_count": len(values),
                key: values[(page - 1) * size : page * size],
            }
        return super().get(path)

    def pages(self, path, key=None):
        if path.startswith("actions/runs/200/attempts/"):
            return copy.deepcopy(self.attempts[int(path.split("/")[4])])
        return super().pages(path, key)


@pytest.fixture
def setup():
    from scripts.ci import ppu_ci_command as command

    api = ReportAPI()
    context = {
        "run_id": 100,
        "control_sha": CONTROL,
        "default_branch": "main",
        "trusted_users": "",
    }
    request = launch(command, api, context)
    run = executor(api, request)
    validate(command, api, run, context)
    run.update(status="completed", conclusion="success")
    return api, request, run, command, context


def snapshot(run):
    return {"action": "completed", "workflow_run": copy.deepcopy(run)}


def gate(api):
    return max(
        (c for c in api.checks.values() if c["name"] == "PPU CI gate"),
        key=lambda c: c["id"],
    )


def test_report_uses_bound_head_not_control_sha(reporter, setup):
    api, _, run, _, _ = setup
    api.pr["head"]["sha"] = "d" * 40
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "success"
    assert all(c["head_sha"] == HEAD for c in api.checks.values())
    assert gate(api)["details_url"].endswith("/runs/200")


@pytest.mark.parametrize(
    "state",
    [
        "failure",
        "cancelled",
        "timed_out",
        "skipped",
        "neutral",
        "action_required",
        "missing",
    ],
)
def test_non_success_stage_never_passes_gate(reporter, setup, state):
    api, _, run, _, _ = setup
    if state == "missing":
        api.attempts[1].pop(2)
    else:
        api.attempts[1][2]["conclusion"] = state
    reporter.report(snapshot(run), api)
    assert gate(api)["status"] == "completed"
    assert gate(api)["conclusion"] != "success"


def test_retry_reuses_successful_build_across_runs(reporter, setup):
    api, request, run, command, context = setup
    run.update(conclusion="failure")
    api.attempts[1][2]["conclusion"] = "failure"
    api.attempts[1][3]["conclusion"] = "failure"
    reporter.report(snapshot(run), api)
    before = copy.deepcopy(api.checks)
    next_comment(api, context)
    command.handle_event(event(api), api, context)
    assert gate(api)["status"] == "queued"
    second = executor(api, latest_request(api), 201)
    validate(command, api, second, context)
    api.job_runs[201] = jobs(run_id=201)[2:]
    second.update(status="completed", conclusion="success")
    reporter.report(snapshot(second), api)
    assert gate(api)["conclusion"] == "success"
    assert {k: api.checks[k] for k in before} == before
    build = next(
        c
        for c in api.checks.values()
        if c.get("external_id") == f"ppu-ci:{latest_request(api)['id']}:PPU wheel build"
    )
    assert "/runs/200/job/2000" in build["output"]["summary"]
    assert "reuse" in build["output"]["summary"].lower()
    assert str(request["id"]) in build["output"]["summary"]


@pytest.mark.parametrize("code", [404, 410, 503])
@pytest.mark.parametrize("missing", ["run", "request", "jobs"])
def test_missing_retry_source_finishes_failed_report(
    reporter, setup, monkeypatch, code, missing
):
    from urllib.error import HTTPError

    api, request, run, command, context = setup
    run["conclusion"] = "failure"
    api.attempts[1][2]["conclusion"] = "failure"
    api.attempts[1][3]["conclusion"] = "failure"
    reporter.report(snapshot(run), api)
    next_comment(api, context)
    command.handle_event(event(api), api, context)
    second = executor(api, latest_request(api), 201)
    validate(command, api, second, context)
    api.job_runs[201] = jobs(run_id=201)[2:]
    second.update(status="completed", conclusion="success")
    original_get, original_pages = api.get, api.pages
    path = {
        "run": "actions/runs/200",
        "request": f"check-runs/{request['id']}",
        "jobs": "actions/runs/200/attempts/1/jobs",
    }[missing]

    def get(target):
        if target == path:
            raise HTTPError("https://api.github.com", code, "Unavailable", {}, None)
        return original_get(target)

    def pages(target, key=None):
        if target == path:
            raise HTTPError("https://api.github.com", code, "Unavailable", {}, None)
        return original_pages(target, key)

    monkeypatch.setattr(api, "get", get)
    monkeypatch.setattr(api, "pages", pages)
    if code == 503:
        with pytest.raises(HTTPError):
            reporter.reconcile(api, run_id=201, pr_number=7)
        assert latest_request(api)["status"] == "in_progress"
        assert gate(api)["status"] == "in_progress"
        return
    for _ in range(2):
        reporter.reconcile(api, run_id=201, pr_number=7)
        assert latest_request(api)["status"] == "completed"
        assert gate(api)["conclusion"] == "failure"
        assert "Missing retry evidence" in gate(api)["output"]["summary"]
    assert reporter.finalized(
        api,
        second,
        latest_request(api),
        json.loads(latest_request(api)["output"]["summary"]),
    )
    dispatched = len(api.dispatches)
    next_comment(api, context, "/ci run")
    command.handle_event(event(api), api, context)
    assert len(api.dispatches) == dispatched + 1


def test_old_callback_cannot_complete_queued_retry(reporter, setup):
    api, _, run, command, context = setup
    run.update(conclusion="failure")
    old = snapshot(run)
    api.comment.update(id=18, body="/ci retry")
    command.handle_event(event(api), api, context)
    before = copy.deepcopy(api.checks)
    reporter.report(old, api)
    assert api.checks == before


def test_rebuilt_wheels_invalidate_older_test_successes(reporter, setup):
    api, _, run, _, _ = setup
    run.update(run_attempt=2)
    api.attempts[2] = [jobs(2)[0], jobs(2)[3]]
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] != "success"


@pytest.mark.parametrize(
    "mutation", ["workflow", "app", "title", "branch", "sha", "repository", "duplicate"]
)
def test_untrusted_callback_cannot_write_checks(reporter, setup, mutation):
    from scripts.ci.ppu_ci_common import Rejected

    api, request, run, _, _ = setup
    if mutation == "workflow":
        run["workflow_id"] = 999
    elif mutation == "app":
        api.checks[request["id"]]["app"]["id"] = 999
    elif mutation == "title":
        run["display_title"] += " extra"
    elif mutation == "branch":
        run["head_branch"] = "evil"
    elif mutation == "sha":
        run["head_sha"] = HEAD
    elif mutation == "repository":
        run["repository"]["full_name"] = "evil/fork"
    else:
        run["id"] = 201
        api.runs[201] = run
    before = copy.deepcopy(api.checks)
    with pytest.raises(Rejected):
        reporter.report(snapshot(run), api)
    assert api.checks == before


def test_cancelled_execution_never_succeeds_even_with_green_jobs(reporter, setup):
    api, _, run, _, _ = setup
    run.update(conclusion="cancelled")
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "cancelled"


def test_missing_validator_binding_never_succeeds(reporter, setup):
    api, request, run, _, _ = setup
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    record.pop("validation", None)
    api.checks[request["id"]]["output"]["summary"] = json.dumps(record)
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] != "success"


def test_duplicate_job_names_fail_closed(reporter, setup):
    api, _, run, _, _ = setup
    api.attempts[1].append(copy.deepcopy(api.attempts[1][0]))
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] != "success"


def test_started_callback_updates_link_but_not_success(reporter, setup):
    api, _, run, _, _ = setup
    run.update(status="in_progress", conclusion=None)
    reporter.report({"action": "in_progress", "workflow_run": copy.deepcopy(run)}, api)
    assert gate(api)["status"] == "in_progress"
    assert gate(api).get("conclusion") != "success"


def test_cancelled_request_can_retry_with_fresh_authorization(reporter, setup):
    api, request, run, command, context = setup
    run.update(status="in_progress", conclusion=None)
    api.comment.update(id=18, body="/ci cancel")
    command.handle_event(event(api), api, context)
    run.update(status="completed", conclusion="cancelled")
    api.comment.update(id=19, body="/ci retry")
    command.handle_event(event(api), api, context)
    assert not api.retries and len(api.dispatches) == 2
    second = executor(api, latest_request(api), 201)
    record = validate(command, api, second, context)
    assert not record.get("cancelled")
    assert json.loads(api.checks[request["id"]]["output"]["summary"])["cancelled"]


def test_old_same_sha_callback_does_not_overwrite_new_request(reporter, setup):
    api, _, run, command, context = setup
    old = snapshot(run)
    reporter.report(old, api)
    context["run_id"] = 101
    api.runs[101] = {**copy.deepcopy(api.runs[100]), "id": 101}
    api.comment.update(id=18, body="/ci run")
    command.handle_event(event(api), api, context)
    assert len(api.checks) == 10
    before = copy.deepcopy(api.checks)
    reporter.report(old, api)
    assert api.checks == before


def test_cancel_does_not_invalidate_completed_results(reporter, setup):
    api, _, run, command, context = setup
    reporter.report(snapshot(run), api)
    before = copy.deepcopy(api.checks)
    api.comment.update(id=18, body="/ci cancel")
    command.handle_event(event(api), api, context)
    assert not api.cancellations
    assert api.checks == before
    assert "No active" in api.messages[-1]["body"]


def test_jobs_api_failure_invalidates_old_gate(reporter, setup, monkeypatch):
    api, _, run, _, _ = setup
    reporter.report(snapshot(run), api)
    original = api.pages

    def fail_jobs(path, key=None):
        if "/attempts/" in path:
            raise OSError("GitHub is unavailable")
        return original(path, key)

    monkeypatch.setattr(api, "pages", fail_jobs)
    with pytest.raises(OSError):
        reporter.report(snapshot(run), api)
    assert gate(api)["status"] != "completed"


@pytest.mark.parametrize("failed_write", range(1, 7))
def test_reconcile_recovers_every_partial_report_write(
    reporter, setup, monkeypatch, failed_write
):
    api, _, run, _, _ = setup
    original = api.patch
    writes = 0

    def fail_once(path, payload):
        nonlocal writes
        writes += 1
        if writes == failed_write:
            raise OSError("Reporter was interrupted")
        return original(path, payload)

    monkeypatch.setattr(api, "patch", fail_once)
    with pytest.raises(OSError):
        reporter.report(snapshot(run), api)
    monkeypatch.setattr(api, "patch", original)
    before = copy.deepcopy(api.checks)
    targets = reporter.discover({"schedule": "*/10 * * * *"}, api)
    assert api.checks == before
    assert targets == [{"pr_number": 7, "run_id": 200}]
    reporter.reconcile(api, targets[0]["run_id"], targets[0]["pr_number"])
    assert all(
        c["status"] == "completed" and c["conclusion"] == "success"
        for c in api.checks.values()
    )
    assert reporter.discover({}, api) == []
    assert len(api.dispatches) == 1 and not api.retries


def add_history(setup, count):
    api, _, _, command, context = setup
    for index in range(1, count):
        api.pr["head"]["sha"] = f"{index:040x}"
        api.comment["id"] += 1
        command.create_request(api, api.pr, api.comment, context, "/ci run")
        request = latest_request(api)
        other = executor(api, request, 200 + index)
        other["display_title"] = other["display_title"].replace(
            HEAD, api.pr["head"]["sha"]
        )
        validate(command, api, other, context)
        api.job_runs[other["id"]] = jobs(run_id=other["id"])
        other.update(status="completed", conclusion="success")


def test_discovery_bounds_reads_and_resumes_past_finalized_history(reporter, setup):
    api, _, run, _, _ = setup
    add_history(setup, 101)
    for run_id in range(201, 301):
        reporter.reconcile(api, run_id=run_id, pr_number=7)
    before = copy.deepcopy(api.checks)
    cursor, visited = {}, []
    for _ in range(30):
        api.reads.clear()
        targets = reporter.discover({}, api, cursor=cursor)
        assert len(api.reads) <= 80
        assert len(targets) <= 5
        visited.extend(t["run_id"] for t in targets)
        if cursor["before_run_id"] is None:
            break
    assert visited == [run["id"]]
    assert api.checks == before
    assert len(api.dispatches) == 101 and not api.retries


def test_discovery_drains_batches_and_revisits_failed_reports(reporter, setup):
    api, _, _, _, _ = setup
    add_history(setup, 16)
    cursor, visited = {}, set()
    for _ in range(6):
        targets = reporter.discover({}, api, cursor=cursor)
        assert len(targets) <= 5
        visited.update(t["run_id"] for t in targets)
        for target in targets:
            if target["run_id"] != 210:
                reporter.reconcile(api, **target)
        if cursor["before_run_id"] is None:
            break
    assert visited == set(range(200, 216))
    revisited = []
    for _ in range(6):
        revisited.extend(reporter.discover({}, api, cursor=cursor))
        if cursor["before_run_id"] is None:
            break
    assert revisited == [{"pr_number": 7, "run_id": 210}]


def test_discovery_survives_new_arrivals_deleted_anchor_and_thousand_runs(
    reporter, setup
):
    api, _, original, _, _ = setup
    for run_id in range(201, 1501):
        api.runs[run_id] = {
            **copy.deepcopy(original),
            "id": run_id,
            "display_title": "automatic",
        }
    cursor = {"before_run_id": 450}
    del api.runs[450]
    for run_id in range(1501, 1601):
        api.runs[run_id] = {
            **copy.deepcopy(original),
            "id": run_id,
            "display_title": "new arrival",
        }
    api.reads.clear()
    assert reporter.discover({}, api, cursor=cursor) == [
        {"pr_number": 7, "run_id": 200}
    ]
    assert cursor["before_run_id"] is None
    assert len(api.reads) <= 80
    pages = [p for p in api.reads if "/runs?" in p]
    assert any(int(p.rsplit("=", 1)[-1]) > 10 for p in pages)
    assert not any("created=" in p or "event=" in p or "status=" in p for p in pages)


def test_arrivals_during_pagination_cannot_reset_an_unfinished_sweep(
    reporter, setup, monkeypatch
):
    api, _, original_run, _, _ = setup
    for run_id in range(201, 400):
        api.runs[run_id] = {
            **copy.deepcopy(original_run),
            "id": run_id,
            "display_title": "automatic",
        }
    original_get = api.get
    inserts = 0

    def get(path):
        nonlocal inserts
        result = original_get(path)
        if path == "actions/workflows/ppu-pr-ci.yaml/runs?per_page=100&page=1":
            start = max(api.runs) + 1
            for run_id in range(start, start + 100):
                api.runs[run_id] = {
                    **copy.deepcopy(original_run),
                    "id": run_id,
                    "display_title": "concurrent arrival",
                }
            inserts += 1
        return result

    monkeypatch.setattr(api, "get", get)
    cursor, found = {}, []
    for _ in range(4):
        api.reads.clear()
        found.extend(reporter.discover({}, api, cursor=cursor))
        assert len(api.reads) <= 80
        if found:
            break
        assert cursor["before_run_id"] is not None
    assert found == [{"pr_number": 7, "run_id": 200}]
    assert inserts


@pytest.mark.parametrize("empty_tail", [False, True])
def test_deletion_during_pagination_preserves_the_unfinished_anchor(
    reporter, setup, monkeypatch, empty_tail
):
    from scripts.ci.ppu_ci_common import Rejected

    api, _, original_run, _, _ = setup
    for run_id in range(201, 401):
        api.runs[run_id] = {
            **copy.deepcopy(original_run),
            "id": run_id,
            "display_title": "automatic",
        }
    if not empty_tail:
        api.runs[199] = {
            **copy.deepcopy(original_run),
            "id": 199,
            "display_title": "automatic",
        }
    original_get = api.get

    def get(path):
        result = original_get(path)
        if path == "actions/workflows/ppu-pr-ci.yaml/runs?per_page=100&page=2":
            api.runs.pop(400, None)
        return result

    monkeypatch.setattr(api, "get", get)
    cursor = {"before_run_id": 201}
    api.reads.clear()
    with pytest.raises(Rejected, match="pagination changed"):
        reporter.discover({}, api, cursor=cursor)
    assert cursor == {"before_run_id": 201}
    assert len(api.reads) <= 80
    assert reporter.discover({}, api, cursor=cursor) == [
        {"pr_number": 7, "run_id": 200}
    ]
    assert cursor["before_run_id"] is None


def test_discovery_budget_covers_inner_pagination_without_stalling(
    reporter, setup, monkeypatch
):
    api, _, _, _, _ = setup
    original = api.get

    def unbounded_checks(path):
        if path.startswith("commits/"):
            api.reads.append(path)
            return {"check_runs": [copy.deepcopy(gate(api)) for _ in range(100)]}
        return original(path)

    monkeypatch.setattr(api, "get", unbounded_checks)
    cursor = {}
    api.reads.clear()
    assert reporter.discover({}, api, cursor=cursor) == [
        {"pr_number": 7, "run_id": 200}
    ]
    assert len(api.reads) <= 80
    assert cursor["before_run_id"] == 200
    assert reporter.discover({}, api, cursor=cursor) == []
    assert cursor["before_run_id"] is None


def test_selected_discovery_does_not_advance_cursor_or_list_history(reporter, setup):
    api, _, _, _, _ = setup
    cursor = {"before_run_id": 300}
    api.reads.clear()
    assert reporter.discover({"inputs": {"run_id": "200"}}, api, cursor=cursor) == [
        {"pr_number": 7, "run_id": 200}
    ]
    assert cursor == {"before_run_id": 300}
    assert not any("/runs?" in p for p in api.reads)


def test_discovery_api_outage_keeps_previous_checkpoint(reporter, setup, monkeypatch):
    from urllib.error import HTTPError

    api, _, _, _, _ = setup
    cursor = {"before_run_id": 300}
    original = api.get

    def unavailable(path):
        if path.startswith("check-runs/"):
            raise HTTPError("https://api.github.com", 503, "unavailable", {}, None)
        return original(path)

    monkeypatch.setattr(api, "get", unavailable)
    with pytest.raises(HTTPError):
        reporter.discover({}, api, cursor=cursor)
    assert cursor == {"before_run_id": 300}


def test_discovery_wrapper_bounds_all_reads_and_has_no_writes(reporter, setup):
    api, _, _, _, _ = setup
    bounded = reporter.DiscoveryAPI(api)
    for _ in range(80):
        bounded.consume()
    with pytest.raises(reporter.ReadBudget):
        bounded.get("")
    assert bounded.reads == 80
    assert not hasattr(bounded, "post") and not hasattr(bounded, "patch")


@pytest.mark.parametrize("selected", [None, "manual", "callback"])
@pytest.mark.parametrize("failure", [None, "restore", "encode"])
def test_discovery_cli_persists_only_successful_sweeps(
    setup, tmp_path, monkeypatch, selected, failure
):
    import sys
    from types import SimpleNamespace

    from scripts import ci
    from scripts.ci import ppu_ci_report as module

    api, _, run, _, _ = setup
    calls = []

    def restore(bounded):
        assert isinstance(bounded, module.DiscoveryAPI)
        calls.append("restore")
        if failure == "restore":
            raise OSError("Checkpoint API unavailable")
        return {"before_run_id": 300}

    def encode(bounded, cursor, run_id, attempt, sha):
        assert bounded.reads <= 80
        assert (run_id, attempt, sha) == (900, 1, CONTROL)
        calls.append("encode")
        if failure == "encode":
            raise OSError("Checkpoint provenance unavailable")
        return json.dumps(cursor)

    checkpoint = SimpleNamespace(restore=restore, encode=encode)
    monkeypatch.setattr(ci, "ppu_ci_checkpoint", checkpoint, raising=False)
    monkeypatch.setattr(module, "GitHub", lambda *args: api)
    monkeypatch.setattr(sys, "argv", ["ppu_ci_report", "discover"])
    ev = {"schedule": "*/10 * * * *"}
    if selected == "manual":
        ev = {"inputs": {"run_id": "200"}}
    elif selected == "callback":
        ev = {"workflow_run": run}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(ev))
    cursor_path = tmp_path / "checkpoint" / "cursor.json"
    output = tmp_path / "output"
    for key, value in {
        "GITHUB_EVENT_PATH": event_path,
        "GITHUB_OUTPUT": output,
        "CURSOR_FILE": cursor_path,
        "GITHUB_RUN_ID": "900",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_WORKFLOW_SHA": CONTROL,
        "GITHUB_REPOSITORY": api.repo,
        "GH_TOKEN": "test-token",
    }.items():
        monkeypatch.setenv(key, str(value))
    if failure and selected is None:
        with pytest.raises(OSError):
            module.main()
        assert not cursor_path.exists() and not output.exists()
    else:
        module.main()
        assert "has_targets=true" in output.read_text()
        if selected:
            assert not calls and not cursor_path.exists()
            assert "cursor_written=true" not in output.read_text()
        else:
            assert calls == ["restore", "encode"]
            assert json.loads(cursor_path.read_text()) == {"before_run_id": None}
            assert "cursor_written=true" in output.read_text()


@pytest.mark.parametrize("warm", [False, True])
def test_discovery_cli_uses_real_checkpoint_schema(setup, tmp_path, monkeypatch, warm):
    import sys

    from test_ppu_ci_checkpoint import archive

    from scripts.ci import ppu_ci_checkpoint as checkpoint
    from scripts.ci import ppu_ci_report as module

    api, _, _, _, _ = setup
    original = api.get
    producer = {
        "id": 900,
        "run_attempt": 1,
        "workflow_id": 30,
        "path": ".github/workflows/ppu-ci-report.yml",
        "event": "schedule",
        "head_branch": "main",
        "head_sha": CONTROL,
        "repository": {"id": 2, "full_name": api.repo},
        "head_repository": {"id": 2, "full_name": api.repo},
    }
    previous = {
        "schema": 1,
        "repository": api.repo,
        "executor_workflow_id": 20,
        "producer": {"run_id": 899, "attempt": 1, "control_sha": CONTROL},
        "before_run_id": 300,
    }
    artifact = {
        "id": 500,
        "name": checkpoint.NAME,
        "size_in_bytes": 400,
        "expired": False,
        "expires_at": "2999-01-01T00:00:00Z",
        "workflow_run": {
            "id": 899,
            "repository_id": 2,
            "head_repository_id": 2,
            "head_sha": CONTROL,
            "head_branch": "main",
        },
    }

    def get(path):
        response = None
        if path == "":
            response = {"id": 2, "full_name": api.repo, "default_branch": "main"}
        elif path == "actions/workflows/ppu-ci-report.yml":
            response = {"id": 30, "path": producer["path"]}
        elif path in {"actions/runs/900", "actions/runs/899"}:
            response = {**producer, "id": int(path.rsplit("/", 1)[-1])}
        elif path.startswith("actions/artifacts?"):
            response = {
                "total_count": int(warm),
                "artifacts": [artifact] if warm else [],
            }
        if response is not None:
            api.reads.append(path)
            return response
        return original(path)

    monkeypatch.setattr(api, "get", get)
    monkeypatch.setattr(
        checkpoint, "_download", lambda *args: archive(json.dumps(previous).encode())
    )
    monkeypatch.setattr(module, "GitHub", lambda *args: api)
    monkeypatch.setattr(sys, "argv", ["ppu_ci_report", "discover"])
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps({"schedule": "*/10 * * * *"}))
    cursor_path = tmp_path / "cursor.json"
    output = tmp_path / "output"
    for key, value in {
        "GITHUB_EVENT_PATH": event_path,
        "GITHUB_OUTPUT": output,
        "CURSOR_FILE": cursor_path,
        "GITHUB_RUN_ID": "900",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_WORKFLOW_SHA": CONTROL,
        "GITHUB_REPOSITORY": api.repo,
        "GH_TOKEN": "test-token",
    }.items():
        monkeypatch.setenv(key, str(value))
    api.reads.clear()
    module.main()
    assert len(api.reads) <= 80
    saved = json.loads(cursor_path.read_text())
    assert saved["producer"] == {"run_id": 900, "attempt": 1, "control_sha": CONTROL}
    assert saved["before_run_id"] is None
    assert "has_targets=true" in output.read_text()
    assert "cursor_written=true" in output.read_text()


def test_repair_rechecks_supersession_and_lock_identity(reporter, setup):
    from scripts.ci.ppu_ci_common import Rejected

    api, _, run, command, context = setup
    target = reporter.discover({}, api)[0]
    with pytest.raises(Rejected):
        reporter.reconcile(api, run_id=200, pr_number=8)
    run["conclusion"] = "failure"
    next_comment(api, context)
    command.handle_event(event(api), api, context)
    before = copy.deepcopy(api.checks)
    reporter.reconcile(api, **target)
    assert api.checks == before
    assert reporter.discover({}, api) == []


def test_reconcile_preserves_cancellation_and_old_head(reporter, setup):
    api, request, run, command, _ = setup
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    record["cancelled"] = True
    command.save_record(
        api, request, record, status="completed", conclusion="cancelled"
    )
    api.pr["head"]["sha"] = "f" * 40
    reporter.reconcile(api, run_id=run["id"], pr_number=7)
    assert all(
        c["head_sha"] == HEAD and c["conclusion"] == "cancelled"
        for c in api.checks.values()
    )
    assert reporter.discover({}, api) == []


def test_discovery_manual_target_and_native_rerun_recheck(reporter, setup):
    api, _, run, _, _ = setup
    reporter.report(snapshot(run), api)
    assert reporter.discover({}, api) == []
    assert reporter.discover({"inputs": {"run_id": "200"}}, api) == [
        {"pr_number": 7, "run_id": 200}
    ]
    run["run_attempt"] = 2
    target = reporter.discover({}, api)[0]
    reporter.reconcile(api, **target)
    assert gate(api)["conclusion"] == "failure"
    assert reporter.discover({}, api) == []


@pytest.mark.parametrize("code", [404, 410, 503])
def test_discovery_isolates_nonexistent_requests_but_not_api_outages(
    reporter, setup, monkeypatch, code
):
    from urllib.error import HTTPError

    api, _, run, _, _ = setup
    bogus = copy.deepcopy(run)
    bogus.update(id=199, display_title=f"PPU PR 7 head {HEAD} request 9999")
    api.runs[199] = bogus
    original = api.get

    def get(path):
        if path == "check-runs/9999":
            raise HTTPError("https://api.github.com", code, "Unavailable", {}, None)
        return original(path)

    monkeypatch.setattr(api, "get", get)
    if code == 503:
        with pytest.raises(HTTPError):
            reporter.discover({}, api)
    else:
        assert reporter.discover({}, api) == [{"pr_number": 7, "run_id": 200}]


def test_reconciliation_after_queue_capacity_or_reporter_cancellation(reporter, setup):
    api, _, run, _, _ = setup
    # max 队列已满时本次回写未运行；后续定时 discovery 仍会发现它。
    pending, cancelled = list(range(100)), []
    if len(pending) == 100:
        cancelled.append(run["id"])
    assert cancelled == [200]
    target = reporter.discover({}, api)[0]
    # 模拟已进入队列的 repair 又被取消，无需伪造 checks 即可再次恢复。
    assert reporter.discover({}, api) == [target]
    reporter.reconcile(api, **target)
    assert gate(api)["conclusion"] == "success"
    # 终态之后若发生部分回写，finalized 不能仅信请求上的完成标记。
    gate(api).update(status="in_progress")
    assert reporter.discover({}, api) == [target]
    reporter.reconcile(api, **target)
    assert reporter.discover({}, api) == []


def test_uncertain_dispatch_without_run_is_never_redispatched(reporter, setup):
    api, _, run, command, context = setup
    reporter.report(snapshot(run), api)
    next_comment(api, context, body="/ci run")
    api.fail_dispatch = True
    command.handle_event(event(api), api, context)
    before = copy.deepcopy(api.checks)
    assert reporter.discover({}, api) == []
    assert api.checks == before and len(api.dispatches) == 1


def test_native_rerun_never_reports_green(reporter, setup):
    api, _, run, _, _ = setup
    run["run_attempt"] = 2
    api.attempts[2] = jobs(2)
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "failure"


@pytest.mark.parametrize(
    "damage", ["run", "attempt", "plan", "missing_job", "failed_job"]
)
def test_validator_requires_run_bound_plan_and_successful_job(reporter, setup, damage):
    api, request, run, _, _ = setup
    record = json.loads(api.checks[request["id"]]["output"]["summary"])
    if damage == "run":
        record["validation"]["run_id"] = 999
    elif damage == "attempt":
        record["validation"]["run_attempt"] = 2
    elif damage == "plan":
        record["validation"]["plan"]["unknown"] = {"mode": "reuse"}
    elif damage == "missing_job":
        api.attempts[1].pop()
    else:
        api.attempts[1][-1]["conclusion"] = "failure"
    api.checks[request["id"]]["output"]["summary"] = json.dumps(record)
    reporter.report(snapshot(run), api)
    assert gate(api)["conclusion"] == "failure"
