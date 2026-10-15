# SPDX-License-Identifier: Apache-2.0
"""Exercise the real CI controller with an in-memory GitHub API boundary."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HEAD, BASE, CONTROL = "a" * 40, "b" * 40, "c" * 40
REPO = "t-head/vllm-sail"


@pytest.fixture
def controller():
    class Controller:
        def __getattr__(self, name):
            path = ROOT / "scripts/ci/ppu_ci_command.py"
            assert path.is_file(), "The trusted PPU comment controller is missing"
            return getattr(importlib.import_module("scripts.ci.ppu_ci_command"), name)

    return Controller()


class FakeGitHub:
    repo = REPO

    def __init__(self):
        self.pr = {
            "number": 7,
            "state": "open",
            "draft": False,
            "user": {"login": "author"},
            "labels": [{"name": "ready"}],
            "head": {
                "sha": HEAD,
                "ref": "main",
                "repo": {"id": 1, "full_name": "author/fork"},
            },
            "base": {"sha": BASE, "ref": "main", "repo": {"id": 2, "full_name": REPO}},
        }
        self.comment = {
            "id": 17,
            "body": "/ci run",
            "created_at": "2026-10-13T08:00:00Z",
            "updated_at": "2026-10-13T08:00:00Z",
            "user": {"id": 1, "login": "author", "type": "User"},
            "issue_url": f"https://api.github.com/repos/{REPO}/issues/7",
        }
        self.permissions = {"author": "read", "maintainer": "write"}
        self.reviews = []
        self.decision = "REVIEW_REQUIRED"
        self.behind = 0
        self.checks = {}
        self.runs = {
            100: {
                "id": 100,
                "workflow_id": 10,
                "path": ".github/workflows/run-ci-command.yml",
                "event": "issue_comment",
                "head_sha": CONTROL,
                "head_branch": "main",
                "repository": {"full_name": REPO},
                "run_attempt": 1,
                "status": "in_progress",
                "conclusion": None,
            }
        }
        self.messages = []
        self.dispatches = []
        self.retries = []
        self.cancellations = []
        self.pr_reads = 0
        self.change_on_read = None
        self.fail_dispatch = False
        self.job_runs = {}
        self.artifacts = {}

    def get(self, path):
        if path == "":
            return {"default_branch": "main", "full_name": REPO}
        if path == "pulls/7":
            self.pr_reads += 1
            result = copy.deepcopy(self.pr)
            if self.change_on_read == self.pr_reads:
                result["head"]["sha"] = "d" * 40
            return result
        if path == f"issues/comments/{self.comment['id']}":
            return copy.deepcopy(self.comment)
        if path.startswith("collaborators/"):
            return {"permission": self.permissions.get(path.split("/")[1], "read")}
        if path.startswith("compare/"):
            return {"behind_by": self.behind}
        if path.startswith("check-runs/"):
            return copy.deepcopy(self.checks[int(path.split("/")[1])])
        if path.startswith("actions/workflows/"):
            filename = path.split("/")[2]
            return {
                "id": {"run-ci-command.yml": 10, "ppu-pr-ci.yaml": 20}[filename],
                "path": f".github/workflows/{filename}",
            }
        if path.startswith("actions/artifacts/"):
            from urllib.error import HTTPError

            artifact = self.artifacts.get(int(path.split("/")[2]))
            if artifact is None:
                raise HTTPError("https://api.github.com", 404, "missing", {}, None)
            return copy.deepcopy(artifact)
        if path.startswith("actions/runs/"):
            return copy.deepcopy(self.runs[int(path.split("/")[2])])
        raise AssertionError(f"Unexpected GET {path}")

    def pages(self, path, key=None):
        if path.startswith("actions/runs/"):
            run_id = int(path.split("/")[2])
            if path.endswith("/jobs"):
                return copy.deepcopy(self.job_runs.get(run_id, []))
            if path.endswith("/artifacts"):
                return copy.deepcopy(
                    [
                        a
                        for a in self.artifacts.values()
                        if a["workflow_run"]["id"] == run_id
                    ]
                )
        if path.startswith("commits/") and "/check-runs" in path:
            sha = path.split("/")[1]
            return [
                copy.deepcopy(c) for c in self.checks.values() if c["head_sha"] == sha
            ]
        if path == "pulls/7/reviews":
            return copy.deepcopy(self.reviews)
        if path == "issues/7/comments":
            return copy.deepcopy(self.messages)
        if path.startswith("actions/workflows/"):
            filename = path.split("/")[2]
            return [
                copy.deepcopy(r)
                for r in self.runs.values()
                if r["path"] == f".github/workflows/{filename}"
            ]
        raise AssertionError(f"Unexpected list {path}")

    def review_decision(self, number):
        assert number == 7
        return self.decision

    def post(self, path, payload=None):
        if path == "check-runs":
            check = {
                "id": len(self.checks) + 1,
                "app": {"id": 15368},
                **copy.deepcopy(payload),
            }
            self.checks[check["id"]] = check
            return copy.deepcopy(check)
        if path.endswith("/dispatches"):
            if self.fail_dispatch:
                raise OSError("Dispatch response was lost")
            self.dispatches.append(copy.deepcopy(payload))
            return None
        if path == "issues/7/comments":
            self.messages.append({"user": {"login": "github-actions[bot]"}, **payload})
            return {}
        if path.endswith("/rerun-failed-jobs"):
            self.retries.append(int(path.split("/")[2]))
            return None
        if path.endswith("/cancel"):
            self.cancellations.append(int(path.split("/")[2]))
            return None
        raise AssertionError(f"Unexpected POST {path}")

    def patch(self, path, payload):
        check_id = int(path.split("/")[1])
        self.checks[check_id].update(copy.deepcopy(payload))
        return copy.deepcopy(self.checks[check_id])


@pytest.fixture
def api():
    return FakeGitHub()


@pytest.fixture
def context():
    return {
        "run_id": 100,
        "control_sha": CONTROL,
        "default_branch": "main",
        "trusted_users": "",
    }


def event(api):
    return {
        "action": "created",
        "issue": {"number": 7, "pull_request": {}},
        "comment": copy.deepcopy(api.comment),
    }


def launch(controller, api, context):
    controller.handle_event(event(api), api, context)
    return next(c for c in api.checks.values() if c["name"] == "PPU CI request")


def executor(api, request, run_id=200):
    run = {
        "id": run_id,
        "workflow_id": 20,
        "event": "workflow_dispatch",
        "path": ".github/workflows/ppu-pr-ci.yaml",
        "head_sha": CONTROL,
        "head_branch": "main",
        "repository": {"full_name": REPO},
        "display_title": f"PPU PR 7 head {HEAD} request {request['id']}",
        "run_attempt": 1,
        "status": "in_progress",
        "conclusion": None,
    }
    api.runs[run_id] = run
    return run


def latest_request(api):
    return max(
        (c for c in api.checks.values() if c["name"] == "PPU CI request"),
        key=lambda c: c["id"],
    )


def next_comment(api, context, body="/ci retry", actor="author"):
    context["run_id"] += 1
    api.runs[context["run_id"]] = {
        **copy.deepcopy(api.runs[100]),
        "id": context["run_id"],
    }
    api.comment.update(
        id=api.comment["id"] + 1,
        body=body,
        user={"id": 1 if actor == "author" else 2, "login": actor, "type": "User"},
    )


def validate(controller, api, run, context):
    return controller.validate_execution(
        api, api.dispatches[-1]["inputs"], run, {**context, "run_id": run["id"]}
    )


@pytest.mark.parametrize(
    "body,expected",
    [
        ("/ci run", "/ci run"),
        ("  /ci retry\n", "/ci retry"),
        ("/ci cancel", "/ci cancel"),
        ("/ci run --allow-stale", "/ci run --allow-stale"),
        ("/ci run all", None),
        ("/ci run nightly", None),
        ("/ci run; id", None),
        ("```\n/ci run\n```", None),
        ("please /ci run", None),
        ("/ci run\nthanks", None),
    ],
)
def test_exact_commands(controller, body, expected):
    assert controller.parse_command(body) == expected


@pytest.mark.parametrize(
    "body,kind",
    [
        ("/ci run", "command"),
        ("  /ci retry\n", "command"),
        ("/ci cancel", "command"),
        ("/ci run --allow-stale", "command"),
        ("/ci run all", "help"),
        ("/ci", "help"),
        ("hello", "ignore"),
        ("```\n/ci run\n```", "ignore"),
        ("    /ci run", "ignore"),
        ("\t/ci run", "ignore"),
        (" \t/ci run", "ignore"),
        ("  \t/ci run", "ignore"),
        ("/ci run\nthanks", "ignore"),
        ("> /ci run", "ignore"),
    ],
)
def test_classification_is_exact_and_local(controller, api, body, kind):
    api.comment["body"] = body
    assert controller.classify(event(api)) == kind
    assert not api.checks and not api.dispatches and not api.messages


@pytest.mark.parametrize("change", ["bot", "issue", "edited"])
def test_classifier_ignores_non_command_sources(controller, api, change):
    ev = event(api)
    if change == "bot":
        ev["comment"]["user"]["type"] = "Bot"
    elif change == "issue":
        ev["issue"].pop("pull_request")
    else:
        ev["action"] = "edited"
    assert controller.classify(ev) == "ignore"


def test_help_does_not_authorize_or_touch_requests(controller, api, monkeypatch):
    api.comment["body"] = "/ci run all"
    monkeypatch.setattr(
        api, "review_decision", lambda _: pytest.fail("No authorization reads")
    )
    controller.help_command(event(api), api)
    assert "Unsupported" in api.messages[-1]["body"]
    assert api.pr_reads == 0 and not api.checks and not api.dispatches


@pytest.mark.parametrize("body", ["/ci run", "/ci run all"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"reactions": {"total_count": 1, "+1": 1}},
        {"user": {"avatar_url": "https://avatars.githubusercontent.com/new"}},
        {"user": {"site_admin": False, "gravatar_id": ""}},
        {"author_association": "CONTRIBUTOR"},
    ],
)
def test_comment_snapshot_ignores_irrelevant_metadata(
    controller, api, context, body, metadata
):
    api.comment["body"] = body
    ev = event(api)
    if "user" in metadata:
        api.comment["user"].update(metadata["user"])
    else:
        api.comment.update(metadata)
    if body == "/ci run":
        controller.handle_event(ev, api, context)
        assert len(api.dispatches) == 1 and len(api.checks) == 5
    else:
        controller.help_command(ev, api)
        assert len(api.messages) == 1 and "Unsupported" in api.messages[0]["body"]
        assert api.pr_reads == 0 and not api.checks and not api.dispatches


@pytest.mark.parametrize("body", ["/ci run", "/ci run all"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 999),
        ("issue_url", f"https://api.github.com/repos/{REPO}/issues/8"),
        ("issue_url", "https://api.github.com/repos/other/repo/issues/7"),
        ("body", "/ci cancel"),
        ("created_at", "2026-10-13T09:00:00Z"),
        ("updated_at", "2026-10-13T09:00:00Z"),
        ("user.id", 999),
        ("user.login", "maintainer"),
        ("user.type", "Bot"),
    ],
)
def test_comment_snapshot_rejects_edits_and_identity_changes(
    controller, api, context, monkeypatch, body, field, value
):
    api.comment["body"] = body
    ev = event(api)
    if field.startswith("user."):
        api.comment["user"][field.split(".")[1]] = value
    else:
        api.comment[field] = value
    # An edit restored to its original body still has a different updated_at.
    if field == "updated_at":
        api.comment["body"] = "/ci cancel"
        api.comment["body"] = ev["comment"]["body"]
    original_get = api.get

    def get_comment(path):
        if path == f"issues/comments/{ev['comment']['id']}":
            return copy.deepcopy(api.comment)
        return original_get(path)

    monkeypatch.setattr(api, "get", get_comment)
    if body == "/ci run":
        controller.handle_event(ev, api, context)
        assert "comment changed" in api.messages[-1]["body"]
    else:
        controller.help_command(ev, api)
        assert not api.messages
    assert api.pr_reads == 0 and not api.checks and not api.dispatches


@pytest.mark.parametrize("body", ["/ci run", "/ci run all"])
def test_comment_snapshot_must_belong_to_event_issue(controller, api, context, body):
    api.comment.update(
        body=body, issue_url=f"https://api.github.com/repos/{REPO}/issues/8"
    )
    if body == "/ci run":
        controller.handle_event(event(api), api, context)
        assert "Invalid command source" in api.messages[-1]["body"]
    else:
        controller.help_command(event(api), api)
        assert not api.messages
    assert api.pr_reads == 0 and not api.checks and not api.dispatches


@pytest.mark.parametrize(
    "permission,draft,labels,actor,trusted,allowed",
    [
        ("read", False, ["ready"], "author", "", True),
        ("read", False, ["ready-run-all-tests"], "author", "", True),
        ("read", False, [], "author", "", False),
        ("read", True, ["ready"], "author", "", False),
        ("read", False, ["ready"], "stranger", "", False),
        ("triage", False, [], "author", "", False),
        ("write", True, [], "maintainer", "", True),
        ("maintain", False, [], "maintainer", "", True),
        ("admin", False, [], "maintainer", "", True),
        ("read", True, [], "Maintainer", "maintainer", True),
    ],
)
def test_authorization_matrix(
    controller, api, permission, draft, labels, actor, trusted, allowed
):
    api.pr.update(draft=draft, labels=[{"name": name} for name in labels])
    api.permissions[actor] = permission
    if allowed:
        controller.authorize(api, api.pr, actor, trusted)
    else:
        with pytest.raises(controller.Rejected):
            controller.authorize(api, api.pr, actor, trusted)


@pytest.mark.parametrize(
    "decision,states,allowed",
    [
        ("APPROVED", ["APPROVED"], True),
        ("REVIEW_REQUIRED", ["APPROVED"], False),
        ("APPROVED", ["APPROVED", "DISMISSED"], False),
        ("APPROVED", ["APPROVED", "CHANGES_REQUESTED"], False),
        ("APPROVED", ["APPROVED", "COMMENTED"], True),
    ],
)
def test_current_trusted_approval(controller, api, decision, states, allowed):
    api.pr["labels"] = []
    api.decision = decision
    api.reviews = [
        {"user": {"login": "maintainer"}, "state": state} for state in states
    ]
    if allowed:
        controller.authorize(api, api.pr, "author", "")
    else:
        with pytest.raises(controller.Rejected):
            controller.authorize(api, api.pr, "author", "")


def test_command_creates_bound_checks_before_dispatch(controller, api, context):
    request = launch(controller, api, context)
    record = json.loads(request["output"]["summary"])
    assert record["head_sha"] == HEAD and record["control_sha"] == CONTROL
    assert record["base_sha"] == BASE and record["comment_id"] == 17
    assert record["controller_run_id"] == 100 and record["pr_number"] == 7
    assert len(api.checks) == 5
    assert all(
        c["head_sha"] == HEAD and c["status"] == "queued" for c in api.checks.values()
    )
    assert api.dispatches == [
        {
            "ref": "main",
            "inputs": {
                "pr_number": "7",
                "head_sha": HEAD,
                "base_sha": BASE,
                "request_check_id": str(request["id"]),
            },
        }
    ]
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 1


@pytest.mark.parametrize(
    "mutation",
    ["closed", "unauthorized", "edited", "issue", "stale", "race", "unknown"],
)
def test_rejected_event_never_dispatches(controller, api, context, mutation):
    ev = event(api)
    if mutation == "closed":
        api.pr["state"] = "closed"
    elif mutation == "unauthorized":
        api.pr["labels"] = []
    elif mutation == "edited":
        ev["action"] = "edited"
    elif mutation == "issue":
        ev["issue"].pop("pull_request")
    elif mutation == "stale":
        api.behind = 3
    elif mutation == "race":
        api.change_on_read = 2
    elif mutation == "unknown":
        ev["comment"]["body"] = "/ci run all"
    controller.handle_event(ev, api, context)
    assert not api.dispatches and not api.checks


@pytest.mark.parametrize("side", ["head", "base"])
def test_stable_pr_ignores_repository_metadata(controller, api, context, side):
    original = copy.deepcopy(api.pr)
    api.pr[side]["repo"]["pushed_at"] = "2026-10-13T08:00:00Z"
    assert controller.stable_pr(api, original, "author", context) == api.pr


@pytest.mark.parametrize("side", ["head", "base"])
@pytest.mark.parametrize("field", ["sha", "ref", "repository", "repository_id"])
def test_stable_pr_rejects_identity_changes(controller, api, context, side, field):
    api.pr[side]["repo"]["id"] = 100
    original = copy.deepcopy(api.pr)
    if field == "repository":
        api.pr[side]["repo"]["full_name"] = "other/repo"
    elif field == "repository_id":
        api.pr[side]["repo"]["id"] = 101
    else:
        api.pr[side][field] = "d" * 40 if field == "sha" else "other-branch"
    with pytest.raises(controller.Rejected, match="head or base changed"):
        controller.stable_pr(api, original, "author", context)


def test_stable_pr_rechecks_authorization(controller, api, context):
    original = copy.deepcopy(api.pr)
    api.pr["labels"] = []
    with pytest.raises(controller.Rejected):
        controller.stable_pr(api, original, "author", context)


def test_allow_stale_runs_but_warns(controller, api, context):
    api.behind = 2
    api.comment["body"] = "/ci run --allow-stale"
    launch(controller, api, context)
    assert api.dispatches
    assert "behind" in api.messages[-1]["body"]


def test_api_failure_is_not_an_authorization(controller, api, context, monkeypatch):
    monkeypatch.setattr(api, "get", lambda _: (_ for _ in ()).throw(OSError("403")))
    with pytest.raises(OSError):
        controller.handle_event(event(api), api, context)
    assert not api.dispatches


def test_lost_dispatch_response_keeps_request_pending(controller, api, context):
    api.fail_dispatch = True
    controller.handle_event(event(api), api, context)
    assert len(api.checks) == 5
    assert all(c["status"] == "queued" for c in api.checks.values())
    assert "confirm" in api.messages[-1]["body"].lower()


def test_executor_validates_record_and_uses_pr_sha(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    inputs = api.dispatches[0]["inputs"]
    record = controller.validate_execution(
        api, inputs, run, {**context, "run_id": run["id"]}
    )
    assert record["head_sha"] == HEAD != run["head_sha"]


@pytest.mark.parametrize(
    "mutation",
    [
        "app",
        "controller",
        "repository",
        "head",
        "base",
        "control",
        "branch",
        "label",
        "input",
        "replay",
    ],
)
def test_executor_fails_closed_before_ppu(controller, api, context, mutation):
    request = launch(controller, api, context)
    run = executor(api, request)
    inputs = copy.deepcopy(api.dispatches[0]["inputs"])
    if mutation == "app":
        api.checks[request["id"]]["app"]["id"] = 999
    elif mutation == "controller":
        api.runs[100]["path"] = ".github/workflows/ut.yaml"
    elif mutation == "repository":
        run["repository"]["full_name"] = "evil/repo"
    elif mutation == "head":
        api.pr["head"]["sha"] = "e" * 40
    elif mutation == "base":
        api.pr["base"]["sha"] = "e" * 40
    elif mutation == "control":
        run["head_sha"] = "e" * 40
    elif mutation == "branch":
        run["head_branch"] = "untrusted"
    elif mutation == "label":
        api.pr["labels"] = []
    elif mutation == "input":
        inputs["head_sha"] = "e" * 40
    elif mutation == "replay":
        api.runs[199] = {**copy.deepcopy(run), "id": 199}
    with pytest.raises(controller.Rejected):
        controller.validate_execution(
            api, inputs, run, {**context, "run_id": run["id"]}
        )


def test_ready_notification_is_idempotent_and_never_dispatches(
    controller, api, context
):
    ev = {"action": "labeled", "label": {"name": "ready"}, "pull_request": api.pr}
    controller.notify(ev, api, context)
    controller.notify(ev, api, context)
    assert len(api.messages) == 1 and "/ci run" in api.messages[0]["body"]
    assert not api.dispatches and not api.checks


def test_retry_requires_an_execution_for_current_head(controller, api, context):
    api.comment["body"] = "/ci retry"
    controller.handle_event(event(api), api, context)
    assert not api.retries and not api.dispatches
    assert "/ci run" in api.messages[-1]["body"]


def test_retry_and_cancel_only_touch_bound_execution(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    run.update(status="completed", conclusion="failure")
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    assert not api.retries and len(api.dispatches) == 2
    retry = latest_request(api)
    assert retry["id"] != request["id"]
    executor(api, retry, 201)
    api.comment.update(id=19, body="/ci cancel")
    controller.handle_event(event(api), api, context)
    assert api.cancellations == [201]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422, 429])
def test_rejected_retry_write_can_be_retried(
    controller, api, context, monkeypatch, code
):
    from urllib.error import HTTPError

    request = launch(controller, api, context)
    run = executor(api, request)
    run.update(status="completed", conclusion="failure")
    original_post = api.post

    before = copy.deepcopy(api.checks)

    def reject(path, payload=None):
        if path.endswith("/dispatches"):
            raise HTTPError("https://api.github.com", code, "rejected", {}, None)
        return original_post(path, payload)

    monkeypatch.setattr(api, "post", reject)
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    retry = latest_request(api)
    record = json.loads(retry["output"]["summary"])
    assert record["dispatch_state"] == "rejected"
    assert {k: api.checks[k] for k in before} == before
    assert all(
        c["status"] == "completed" and c["conclusion"] == "failure"
        for k, c in api.checks.items()
        if k not in before
    )
    assert "not accepted" in api.messages[-1]["body"]
    monkeypatch.setattr(api, "post", original_post)
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    assert not api.retries and len(api.dispatches) == 2
    assert (
        json.loads(latest_request(api)["output"]["summary"])["retry_of_request_id"]
        == request["id"]
    )


@pytest.mark.parametrize("failure_step", ["record", "check", "request"])
def test_retry_preparation_failure_does_not_wedge_request(
    controller, api, context, monkeypatch, failure_step
):
    request = launch(controller, api, context)
    executor(api, request).update(status="completed", conclusion="failure")
    original_patch, original_post = api.patch, api.post
    before = copy.deepcopy(api.checks)
    failed = []

    def fail_patch(path, payload):
        result = original_patch(path, payload)
        if failure_step == "record" and not failed:
            failed.append(path)
            raise OSError("record response lost before dispatch")
        return result

    def fail_post(path, payload=None):
        result = original_post(path, payload)
        if (
            failure_step in {"check", "request"}
            and path == "check-runs"
            and (payload["name"] == "PPU CI request") == (failure_step == "request")
            and not failed
        ):
            failed.append(path)
            raise OSError("check response lost before dispatch")
        return result

    monkeypatch.setattr(api, "patch", fail_patch)
    monkeypatch.setattr(api, "post", fail_post)
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    assert failed and not api.retries and len(api.dispatches) == 1
    assert (
        json.loads(latest_request(api)["output"]["summary"])["dispatch_state"]
        == "rejected"
    )
    assert {k: api.checks[k] for k in before} == before
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 2 and not api.retries


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("error_kind", ["connection", "http408", "http503"])
def test_uncertain_retry_write_reconciles_without_duplicate(
    controller, api, context, monkeypatch, accepted, error_kind
):
    from urllib.error import HTTPError

    request = launch(controller, api, context)
    run = executor(api, request)
    run.update(status="completed", conclusion="failure")
    original_post = api.post
    writes = []

    def lose_response(path, payload=None):
        if path.endswith("/dispatches"):
            writes.append(path)
            if accepted:
                original_post(path, payload)
                executor(api, latest_request(api), 201)
            if error_kind.startswith("http"):
                raise HTTPError(
                    "https://api.github.com", int(error_kind[4:]), "uncertain", {}, None
                )
            raise OSError("dispatch response lost")
        return original_post(path, payload)

    monkeypatch.setattr(api, "post", lose_response)
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    record = json.loads(latest_request(api)["output"]["summary"])
    assert record["dispatch_state"] == ("accepted" if accepted else "uncertain")
    assert ("confirmed" if accepted else "uncertain") in api.messages[-1]["body"]
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    assert len(writes) == 1 and not api.retries
    if not accepted:
        assert "inspect" in api.messages[-1]["body"]
    monkeypatch.setattr(api, "post", original_post)
    if accepted:
        api.runs[201].update(status="completed", conclusion="failure")
        next_comment(api, context)
        controller.handle_event(event(api), api, context)
        assert len(api.dispatches) == 3
    else:
        next_comment(api, context, "/ci cancel")
        controller.handle_event(event(api), api, context)
        next_comment(api, context, "/ci run")
        controller.handle_event(event(api), api, context)
        assert len(api.dispatches) == 2


def test_http_transport_accepts_compare_and_never_retries_writes(monkeypatch):
    import io

    from scripts.ci.ppu_ci_common import GitHub

    urls = []

    def respond(request, timeout):
        urls.append(request.full_url)
        return io.BytesIO(b'{"behind_by":0}')

    monkeypatch.setattr("urllib.request.urlopen", respond)
    client = GitHub(REPO, "test-token")
    assert client.get(f"compare/main...{HEAD}") == {"behind_by": 0}
    assert urls == [f"https://api.github.com/repos/{REPO}/compare/main...{HEAD}"]
    calls = []

    def unavailable(request, timeout):
        calls.append(request)
        raise OSError("response lost")

    monkeypatch.setattr("urllib.request.urlopen", unavailable)
    with pytest.raises(OSError):
        client.post("actions/workflows/ppu-pr-ci.yaml/dispatches", {})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "path", ["/evil", "../evil", "foo/../evil", "https://evil/a", "foo/%2e%2e/evil"]
)
def test_http_transport_rejects_unsafe_paths(path, monkeypatch):
    from scripts.ci.ppu_ci_common import GitHub, Rejected

    def unexpected(*args, **kwargs):
        pytest.fail("Unsafe API path reached the transport")

    monkeypatch.setattr("urllib.request.urlopen", unexpected)
    with pytest.raises(Rejected):
        GitHub(REPO, "test-token").get(path)


def test_active_automatic_run_is_deduplicated(controller, api, context):
    api.runs[50] = {
        "id": 50,
        "event": "pull_request",
        "status": "in_progress",
        "path": ".github/workflows/build-ppu-wheels.yaml",
        "repository": {"full_name": REPO},
        "pull_requests": [{"number": 7, "head": {"sha": HEAD}}],
    }
    controller.handle_event(event(api), api, context)
    assert not api.dispatches and not api.checks
    assert "/runs/50" in api.messages[-1]["body"]


@pytest.mark.parametrize("target", [100, 200])
def test_workflow_id_is_verified(controller, api, context, target):
    request = launch(controller, api, context)
    run = executor(api, request)
    api.runs[target]["workflow_id"] = 999
    with pytest.raises(controller.Rejected):
        controller.validate_execution(
            api, api.dispatches[0]["inputs"], run, {**context, "run_id": run["id"]}
        )


def test_cancel_after_push_only_cancels_this_pr(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    api.runs[201] = {
        **copy.deepcopy(run),
        "id": 201,
        "display_title": run["display_title"].replace("PR 7", "PR 8"),
    }
    api.pr["head"]["sha"] = "d" * 40
    api.comment.update(id=18, body="/ci cancel")
    controller.handle_event(event(api), api, context)
    assert api.cancellations == [200]


def test_cancel_queued_dispatch_blocks_late_execution(controller, api, context):
    request = launch(controller, api, context)
    api.comment.update(id=18, body="/ci cancel")
    controller.handle_event(event(api), api, context)
    run = executor(api, request)
    with pytest.raises(controller.Rejected):
        controller.validate_execution(
            api, api.dispatches[0]["inputs"], run, {**context, "run_id": run["id"]}
        )


@pytest.mark.parametrize("dispatch_state", ["accepted", "uncertain", "preparing"])
def test_cancel_old_head_without_executor(controller, api, context, dispatch_state):
    api.fail_dispatch = dispatch_state == "uncertain"
    request = launch(controller, api, context)
    record = json.loads(request["output"]["summary"])
    record["dispatch_state"] = dispatch_state
    controller.save_record(api, request, record)
    inputs = {
        "pr_number": "7",
        "head_sha": HEAD,
        "base_sha": BASE,
        "request_check_id": str(request["id"]),
    }
    before_dispatches = copy.deepcopy(api.dispatches)
    api.pr["head"]["sha"] = "d" * 40
    next_comment(api, context, "/ci cancel")
    controller.handle_event(event(api), api, context)
    assert json.loads(api.checks[request["id"]]["output"]["summary"])["cancelled"]
    assert all(c["conclusion"] == "cancelled" for c in api.checks.values())
    assert not api.cancellations and api.dispatches == before_dispatches
    api.pr["head"]["sha"] = HEAD
    run = executor(api, request)
    with pytest.raises(controller.Rejected, match="cancelled"):
        controller.validate_execution(
            api, inputs, run, {**context, "run_id": run["id"]}
        )


def test_request_receipt_indexes_head_before_any_check(
    controller, api, context, monkeypatch
):
    original = api.post

    def check_receipt(path, payload=None):
        if path == "check-runs":
            assert any(
                f"<!-- ppu-ci-head:{HEAD} -->" in c["body"] for c in api.messages
            )
        return original(path, payload)

    monkeypatch.setattr(api, "post", check_receipt)
    launch(controller, api, context)


def test_cancel_discovers_old_receipts_across_pages(
    controller, api, context, monkeypatch
):
    from scripts.ci.ppu_ci_common import GitHub

    launch(controller, api, context)
    api.messages[:0] = [
        {"user": {"login": "author"}, "body": "ordinary comment"} for _ in range(100)
    ]
    original_get, original_pages = api.get, api.pages
    fetched = []

    def get_page(path):
        if path.startswith("issues/7/comments?"):
            page = int(path.rsplit("=", 1)[1])
            fetched.append(page)
            return copy.deepcopy(api.messages[(page - 1) * 100 : page * 100])
        return original_get(path)

    def pages(path, key=None):
        if path == "issues/7/comments":
            return GitHub.pages(api, path, key)
        return original_pages(path, key)

    monkeypatch.setattr(api, "get", get_page)
    monkeypatch.setattr(api, "pages", pages)
    api.pr["head"]["sha"] = "d" * 40
    next_comment(api, context, "/ci cancel")
    controller.handle_event(event(api), api, context)
    assert 2 in fetched
    assert all(c["conclusion"] == "cancelled" for c in api.checks.values())


@pytest.mark.parametrize("mutation", ["user", "quoted", "invalid_sha"])
def test_receipt_hint_is_not_accepted_from_arbitrary_comment(
    controller, api, context, mutation
):
    launch(controller, api, context)
    receipt = copy.deepcopy(api.messages[0])
    if mutation == "user":
        receipt["user"]["login"] = "author"
    elif mutation == "quoted":
        receipt["body"] = "> " + receipt["body"]
    else:
        receipt["body"] = receipt["body"].replace(HEAD, "../../invalid")
    api.messages = [receipt]
    api.pr["head"]["sha"] = "d" * 40
    before = copy.deepcopy(api.checks)
    next_comment(api, context, "/ci cancel")
    controller.handle_event(event(api), api, context)
    assert api.checks == before and not api.cancellations


def test_receipt_lookup_preserves_other_pr_and_completed_checks(
    controller, api, context
):
    request = launch(controller, api, context)
    for check in api.checks.values():
        check.update(status="completed", conclusion="success")
    completed = copy.deepcopy(api.checks)
    api.pr["head"]["sha"] = "d" * 40
    next_comment(api, context, "/ci run")
    controller.handle_event(event(api), api, context)
    pending = latest_request(api)
    other = copy.deepcopy(pending)
    other["id"] = 999
    record = json.loads(other["output"]["summary"])
    record["pr_number"] = 8
    other["output"]["summary"] = json.dumps(record)
    api.checks[999] = copy.deepcopy(other)
    api.pr["head"]["sha"] = "e" * 40
    next_comment(api, context, "/ci cancel")
    controller.handle_event(event(api), api, context)
    assert {key: api.checks[key] for key in completed} == completed
    assert api.checks[999] == other
    assert api.checks[pending["id"]]["conclusion"] == "cancelled"
    assert api.checks[request["id"]]["conclusion"] == "success"


@pytest.mark.parametrize("dispatch_state", ["accepted", "uncertain"])
@pytest.mark.parametrize("run_state", [None, "in_progress", "completed"])
@pytest.mark.parametrize("older_head", [False, True])
@pytest.mark.parametrize("write_index", range(5))
@pytest.mark.parametrize("applied", [False, True])
def test_new_cancel_repairs_each_failed_check_write(
    controller,
    api,
    context,
    monkeypatch,
    dispatch_state,
    run_state,
    older_head,
    write_index,
    applied,
):
    completed_request = launch(controller, api, context)
    executor(api, completed_request, 199).update(
        status="completed", conclusion="success"
    )
    for check in api.checks.values():
        check.update(status="completed", conclusion="success")
    completed = copy.deepcopy(api.checks)
    next_comment(api, context, "/ci run")
    api.fail_dispatch = dispatch_state == "uncertain"
    controller.handle_event(event(api), api, context)
    request = latest_request(api)
    assert json.loads(request["output"]["summary"])["dispatch_state"] == dispatch_state
    target_ids = [key for key in api.checks if key not in completed]
    if run_state:
        executor(api, request).update(status=run_state)
    if older_head:
        api.pr["head"]["sha"] = "d" * 40
    original_patch, original_post = api.patch, api.post
    failed, writes = [], []

    def fail_once(path, payload):
        writes.append(path)
        if path == f"check-runs/{target_ids[write_index]}" and not failed:
            failed.append(path)
            if applied:
                original_patch(path, payload)
            raise OSError("cancel check write failed")
        return original_patch(path, payload)

    def cancel_only_after_invalidation(path, payload=None):
        if path.endswith("/cancel"):
            assert json.loads(api.checks[request["id"]]["output"]["summary"])[
                "cancelled"
            ]
            assert all(
                api.checks[key]["status"] == "completed"
                and api.checks[key]["conclusion"] == "cancelled"
                for key in target_ids
            )
        return original_post(path, payload)

    monkeypatch.setattr(api, "patch", fail_once)
    monkeypatch.setattr(api, "post", cancel_only_after_invalidation)
    next_comment(api, context, "/ci cancel")
    ev = event(api)
    with pytest.raises(OSError, match="cancel check write failed"):
        controller.handle_event(ev, api, context)
    assert failed and not api.cancellations
    assert {key: api.checks[key] for key in completed} == completed
    partial, write_count = copy.deepcopy(api.checks), len(writes)
    controller.handle_event(ev, api, context)
    assert "already been processed" in api.messages[-1]["body"]
    assert api.checks == partial and len(writes) == write_count
    assert not api.cancellations

    next_comment(api, context, "/ci cancel")
    controller.handle_event(event(api), api, context)
    assert all(
        api.checks[key]["status"] == "completed"
        and api.checks[key].get("conclusion") == "cancelled"
        for key in target_ids
    )
    assert json.loads(api.checks[request["id"]]["output"]["summary"])["cancelled"]
    assert {key: api.checks[key] for key in completed} == completed
    assert api.cancellations == ([200] if run_state == "in_progress" else [])


@pytest.mark.parametrize("applied", [False, True])
def test_partial_cancel_is_not_completed_by_handled_comment(
    controller, api, context, monkeypatch, applied
):
    request = launch(controller, api, context)
    executor(api, request)
    next_comment(api, context, "/ci cancel")
    original_post = api.post
    failed = []

    def fail_cancel_once(path, payload=None):
        if path.endswith("/cancel"):
            saved = api.checks[request["id"]]
            assert json.loads(saved["output"]["summary"])["cancelled"]
            assert saved["status"] == "completed"
            assert saved["conclusion"] == "cancelled"
            if not failed:
                failed.append(path)
                if applied:
                    original_post(path, payload)
                raise OSError("Actions cancel response lost")
        return original_post(path, payload)

    monkeypatch.setattr(api, "post", fail_cancel_once)
    with pytest.raises(OSError, match="Actions cancel response lost"):
        controller.cancel_pr(api, api.pr, api.comment, context)
    # Only the event receipt prevents replay; a partial record is not success.
    controller.cancel_pr(api, api.pr, api.comment, context)
    assert api.cancellations == [200] * (2 if applied else 1)
    assert "Cancellation requested" in api.messages[-1]["body"]


@pytest.mark.parametrize("mutation", ["head", "authorization", "provenance"])
def test_old_receipt_does_not_bypass_final_cancel_validation(
    controller, api, context, mutation
):
    request = launch(controller, api, context)
    api.pr["head"]["sha"] = "d" * 40
    if mutation == "head":
        api.change_on_read = api.pr_reads + 2
        api.pr["head"]["sha"] = "e" * 40
    elif mutation == "authorization":
        api.pr["labels"] = []
    else:
        api.runs[100]["event"] = "pull_request"
    before = copy.deepcopy(api.checks)
    next_comment(api, context, "/ci cancel")
    controller.handle_event(event(api), api, context)
    assert api.checks == before and not api.cancellations
    assert not json.loads(api.checks[request["id"]]["output"]["summary"]).get(
        "cancelled"
    )


@pytest.mark.parametrize("lost_dispatch", [False, True])
def test_retry_ignores_older_cancelled_request_without_run(
    controller, api, context, lost_dispatch
):
    api.fail_dispatch = lost_dispatch
    old = launch(controller, api, context)
    next_comment(api, context, "/ci cancel")
    controller.handle_event(event(api), api, context)
    api.fail_dispatch = False
    next_comment(api, context, "/ci run")
    controller.handle_event(event(api), api, context)
    source = latest_request(api)
    executor(api, source).update(status="completed", conclusion="failure")
    before = copy.deepcopy(api.checks)
    dispatched = len(api.dispatches)
    next_comment(api, context, "/ci retry")
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == dispatched + 1
    retry = json.loads(latest_request(api)["output"]["summary"])
    assert retry["retry_of_request_id"] == source["id"] != old["id"]
    assert {k: api.checks[k] for k in before} == before


def test_latest_cancelled_no_run_does_not_revive_older_execution(
    controller, api, context
):
    old = launch(controller, api, context)
    executor(api, old).update(status="completed", conclusion="failure")
    for check in api.checks.values():
        check.update(status="completed", conclusion="failure")
    next_comment(api, context, "/ci run")
    controller.handle_event(event(api), api, context)
    next_comment(api, context, "/ci cancel")
    controller.handle_event(event(api), api, context)
    before = copy.deepcopy(api.checks)
    dispatched = len(api.dispatches)
    next_comment(api, context, "/ci retry")
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == dispatched and api.checks == before


def test_retry_respects_revoked_authorization(controller, api, context):
    request = launch(controller, api, context)
    executor(api, request).update(status="completed", conclusion="failure")
    api.pr["labels"] = []
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    assert not api.retries


@pytest.mark.parametrize("revoke_retry_actor", [False, True])
def test_retry_validator_uses_current_requester(
    controller, api, context, revoke_retry_actor
):
    request = launch(controller, api, context)
    run = executor(api, request)
    api.pr["labels"] = []
    with pytest.raises(controller.Rejected):
        controller.validate_execution(
            api, api.dispatches[0]["inputs"], run, {**context, "run_id": run["id"]}
        )
    run.update(status="completed", conclusion="failure")
    api.comment.update(
        id=18, body="/ci retry", user={"id": 2, "login": "maintainer", "type": "User"}
    )
    controller.handle_event(event(api), api, context)
    assert not api.retries and len(api.dispatches) == 2
    run = executor(api, latest_request(api), 201)
    if revoke_retry_actor:
        api.permissions["maintainer"] = "read"
        with pytest.raises(controller.Rejected):
            validate(controller, api, run, context)
    else:
        record = validate(controller, api, run, context)
        assert record["validation"]["run_id"] == 201
        assert record["actor"] == "maintainer"


def test_duplicate_delivery_after_push_cannot_test_new_head(controller, api, context):
    launch(controller, api, context)
    api.pr["head"]["sha"] = "d" * 40
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 1


def test_duplicate_retry_comment_cannot_retry_again_after_new_attempt(
    controller, api, context
):
    request = launch(controller, api, context)
    run = executor(api, request)
    run.update(status="completed", conclusion="failure")
    api.comment.update(id=18, body="/ci retry")
    controller.handle_event(event(api), api, context)
    run["run_attempt"] = 2
    controller.handle_event(event(api), api, context)
    assert not api.retries
    assert len(api.dispatches) == 2


def test_retry_creates_new_request_without_mutating_source(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    validate(controller, api, run, context)
    run.update(status="completed", conclusion="failure")
    before = copy.deepcopy(api.checks)
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    retry = latest_request(api)
    record = json.loads(retry["output"]["summary"])
    assert retry["id"] != request["id"]
    assert not api.retries and len(api.dispatches) == 2
    assert record["schema"] == 2
    assert record["retry_of_request_id"] == request["id"]
    assert record["retry_of_run_id"] == run["id"]
    assert {k: api.checks[k] for k in before} == before
    assert set(api.dispatches[-1]["inputs"]) == {
        "pr_number",
        "head_sha",
        "base_sha",
        "request_check_id",
    }
    new_run = executor(api, retry, 201)
    result = validate(controller, api, new_run, context)
    assert result["validation"]["run_id"] == 201
    assert result["validation"]["run_attempt"] == 1
    assert result["validation"]["plan"] == {
        stage: {"mode": "run"} for stage in ("build", "pin", "smoke")
    }


@pytest.mark.parametrize("changed", ["head", "base", "control"])
def test_retry_rejects_changed_revision(controller, api, context, changed):
    request = launch(controller, api, context)
    run = executor(api, request)
    validate(controller, api, run, context)
    run.update(status="completed", conclusion="failure")
    next_comment(api, context)
    if changed == "control":
        context["control_sha"] = "d" * 40
        api.runs[context["run_id"]]["head_sha"] = "d" * 40
    else:
        api.pr[changed]["sha"] = "d" * 40
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 1 and not api.retries
    assert "/ci run" in api.messages[-1]["body"]


@pytest.mark.parametrize("changed", ["label", "draft", "closed", "permission", "base"])
def test_new_retry_validate_rechecks_waiting_request(controller, api, context, changed):
    request = launch(controller, api, context)
    run = executor(api, request)
    validate(controller, api, run, context)
    run.update(status="completed", conclusion="failure")
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 2
    run = executor(api, latest_request(api), 201)
    if changed == "label":
        api.pr["labels"] = []
    elif changed == "draft":
        api.pr["draft"] = True
    elif changed == "closed":
        api.pr["state"] = "closed"
    elif changed == "permission":
        api.pr["labels"] = []
        api.permissions["author"] = "none"
    else:
        api.pr["base"]["sha"] = "d" * 40
    with pytest.raises(controller.Rejected):
        validate(controller, api, run, context)


def test_native_rerun_cannot_validate_again(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    validate(controller, api, run, context)
    run["run_attempt"] = 2
    with pytest.raises(controller.Rejected, match="/ci retry"):
        validate(controller, api, run, context)


def complete_jobs(api, run, outcomes=("success", "success", "failure")):
    from scripts.ci.ppu_ci_common import CONTEXTS

    api.job_runs[run["id"]] = [
        {
            "id": run["id"] * 10 + i,
            "run_id": run["id"],
            "run_attempt": 1,
            "name": name,
            "status": "completed",
            "conclusion": outcome,
        }
        for i, (name, outcome) in enumerate(
            zip(
                [*("PPU / " + n for n in CONTEXTS), "validate"],
                [*outcomes, "failure", "success"],
                strict=False,
            )
        )
    ]
    artifact_id = run["id"] * 100
    api.artifacts[artifact_id] = {
        "id": artifact_id,
        "name": f"ppu-wheels-{run['id']}",
        "digest": "sha256:" + "e" * 64,
        "expired": False,
        "expires_at": "2099-01-01T00:00:00Z",
        "workflow_run": {"id": run["id"], "head_sha": CONTROL},
    }
    run.update(status="completed", conclusion="failure")


@pytest.mark.parametrize("build", ["success", "failure", "skipped", "missing"])
@pytest.mark.parametrize("pin", ["success", "failure"])
@pytest.mark.parametrize("smoke", ["success", "failure"])
def test_retry_plan_uses_real_leaf_jobs(controller, api, context, build, pin, smoke):
    request = launch(controller, api, context)
    run = executor(api, request)
    validate(controller, api, run, context)
    complete_jobs(api, run, (build, pin, smoke))
    if build == "missing":
        api.job_runs[200].pop(0)
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 2
    retry = latest_request(api)
    record = validate(controller, api, executor(api, retry, 201), context)
    plan = record["validation"]["plan"]
    for stage, outcome, job_id in zip(
        ("build", "pin", "smoke"), (build, pin, smoke), (2000, 2001, 2002), strict=False
    ):
        reuse = build == "success" and outcome == "success"
        assert plan[stage]["mode"] == ("reuse" if reuse else "run")
        if reuse:
            assert plan[stage]["source"]["request_id"] == request["id"]
            assert plan[stage]["source"]["run_id"] == 200
            assert plan[stage]["source"]["run_attempt"] == 1
            assert plan[stage]["source"]["job_id"] == job_id
    if build == "success":
        assert plan["build"]["source"]["artifact"] == {
            "id": 20000,
            "run_id": 200,
            "digest": "sha256:" + "e" * 64,
        }


@pytest.mark.parametrize(
    "damage",
    ["expired", "deleted", "digest", "run", "duplicate", "job", "attempt", "schema"],
)
def test_retry_rejects_untrustworthy_evidence(controller, api, context, damage):
    request = launch(controller, api, context)
    run = executor(api, request)
    validate(controller, api, run, context)
    complete_jobs(api, run)
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 2
    retry_run = executor(api, latest_request(api), 201)
    if damage == "expired":
        api.artifacts[20000]["expired"] = True
    elif damage == "deleted":
        api.artifacts.clear()
    elif damage == "digest":
        api.artifacts[20000]["digest"] = "sha256:" + "f" * 64
    elif damage == "run":
        api.artifacts[20000]["workflow_run"]["id"] = 999
    elif damage == "duplicate":
        api.artifacts[20001] = {**api.artifacts[20000], "id": 20001}
    elif damage == "job":
        api.job_runs[200].append(copy.deepcopy(api.job_runs[200][0]))
    elif damage == "attempt":
        run["run_attempt"] = 2
    else:
        record = json.loads(api.checks[request["id"]]["output"]["summary"])
        record["schema"] = 999
        api.checks[request["id"]]["output"]["summary"] = json.dumps(record)
    with pytest.raises(controller.Rejected):
        validate(controller, api, retry_run, context)


@pytest.mark.parametrize("phase", ["retry", "validate"])
def test_active_automatic_run_blocks_retry_and_start(controller, api, context, phase):
    request = launch(controller, api, context)
    run = executor(api, request)
    validate(controller, api, run, context)
    complete_jobs(api, run)
    next_comment(api, context)
    if phase == "validate":
        controller.handle_event(event(api), api, context)
        retry_run = executor(api, latest_request(api), 201)
    api.runs[300] = {
        "id": 300,
        "event": "pull_request",
        "status": "in_progress",
        "path": ".github/workflows/build-ppu-wheels.yaml",
        "repository": {"full_name": REPO},
        "pull_requests": [{"number": 7, "head": {"sha": HEAD}}],
    }
    if phase == "retry":
        controller.handle_event(event(api), api, context)
        assert len(api.dispatches) == 1
    else:
        with pytest.raises(controller.Rejected, match="utomatic"):
            validate(controller, api, retry_run, context)


def test_multi_retry_keeps_original_leaf_sources(controller, api, context):
    request = launch(controller, api, context)
    run = executor(api, request)
    validate(controller, api, run, context)
    complete_jobs(api, run)
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    retry = latest_request(api)
    second = executor(api, retry, 201)
    second_record = validate(controller, api, second, context)
    complete_jobs(api, second, ("skipped", "skipped", "success"))
    next_comment(api, context)
    controller.handle_event(event(api), api, context)
    assert len(api.dispatches) == 3
    third = executor(api, latest_request(api), 202)
    plan = validate(controller, api, third, context)["validation"]["plan"]
    assert plan["build"] == second_record["validation"]["plan"]["build"]
    assert plan["pin"]["source"]["run_id"] == 200
    assert plan["smoke"]["source"]["run_id"] == 201
    assert all(value["mode"] == "reuse" for value in plan.values())
