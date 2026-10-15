# SPDX-License-Identifier: Apache-2.0
"""Cursor artifacts are scheduling hints, never execution authorization."""

import copy
import importlib
import io
import json
import stat
import struct
import urllib.request
import zipfile
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

REPO = "t-head/vllm-sail"
SHA = "a" * 40
REPORTER = "ppu-ci-report.yml"
EXECUTOR = "ppu-pr-ci.yaml"
LIST = "actions/artifacts?name=ppu-report-cursor-v1&per_page=100&page=1"


class API:
    repo = REPO
    token = "private-test-token"

    def __init__(self):
        self.reads = []
        self.consumed = 0
        self.limit = 100
        self.responses = {
            "": {"id": 10, "full_name": REPO, "default_branch": "main"},
            f"actions/workflows/{REPORTER}": {
                "id": 20,
                "path": f".github/workflows/{REPORTER}",
            },
            f"actions/workflows/{EXECUTOR}": {
                "id": 30,
                "path": f".github/workflows/{EXECUTOR}",
            },
            "actions/runs/400": {
                "id": 400,
                "run_attempt": 1,
                "workflow_id": 20,
                "path": f".github/workflows/{REPORTER}",
                "repository": {"id": 10, "full_name": REPO},
                "head_repository": {"id": 10, "full_name": REPO},
                "head_branch": "main",
                "head_sha": SHA,
                "event": "schedule",
                "status": "completed",
                "conclusion": "success",
            },
        }

    def consume(self):
        if self.consumed >= self.limit:
            raise RuntimeError("budget exhausted")
        self.consumed += 1

    def get(self, path):
        self.consume()
        self.reads.append(path)
        value = self.responses[path]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)

    def post(self, *args, **kwargs):
        pytest.fail("Checkpoint must never POST")

    def patch(self, *args, **kwargs):
        pytest.fail("Checkpoint must never PATCH")


@pytest.fixture
def checkpoint():
    class Checkpoint:
        def __getattr__(self, name):
            path = (
                Path(__file__).resolve().parents[2] / "scripts/ci/ppu_ci_checkpoint.py"
            )
            assert path.exists(), "Trusted checkpoint module is missing"
            return getattr(
                importlib.import_module("scripts.ci.ppu_ci_checkpoint"), name
            )

    return Checkpoint()


@pytest.fixture
def api():
    return API()


def payload():
    return {
        "schema": 1,
        "repository": REPO,
        "executor_workflow_id": 30,
        "producer": {"run_id": 400, "attempt": 1, "control_sha": SHA},
        "before_run_id": 200,
    }


@pytest.mark.parametrize(
    "cursor", [{}, {"before_run_id": None}, {"before_run_id": 200}]
)
def test_encode_binds_live_producer_and_executor(checkpoint, api, cursor):
    encoded = checkpoint.encode(api, cursor, 400, 1, SHA)
    expected = payload()
    expected["before_run_id"] = cursor.get("before_run_id")
    assert json.loads(encoded) == expected
    assert encoded == json.dumps(expected, separators=(",", ":"))
    assert len(encoded.encode()) <= 8192
    assert set(api.reads) == {
        "",
        f"actions/workflows/{REPORTER}",
        f"actions/workflows/{EXECUTOR}",
        "actions/runs/400",
    }
    assert api.consumed == len(api.reads)
    assert checkpoint.NAME == "ppu-report-cursor-v1"
    assert checkpoint.FILENAME == "cursor.json"


@pytest.mark.parametrize("event", ["schedule", "workflow_dispatch"])
def test_encode_accepts_trusted_triggers_and_current_rerun(checkpoint, api, event):
    api.responses["actions/runs/400"].update(event=event, run_attempt=2)
    assert (
        json.loads(checkpoint.encode(api, {}, 400, 2, SHA))["producer"]["attempt"] == 2
    )


@pytest.mark.parametrize(
    "cursor",
    [None, [], True, {"extra": 1}]
    + [{"before_run_id": x} for x in [True, False, 0, -1, "200", 2.0, [], {}]],
)
def test_encode_rejects_invalid_cursor(checkpoint, api, cursor):
    with pytest.raises(ValueError):
        checkpoint.encode(api, cursor, 400, 1, SHA)


@pytest.mark.parametrize("field", ["run_id", "attempt", "control_sha"])
@pytest.mark.parametrize("value", [True, False, 0, -1, "400", None, [], {}])
def test_encode_rejects_invalid_producer_arguments(checkpoint, api, field, value):
    args = {"run_id": 400, "attempt": 1, "control_sha": SHA}
    args[field] = value
    with pytest.raises(ValueError):
        checkpoint.encode(api, {}, args["run_id"], args["attempt"], args["control_sha"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 401),
        ("id", True),
        ("run_attempt", 2),
        ("run_attempt", True),
        ("workflow_id", 21),
        ("workflow_id", True),
        ("path", ".github/workflows/ppu-pr-ci.yaml"),
        ("repository", {"id": 10, "full_name": "attacker/fork"}),
        ("repository", {"id": 11, "full_name": REPO}),
        ("head_repository", {"id": 11, "full_name": "attacker/fork"}),
        ("head_branch", "feature"),
        ("head_sha", "b" * 40),
        ("head_sha", "bad"),
        ("event", "pull_request"),
        ("event", "workflow_run"),
    ],
)
def test_encode_rejects_untrusted_live_producer(checkpoint, api, field, value):
    api.responses["actions/runs/400"][field] = value
    with pytest.raises(ValueError):
        checkpoint.encode(api, {}, 400, 1, SHA)


@pytest.mark.parametrize(
    "route,field,value",
    [
        ("", "full_name", "attacker/fork"),
        ("", "id", True),
        ("", "default_branch", "new-main"),
        (f"actions/workflows/{REPORTER}", "id", 21),
        (f"actions/workflows/{REPORTER}", "id", True),
        (f"actions/workflows/{REPORTER}", "path", "other.yml"),
        (f"actions/workflows/{EXECUTOR}", "id", True),
        (f"actions/workflows/{EXECUTOR}", "path", "other.yml"),
    ],
)
def test_encode_rejects_changed_repository_or_workflows(
    checkpoint, api, route, field, value
):
    api.responses[route][field] = value
    with pytest.raises(ValueError):
        checkpoint.encode(api, {}, 400, 1, SHA)


def test_encode_rejects_other_api_repository_without_reads(checkpoint, api):
    api.repo = "attacker/fork"
    with pytest.raises(ValueError):
        checkpoint.encode(api, {}, 400, 1, SHA)
    assert api.reads == []


def test_encode_budget_exhaustion_propagates(checkpoint, api):
    api.limit = 0
    with pytest.raises(RuntimeError, match="budget exhausted"):
        checkpoint.encode(api, {}, 400, 1, SHA)


def test_encode_missing_metadata_is_not_an_empty_cursor(checkpoint, api):
    api.responses["actions/runs/400"] = HTTPError("test", 404, "missing", {}, None)
    with pytest.raises(HTTPError):
        checkpoint.encode(api, {}, 400, 1, SHA)


def artifact():
    return {
        "id": 500,
        "name": "ppu-report-cursor-v1",
        "expired": False,
        "expires_at": "2999-01-01T00:00:00Z",
        "size_in_bytes": 1000,
        "workflow_run": {
            "id": 400,
            "repository_id": 10,
            "head_repository_id": 10,
            "head_branch": "main",
            "head_sha": SHA,
        },
        "archive_download_url": "https://attacker.invalid/never-use-this",
    }


def archive(data=None, name="cursor.json", extra=False):
    if data is None:
        data = json.dumps(payload()).encode()
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, data)
        if extra:
            zf.writestr("extra.txt", b"poison")
    return target.getvalue()


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, headers=None):
        super().__init__(body)
        self.headers = headers or {}
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


@pytest.fixture
def transport(api, monkeypatch):
    api.responses[LIST] = {"total_count": 1, "artifacts": [artifact()]}
    signed = "https://storage.example.invalid/cursor.zip?signature=secret"
    redirect = HTTPError(
        "https://api.github.com", 302, "Found", {"Location": signed}, None
    )
    response = Response(archive())
    requests = []
    replies = [redirect, response]

    def open_request(request, timeout):
        assert timeout > 0
        assert request.get_method() == "GET"
        assert api.consumed == len(api.reads) + len(requests) + 1
        requests.append(request)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    opener = Mock()
    opener.open.side_effect = open_request
    builder = Mock(return_value=opener)
    monkeypatch.setattr(urllib.request, "build_opener", builder)
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **kw: pytest.fail("Unbounded urlopen")
    )
    return {
        "requests": requests,
        "replies": replies,
        "builder": builder,
        "response": response,
        "signed": signed,
    }


def test_restore_uses_fixed_id_and_strips_bearer_at_redirect(
    checkpoint, api, transport
):
    assert checkpoint.restore(api) == {"before_run_id": 200}
    first, second = transport["requests"]
    assert (
        first.full_url
        == f"https://api.github.com/repos/{REPO}/actions/artifacts/500/zip"
    )
    assert first.get_header("Authorization") == f"Bearer {api.token}"
    assert second.full_url == transport["signed"]
    assert second.get_header("Authorization") is None
    assert api.token not in str(second.header_items())
    assert api.reads[0] == LIST
    assert len(api.reads) == 5
    assert api.consumed == 7
    assert transport["response"].read_sizes == [64 * 1024 + 1]
    for call in transport["builder"].call_args_list:
        assert any(isinstance(h, urllib.request.HTTPRedirectHandler) for h in call.args)
        for handler in call.args:
            if isinstance(handler, urllib.request.HTTPRedirectHandler):
                assert (
                    handler.redirect_request(
                        first, None, 302, "", {}, transport["signed"]
                    )
                    is None
                )


@pytest.mark.parametrize("conclusion", ["success", "failure", "cancelled", None])
def test_restore_accepts_old_sha_despite_downstream_failure(
    checkpoint, api, transport, conclusion
):
    api.responses["actions/runs/400"]["conclusion"] = conclusion
    api.responses[""]["current_main_sha"] = "b" * 40
    assert checkpoint.restore(api) == {"before_run_id": 200}
    assert not any(path.startswith(("commits/", "branches/")) for path in api.reads)


def test_restore_accepts_null_cursor(checkpoint, api, transport):
    data = payload()
    data["before_run_id"] = None
    transport["replies"][1] = Response(archive(json.dumps(data).encode()))
    assert checkpoint.restore(api) == {"before_run_id": None}


def test_restore_rejects_rerun_attempt_ambiguity(checkpoint, api, transport, capsys):
    api.responses["actions/runs/400"]["run_attempt"] = 2
    with pytest.raises(ValueError, match="attempt"):
        checkpoint.restore(api)
    assert not capsys.readouterr().err


def test_restore_accepts_matching_current_attempt(checkpoint, api, transport):
    api.responses["actions/runs/400"].update(run_attempt=2, event="workflow_dispatch")
    data = payload()
    data["producer"]["attempt"] = 2
    transport["replies"][1] = Response(archive(json.dumps(data).encode()))
    assert checkpoint.restore(api) == {"before_run_id": 200}


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", True),
        ("id", "500"),
        ("name", "other-artifact"),
        ("expired", "false"),
        ("expires_at", "not-a-date"),
        ("expires_at", "2999-01-01T00:00:00"),
        ("size_in_bytes", True),
        ("size_in_bytes", 65537),
        ("workflow_run", None),
    ],
)
def test_restore_rejects_malformed_artifact_metadata(
    checkpoint, api, transport, field, value
):
    api.responses[LIST]["artifacts"][0][field] = value
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert transport["requests"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", True),
        ("repository_id", 11),
        ("head_repository_id", 11),
        ("repository_id", True),
        ("head_sha", "b" * 40),
        ("head_sha", "bad"),
        ("head_branch", "feature"),
    ],
)
def test_restore_binds_artifact_to_live_producer(
    checkpoint, api, transport, field, value
):
    api.responses[LIST]["artifacts"][0]["workflow_run"][field] = value
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert transport["requests"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 401),
        ("workflow_id", 21),
        ("path", ".github/workflows/ppu-pr-ci.yaml"),
        ("repository", {"id": 11, "full_name": "attacker/fork"}),
        ("head_repository", {"id": 11, "full_name": "attacker/fork"}),
        ("head_branch", "feature"),
        ("head_sha", "b" * 40),
        ("event", "pull_request"),
        ("event", "workflow_run"),
        ("run_attempt", True),
    ],
)
def test_restore_rejects_wrong_producer(checkpoint, api, transport, field, value):
    api.responses["actions/runs/400"][field] = value
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert transport["requests"] == []


@pytest.mark.parametrize(
    "expiry", [{"expired": True}, {"expires_at": "2000-01-01T00:00:00Z"}]
)
def test_expired_artifact_starts_empty_with_warning(
    checkpoint, api, transport, capsys, expiry
):
    api.responses[LIST]["artifacts"][0].update(expiry)
    assert checkpoint.restore(api) == {"before_run_id": None}
    assert "warning" in capsys.readouterr().err.lower()
    assert transport["requests"] == []


def test_expiry_cannot_bypass_producer_trust(checkpoint, api, transport, capsys):
    api.responses[LIST]["artifacts"][0]["expired"] = True
    api.responses["actions/runs/400"]["event"] = "pull_request"
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert not capsys.readouterr().err


def test_latest_poisoned_artifact_cannot_be_skipped(checkpoint, api, transport):
    older = artifact()
    older["id"] = 499
    api.responses[LIST]["artifacts"].append(older)
    api.responses[LIST]["total_count"] = 2
    api.responses[LIST]["artifacts"][0]["workflow_run"]["head_sha"] = "b" * 40
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert transport["requests"] == []


def test_artifact_lookup_is_one_bounded_page(checkpoint, api, transport):
    api.responses[LIST] = {"total_count": 50000, "artifacts": [artifact()] * 100}
    assert checkpoint.restore(api) == {"before_run_id": 200}
    assert api.reads.count(LIST) == 1
    assert len(api.reads) == 5


@pytest.mark.parametrize(
    "listing", [None, {}, {"artifacts": None}, {"total_count": 1, "artifacts": []}]
)
def test_malformed_listing_fails_closed(checkpoint, api, transport, listing, capsys):
    api.responses[LIST] = listing
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert not capsys.readouterr().err


def test_no_artifact_starts_empty_with_warning(checkpoint, api, transport, capsys):
    api.responses[LIST] = {"total_count": 0, "artifacts": []}
    assert checkpoint.restore(api) == {"before_run_id": None}
    assert "warning" in capsys.readouterr().err.lower()
    assert api.reads == [LIST]
    assert transport["requests"] == []


@pytest.mark.parametrize("code", [403, 429, 500, 502, 503])
@pytest.mark.parametrize(
    "stage",
    [
        LIST,
        "",
        f"actions/workflows/{REPORTER}",
        f"actions/workflows/{EXECUTOR}",
        "actions/runs/400",
        "redirect",
        "download",
    ],
)
def test_transient_errors_propagate_without_reset(
    checkpoint, api, transport, capsys, code, stage
):
    error = HTTPError("test", code, "transient", {}, None)
    if stage in {"redirect", "download"}:
        transport["replies"][stage == "download"] = error
    else:
        api.responses[stage] = error
    with pytest.raises(HTTPError) as caught:
        checkpoint.restore(api)
    assert caught.value is error
    assert not capsys.readouterr().err


@pytest.mark.parametrize("code", [404, 410])
@pytest.mark.parametrize("stage", [LIST, "redirect", "download"])
def test_missing_artifact_starts_empty_with_warning(
    checkpoint, api, transport, capsys, code, stage
):
    error = HTTPError("test", code, "deleted", {}, None)
    if stage == LIST:
        api.responses[stage] = error
    else:
        transport["replies"][stage == "download"] = error
    assert checkpoint.restore(api) == {"before_run_id": None}
    assert "warning" in capsys.readouterr().err.lower()


def test_missing_producer_is_not_trusted(checkpoint, api, transport, capsys):
    api.responses["actions/runs/400"] = HTTPError("test", 404, "missing", {}, None)
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert not capsys.readouterr().err


@pytest.mark.parametrize("limit", [0, 1, 4, 5, 6])
def test_restore_accounts_for_every_http_call(checkpoint, api, transport, limit):
    api.limit = limit
    with pytest.raises(RuntimeError, match="budget exhausted"):
        checkpoint.restore(api)
    assert api.consumed == limit
    assert len(api.reads) + len(transport["requests"]) == limit


@pytest.mark.parametrize(
    "location",
    [
        "http://storage.example.invalid/a.zip",
        "https://user:pass@storage.example.invalid/a.zip",
        "https://user@storage.example.invalid/a.zip",
        "https://@storage.example.invalid/a.zip",
        "file:///tmp/cursor.zip",
        "//storage.example.invalid/a.zip",
        "https:///missing-host",
        "https://storage.example.invalid/\nheader",
        "https://storage.example.invalid\\@evil.invalid/a.zip",
        "",
        None,
    ],
)
def test_redirect_rejects_unsafe_location(checkpoint, api, transport, location):
    transport["replies"][0].headers["Location"] = location
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert len(transport["requests"]) == 1


def test_signed_download_does_not_follow_another_redirect(checkpoint, api, transport):
    transport["replies"][1] = HTTPError(
        "test", 302, "Found", {"Location": "https://other.invalid"}, None
    )
    with pytest.raises((HTTPError, ValueError)):
        checkpoint.restore(api)
    assert len(transport["requests"]) == 2


def test_download_requires_expected_redirect(checkpoint, api, transport):
    transport["replies"][0] = Response(archive())
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert len(transport["requests"]) == 1


@pytest.mark.parametrize(
    "header", [{}, {"Content-Length": "65537"}, {"Content-Length": "bad"}]
)
def test_download_rejects_oversize_or_invalid_length(
    checkpoint, api, transport, header
):
    response = Response(b"x" * 65537, header)
    transport["replies"][1] = response
    with pytest.raises(ValueError):
        checkpoint.restore(api)
    assert all(0 <= size <= 65537 for size in response.read_sizes)


@pytest.mark.parametrize(
    "name",
    [
        "other.json",
        "../cursor.json",
        "nested/cursor.json",
        "/cursor.json",
        "cursor.json/",
    ],
)
def test_zip_rejects_other_names(checkpoint, api, transport, name):
    transport["replies"][1] = Response(archive(name=name))
    with pytest.raises(ValueError):
        checkpoint.restore(api)


@pytest.mark.parametrize(
    "mode", [stat.S_IFLNK | 0o777, stat.S_IFDIR | 0o755, stat.S_IFIFO | 0o600]
)
def test_zip_rejects_nonregular_member(checkpoint, api, transport, mode):
    info = zipfile.ZipInfo("cursor.json")
    info.create_system = 3
    info.external_attr = mode << 16
    transport["replies"][1] = Response(archive(name=info))
    with pytest.raises(ValueError):
        checkpoint.restore(api)


@pytest.mark.parametrize(
    "kind", ["extra", "encrypted", "malformed", "empty", "large-member", "corrupt"]
)
def test_zip_rejects_unsafe_archive(checkpoint, api, transport, kind):
    raw = archive()
    if kind == "extra":
        raw = archive(extra=True)
    elif kind == "encrypted":
        raw = bytearray(raw)
        struct.pack_into("<H", raw, 6, 1)
        struct.pack_into("<H", raw, raw.index(b"PK\x01\x02") + 8, 1)
    elif kind == "malformed":
        raw = b"not a zip"
    elif kind == "empty":
        target = io.BytesIO()
        with zipfile.ZipFile(target, "w"):
            pass
        raw = target.getvalue()
    elif kind == "large-member":
        raw = archive(b" " * 8193)
    elif kind == "corrupt":
        raw = bytearray(raw)
        struct.pack_into("<I", raw, raw.index(b"PK\x01\x02") + 16, 0)
    transport["replies"][1] = Response(raw)
    with pytest.raises(ValueError):
        checkpoint.restore(api)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        b"null",
        b"true",
        b"\xff",
        b'{"schema":1,"schema":1}',
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_json_rejects_malformed_or_duplicate_keys(checkpoint, api, transport, raw):
    transport["replies"][1] = Response(archive(raw))
    with pytest.raises(ValueError):
        checkpoint.restore(api)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", True),
        ("schema", 1.0),
        ("schema", 2),
        ("repository", "attacker/fork"),
        ("executor_workflow_id", True),
        ("executor_workflow_id", 31),
        ("executor_workflow_id", "30"),
        ("before_run_id", True),
        ("before_run_id", False),
        ("before_run_id", 0),
        ("before_run_id", -1),
        ("before_run_id", "200"),
        ("before_run_id", 200.0),
        ("before_run_id", []),
        ("before_run_id", {}),
        ("producer", None),
        ("extra", "not allowed"),
    ],
)
def test_json_rejects_poisoned_schema(checkpoint, api, transport, field, value):
    data = payload()
    data[field] = value
    transport["replies"][1] = Response(archive(json.dumps(data).encode()))
    with pytest.raises(ValueError):
        checkpoint.restore(api)


@pytest.mark.parametrize("field", list(payload()))
def test_json_requires_all_schema_keys(checkpoint, api, transport, field):
    data = payload()
    del data[field]
    transport["replies"][1] = Response(archive(json.dumps(data).encode()))
    with pytest.raises(ValueError):
        checkpoint.restore(api)


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", True),
        ("run_id", 401),
        ("attempt", True),
        ("attempt", 2),
        ("attempt", "1"),
        ("control_sha", "b" * 40),
        ("control_sha", "bad"),
        ("extra", 1),
    ],
)
def test_json_requires_bound_producer(checkpoint, api, transport, field, value):
    data = payload()
    data["producer"][field] = value
    transport["replies"][1] = Response(archive(json.dumps(data).encode()))
    with pytest.raises(ValueError):
        checkpoint.restore(api)


@pytest.mark.parametrize("field", list(payload()["producer"]))
def test_json_requires_all_producer_keys(checkpoint, api, transport, field):
    data = payload()
    del data["producer"][field]
    transport["replies"][1] = Response(archive(json.dumps(data).encode()))
    with pytest.raises(ValueError):
        checkpoint.restore(api)


def test_json_allows_exact_size_limit(checkpoint, api, transport):
    raw = json.dumps(payload()).encode().ljust(8192, b" ")
    transport["replies"][1] = Response(archive(raw))
    assert checkpoint.restore(api) == {"before_run_id": 200}
