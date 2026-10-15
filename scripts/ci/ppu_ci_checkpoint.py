# SPDX-License-Identifier: Apache-2.0
"""Trusted report-only cursor artifacts; never authorization to execute PPU CI."""

from __future__ import annotations

import io
import json
import stat
import sys
import urllib.request
import zipfile
import zlib
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.parse import urlsplit

from scripts.ci.ppu_ci_common import EXECUTOR, REPOSITORY, Rejected, require, sha

NAME = "ppu-report-cursor-v1"
FILENAME = "cursor.json"
_REPORTER = "ppu-ci-report.yml"
_MAX_FILE = 8 * 1024
_MAX_ZIP = 64 * 1024


def _positive(value):
    require(type(value) is int and value > 0, "Invalid cursor identifier.")
    return value


def _cursor(value):
    require(
        type(value) is dict and set(value) <= {"before_run_id"},
        "Invalid report cursor.",
    )
    before = value.get("before_run_id")
    if before is not None:
        _positive(before)
    return {"before_run_id": before}


def _producer(api, run_id):
    require(api.repo == REPOSITORY, "Cursor repository mismatch.")
    _positive(run_id)
    repository = api.get("")
    reporter = api.get(f"actions/workflows/{_REPORTER}")
    executor = api.get(f"actions/workflows/{EXECUTOR}")
    run = api.get(f"actions/runs/{run_id}")
    try:
        repository_id = _positive(repository["id"])
        require(
            repository["full_name"] == api.repo
            and isinstance(repository["default_branch"], str)
            and repository["default_branch"],
            "Cursor repository mismatch.",
        )
        for workflow, filename in ((reporter, _REPORTER), (executor, EXECUTOR)):
            _positive(workflow["id"])
            require(
                workflow["path"] == f".github/workflows/{filename}",
                "Cursor workflow mismatch.",
            )
        require(
            _positive(run["id"]) == run_id
            and _positive(run["workflow_id"]) == reporter["id"]
            and run["path"] == reporter["path"]
            and run["event"] in {"schedule", "workflow_dispatch"}
            and run["head_branch"] == repository["default_branch"],
            "Untrusted cursor producer.",
        )
        for key in ("repository", "head_repository"):
            require(
                _positive(run[key]["id"]) == repository_id
                and run[key]["full_name"] == api.repo,
                "Cursor producer repository mismatch.",
            )
        _positive(run["run_attempt"])
        sha(run["head_sha"])
    except (KeyError, TypeError) as error:
        raise Rejected("Malformed cursor producer metadata.") from error
    return repository, executor, run


def encode(api, cursor, producer_run_id, producer_attempt, control_sha) -> str:
    """Return a small JSON scheduling hint bound to the current trusted producer."""
    cursor = _cursor(cursor)
    _positive(producer_run_id)
    _positive(producer_attempt)
    sha(control_sha)
    _, executor, run = _producer(api, producer_run_id)
    require(
        run["run_attempt"] == producer_attempt and run["head_sha"] == control_sha,
        "Cursor producer attempt or SHA mismatch.",
    )
    result = json.dumps(
        {
            "schema": 1,
            "repository": api.repo,
            "executor_workflow_id": executor["id"],
            "producer": {
                "run_id": producer_run_id,
                "attempt": producer_attempt,
                "control_sha": control_sha,
            },
            **cursor,
        },
        separators=(",", ":"),
    )
    require(len(result.encode()) <= _MAX_FILE, "Cursor JSON is too large.")
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _download(api, artifact_id):
    _positive(artifact_id)
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(
        f"https://api.github.com/repos/{api.repo}/actions/artifacts/{artifact_id}/zip",
        headers={
            "Authorization": f"Bearer {api.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "vllm-sail-ppu-ci",
        },
        method="GET",
    )
    api.consume()
    try:
        with opener.open(request, timeout=30):
            raise Rejected("Expected a cursor artifact redirect.")
    except HTTPError as error:
        if error.code != 302:
            raise
        location = error.headers.get("Location")
        error.close()
    require(
        isinstance(location, str)
        and location
        and not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in location)
        and "\\" not in location,
        "Invalid cursor download redirect.",
    )
    try:
        parsed = urlsplit(location)
        require(
            parsed.scheme == "https"
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.fragment,
            "Unsafe cursor download redirect.",
        )
    except ValueError as error:
        raise Rejected("Unsafe cursor download redirect.") from error
    request = urllib.request.Request(location, method="GET")
    api.consume()
    with opener.open(request, timeout=30) as response:
        require(response.status == 200, "Unexpected cursor download response.")
        length = response.headers.get("Content-Length")
        if length is not None:
            try:
                require(0 <= int(length) <= _MAX_ZIP, "Cursor ZIP is too large.")
            except (ValueError, TypeError) as error:
                raise Rejected("Invalid cursor ZIP length.") from error
        raw = response.read(_MAX_ZIP + 1)
    require(len(raw) <= _MAX_ZIP, "Cursor ZIP is too large.")
    return raw


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate cursor JSON key.")
        result[key] = value
    return result


def _read_zip(raw):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = archive.infolist()
            require(len(members) == 1, "Cursor ZIP must contain exactly one file.")
            member = members[0]
            require(
                member.filename == member.orig_filename == FILENAME
                and not member.is_dir()
                and not member.flag_bits & 1
                and not member.external_attr & 0x10
                and stat.S_IFMT(member.external_attr >> 16) in {0, stat.S_IFREG}
                and member.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                and 0 <= member.file_size <= _MAX_FILE,
                "Unsafe cursor ZIP member.",
            )
            with archive.open(member) as source:
                data = source.read(_MAX_FILE + 1)
            require(
                len(data) == member.file_size and len(data) <= _MAX_FILE,
                "Invalid cursor JSON size.",
            )
        return json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (
        zipfile.BadZipFile,
        RuntimeError,
        NotImplementedError,
        EOFError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        zlib.error,
    ) as error:
        raise Rejected("Malformed cursor ZIP or JSON.") from error


def _record_cursor(record, api, executor, run):
    require(
        type(record) is dict
        and set(record)
        == {
            "schema",
            "repository",
            "executor_workflow_id",
            "producer",
            "before_run_id",
        },
        "Invalid cursor JSON schema.",
    )
    require(
        type(record["schema"]) is int
        and record["schema"] == 1
        and record["repository"] == api.repo
        and _positive(record["executor_workflow_id"]) == executor["id"],
        "Cursor schema, repository or executor mismatch.",
    )
    producer = record["producer"]
    require(
        type(producer) is dict
        and set(producer) == {"run_id", "attempt", "control_sha"},
        "Invalid cursor producer schema.",
    )
    require(
        _positive(producer["run_id"]) == run["id"]
        and _positive(producer["attempt"]) == run["run_attempt"]
        and sha(producer["control_sha"]) == run["head_sha"],
        "Cursor producer attempt or SHA mismatch.",
    )
    return _cursor({"before_run_id": record["before_run_id"]})


def _empty(reason):
    print(f"Warning: report cursor {reason}; starting a new sweep.", file=sys.stderr)
    return {"before_run_id": None}


def restore(api) -> dict:
    """Restore only the newest trusted hint; missing artifacts restart the sweep."""
    require(api.repo == REPOSITORY, "Cursor repository mismatch.")
    try:
        listing = api.get(f"actions/artifacts?name={NAME}&per_page=100&page=1")
    except HTTPError as error:
        if error.code in {404, 410}:
            return _empty("is unavailable")
        raise
    try:
        artifacts = listing["artifacts"]
        require(
            type(artifacts) is list
            and len(artifacts) <= 100
            and type(listing["total_count"]) is int
            and listing["total_count"] >= len(artifacts)
            and (artifacts or listing["total_count"] == 0),
            "Malformed cursor artifact listing.",
        )
        if not artifacts:
            return _empty("is missing")
        artifact = artifacts[0]
        _positive(artifact["id"])
        require(
            artifact["name"] == NAME
            and type(artifact["expired"]) is bool
            and type(artifact["size_in_bytes"]) is int
            and 0 <= artifact["size_in_bytes"] <= _MAX_ZIP,
            "Malformed cursor artifact metadata.",
        )
        require(isinstance(artifact["expires_at"], str), "Invalid cursor expiry.")
        expires = datetime.fromisoformat(artifact["expires_at"].replace("Z", "+00:00"))
        require(expires.tzinfo is not None, "Invalid cursor expiry timezone.")
        source = artifact["workflow_run"]
        _positive(source["id"])
    except (KeyError, TypeError, ValueError) as error:
        raise Rejected("Malformed cursor artifact metadata.") from error
    try:
        repository, executor, run = _producer(api, source["id"])
    except HTTPError as error:
        if error.code in {404, 410}:
            raise Rejected("Cursor producer metadata is unavailable.") from error
        raise
    try:
        require(
            _positive(source["repository_id"]) == repository["id"]
            and _positive(source["head_repository_id"]) == repository["id"]
            and source["head_branch"] == run["head_branch"]
            and sha(source["head_sha"]) == run["head_sha"],
            "Cursor artifact producer mismatch.",
        )
    except (KeyError, TypeError) as error:
        raise Rejected("Malformed cursor artifact producer.") from error
    if artifact["expired"] or expires <= datetime.now(timezone.utc):
        return _empty("has expired")
    try:
        raw = _download(api, artifact["id"])
    except HTTPError as error:
        if error.code in {404, 410}:
            return _empty("was deleted or expired")
        raise
    return _record_cursor(_read_zip(raw), api, executor, run)
