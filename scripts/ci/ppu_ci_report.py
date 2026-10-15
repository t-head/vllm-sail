# SPDX-License-Identifier: Apache-2.0
"""Report PR checks using only trusted requests and the GitHub jobs API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError

from scripts.ci.ppu_ci_command import requests_for_head, save_record
from scripts.ci.ppu_ci_common import (
    CONTEXTS,
    EXECUTOR,
    STAGES,
    TITLE,
    GitHub,
    Rejected,
    bound_checks,
    execution_evidence,
    execution_identity,
    matching_runs,
    number,
    outcome,
    request_record,
    require,
    run_url,
)


def identify(event, api, known_runs=None):
    run = api.get(f"actions/runs/{number(event['workflow_run']['id'])}")
    title = TITLE.fullmatch(run.get("display_title", ""))
    require(title, "Unexpected executor run title.")
    check = api.get(f"check-runs/{number(title[3])}")
    record = request_record(api, check)
    execution_identity(api, run, check, record)
    require(
        record["default_branch"] == api.get("")["default_branch"],
        "Default branch changed.",
    )
    runs = matching_runs(api, check, record, known_runs)
    require(
        runs
        and runs[0]["id"] == run["id"]
        and record.get("run_id", run["id"]) == run["id"],
        "Replayed execution.",
    )
    return run, check, record


def aggregate(api, run, check, record):
    stages, sources, jobs = execution_evidence(api, check, record, run)
    result = {name: stages[stage] for stage, name in STAGES.items()}
    result[CONTEXTS[-1]] = outcome(jobs.get("PPU / " + CONTEXTS[-1], {}))
    if any(value != "success" for value in result.values()):
        result[CONTEXTS[-1]] = "failure"
    summaries = {}
    for stage, source in sources.items():
        mode = record["validation"]["plan"][stage]["mode"]
        link = f"{run_url(api, source['run_id'])}/job/{source['job_id']}"
        summaries[STAGES[stage]] = (
            f"{mode}: request {source['request_id']}, run {source['run_id']}, attempt 1, job {source['job_id']}. {link}"
        )
    return result, summaries


def report(event, api, expected_pr=None):
    require(
        event.get("action") in {"requested", "in_progress", "completed"},
        "Unexpected report event.",
    )
    run, check, record = identify(event, api)
    require(
        expected_pr is None or number(expected_pr) == record["pr_number"],
        "Report lock identity mismatch.",
    )
    if event["workflow_run"]["run_attempt"] != run["run_attempt"]:
        return
    latest = requests_for_head(
        api, {"number": record["pr_number"], "head": {"sha": record["head_sha"]}}
    )
    if not latest or latest[0][0]["id"] != check["id"]:
        return
    statuses = {c["name"]: c for c in bound_checks(api, check, record)}
    link = run_url(api, run["id"])
    gate = statuses[CONTEXTS[-1]]
    api.patch(
        f"check-runs/{gate['id']}", {"status": "in_progress", "details_url": link}
    )
    if run["status"] != "completed" and not record.get("cancelled"):
        for name in CONTEXTS[:-1]:
            api.patch(
                f"check-runs/{statuses[name]['id']}",
                {"status": "in_progress", "details_url": link},
            )
        return
    summaries = {}
    try:
        result, summaries = aggregate(api, run, check, record)
    except (Rejected, KeyError, TypeError) as error:
        result = dict.fromkeys(CONTEXTS, "failure")
        summaries = dict.fromkeys(CONTEXTS, str(error))
    if record.get("cancelled") or run["conclusion"] == "cancelled":
        result = dict.fromkeys(CONTEXTS, "cancelled")
    elif run["conclusion"] != "success":
        result[CONTEXTS[-1]] = "failure"
    for name in CONTEXTS:
        api.patch(
            f"check-runs/{statuses[name]['id']}",
            {
                "status": "completed",
                "conclusion": result[name],
                "details_url": link,
                "output": {
                    "title": name,
                    "summary": f"Attempt {run['run_attempt']}: {result[name]}. {summaries.get(name, link)}",
                },
            },
        )
    record["run_id"] = run["id"]
    record["reported"] = {**report_identity(run), "conclusions": result}
    save_record(api, check, record, status="completed", conclusion=result[CONTEXTS[-1]])


def report_identity(run):
    return {
        "run_id": run["id"],
        "run_attempt": run["run_attempt"],
        "updated_at": run.get("updated_at"),
        "conclusion": run["conclusion"],
    }


def finalized(api, run, check, record):
    reported = record.get("reported", {})
    if any(reported.get(key) != value for key, value in report_identity(run).items()):
        return False
    results = reported.get("conclusions", {})
    if set(results) != set(CONTEXTS):
        return False
    if record.get("cancelled") and any(
        value != "cancelled" for value in results.values()
    ):
        return False
    if (
        check["status"] != "completed"
        or check.get("conclusion") != results[CONTEXTS[-1]]
    ):
        return False
    return all(
        c["status"] == "completed"
        and c.get("conclusion") == results[c["name"]]
        and c.get("details_url") == run_url(api, run["id"])
        for c in bound_checks(api, check, record)
    )


class ReadBudget(Exception):
    pass


class DiscoveryAPI:
    def __init__(self, api):
        self.api = api
        self.repo = api.repo
        self.token = getattr(api, "token", "")
        self.limit = 80
        self.reads = 0
        self.cache = {}

    def consume(self):
        if self.reads >= self.limit:
            raise ReadBudget("Discovery read budget exhausted.")
        self.reads += 1

    def get(self, path, *, fresh=False):
        if fresh or path not in self.cache:
            self.consume()
            self.cache[path] = self.api.get(path)
        return self.cache[path]

    pages = GitHub.pages


def routing_identity(api, run_id):
    run = api.get(f"actions/runs/{number(run_id)}")
    title = TITLE.fullmatch(run.get("display_title", ""))
    require(title, "Unexpected executor run title.")
    check = api.get(f"check-runs/{number(title[3])}")
    record = request_record(api, check)
    execution_identity(api, run, check, record)
    require(
        record["default_branch"] == api.get("")["default_branch"]
        and record.get("run_id", run["id"]) == run["id"],
        "Execution routing identity changed.",
    )
    # Uniqueness and all write authorization are rechecked under the PR lock.
    return run, check, record


def scan_page(api, before):
    def page(index, *, fresh=False):
        data = api.get(
            f"actions/workflows/{EXECUTOR}/runs?per_page=100&page={index}", fresh=fresh
        )
        runs = data["workflow_runs"]
        ids = [number(run["id"]) for run in runs]
        require(ids == sorted(set(ids), reverse=True), "Unordered executor listing.")
        return data, runs

    data, runs = page(1)
    if before is None or not runs:
        return runs
    low, high = 1, max(1, (data["total_count"] + 99) // 100)
    while low < high:
        middle = (low + high) // 2
        _, items = page(middle)
        if not items or items[-1]["id"] < before:
            high = middle
        else:
            low = middle + 1
    _, runs = page(low)
    # New arrivals can extend the tail beyond page one's cached total_count.
    while runs and runs[-1]["id"] >= before:
        low += 1
        _, runs = page(low)
    if low > 1 and (not runs or runs[0]["id"] < before):
        # Deletions can move unseen older runs into the preceding page.
        _, previous = page(low - 1, fresh=True)
        require(
            previous and previous[-1]["id"] >= before,
            "Executor pagination changed; keep the checkpoint and retry.",
        )
    return [run for run in runs if run["id"] < before]


def discover(event, api, cursor=None):
    api = api if isinstance(api, DiscoveryAPI) else DiscoveryAPI(api)
    selected = event.get("workflow_run", {}).get("id") or event.get("inputs", {}).get(
        "run_id"
    )
    if selected:
        run, _, record = routing_identity(api, selected)
        return [{"pr_number": record["pr_number"], "run_id": run["id"]}]
    cursor = {} if cursor is None else cursor
    before = cursor.get("before_run_id")
    targets, inspected = [], 0
    limit = api.limit
    api.limit -= 8  # Reserve reads for checkpoint provenance after scanning.
    try:
        while inspected < 5 and api.reads + 8 < api.limit:
            runs = scan_page(api, before)
            if not runs:
                before = None
                break
            for candidate in runs:
                if inspected == 5 or api.reads + 8 >= api.limit:
                    break
                if candidate["status"] != "completed" or not TITLE.fullmatch(
                    candidate.get("display_title", "")
                ):
                    before = candidate["id"]
                    continue
                inspected += 1
                try:
                    run, check, record = routing_identity(api, candidate["id"])
                    target = {"pr_number": record["pr_number"], "run_id": run["id"]}
                    try:
                        latest = requests_for_head(
                            api,
                            {
                                "number": record["pr_number"],
                                "head": {"sha": record["head_sha"]},
                            },
                        )
                        if (
                            latest
                            and latest[0][0]["id"] == check["id"]
                            and not finalized(api, run, check, record)
                        ):
                            targets.append(target)
                    except ReadBudget:
                        # A costly optional probe must not starve older requests.
                        targets.append(target)
                except HTTPError as error:
                    if error.code not in {404, 410}:
                        raise
                    print(
                        f"Ignoring missing source for run {candidate['id']}: {error}",
                        file=sys.stderr,
                    )
                except Rejected as error:
                    print(
                        f"Ignoring untrusted run {candidate['id']}: {error}",
                        file=sys.stderr,
                    )
                before = candidate["id"]
    except ReadBudget:
        pass
    finally:
        api.limit = limit
    cursor["before_run_id"] = before
    return targets


def reconcile(api, run_id, pr_number):
    # 此函数仅从已取得该 PR 共享锁的 job 调用，重新读取全部执行证据。
    run = api.get(f"actions/runs/{number(run_id)}")
    report(
        {
            "action": "completed" if run["status"] == "completed" else "in_progress",
            "workflow_run": run,
        },
        api,
        expected_pr=pr_number,
    )


def main():
    api = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    if len(sys.argv) > 1 and sys.argv[1] == "discover":
        api = DiscoveryAPI(api)
        selected = event.get("workflow_run", {}).get("id") or event.get(
            "inputs", {}
        ).get("run_id")
        if selected:
            targets = discover(event, api)
        else:
            from scripts.ci import ppu_ci_checkpoint

            cursor = ppu_ci_checkpoint.restore(api)
            targets = discover(event, api, cursor=cursor)
            encoded = ppu_ci_checkpoint.encode(
                api,
                cursor,
                number(os.environ["GITHUB_RUN_ID"]),
                number(os.environ["GITHUB_RUN_ATTEMPT"]),
                os.environ["GITHUB_WORKFLOW_SHA"],
            )
            path = Path(os.environ["CURSOR_FILE"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded)
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(
                f"matrix={json.dumps({'include': targets}, separators=(',', ':'))}\n"
            )
            stream.write(f"has_targets={str(bool(targets)).lower()}\n")
            if not selected:
                stream.write("cursor_written=true\n")
    elif os.environ.get("SOURCE_RUN_ID"):
        reconcile(api, os.environ["SOURCE_RUN_ID"], os.environ["REPORT_PR_NUMBER"])
    else:
        report(event, api)


if __name__ == "__main__":
    main()
