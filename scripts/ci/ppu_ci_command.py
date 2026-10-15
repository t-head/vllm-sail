# SPDX-License-Identifier: Apache-2.0
"""Trusted PPU comment controller; never execute this from a PR revision."""

from __future__ import annotations

import json
import os
import re
import sys
from urllib.error import HTTPError
from urllib.parse import quote

from scripts.ci.ppu_ci_common import (
    ACTIONS_APP_ID,
    CONTEXTS,
    EXECUTOR,
    REQUEST_NAME,
    TITLE,
    GitHub,
    Rejected,
    behind_base,
    bound_checks,
    checks,
    context_from_env,
    execution_identity,
    matching_runs,
    number,
    pr_identity,
    request_plan,
    request_record,
    require,
    retry_plan,
    run_url,
    sha,
    unique_execution,
)

COMMANDS = {"/ci run", "/ci run --allow-stale", "/ci retry", "/ci cancel"}
READY = {"ready", "ready-run-all-tests"}
HELP = (
    "PPU CI is authorized. Comment `/ci run` to test the current head, "
    "`/ci retry` to retry its failed jobs, or `/ci cancel` to cancel its PPU run. "
    "New commits require a new `/ci run`. CPU checks remain automatic. "
    "`/ci run all` and `/ci run nightly` are not supported."
)


def command_line(body):
    # Markdown 缩进代码块和多行内容不进入命令路径。
    body = body.strip("\r\n")
    if body.expandtabs(4).startswith("    ") or "\n" in body or "\r" in body:
        return ""
    return body.strip()


def parse_command(body):
    body = command_line(body)
    return body if body in COMMANDS else None


def classify(event):
    if (
        event.get("action") != "created"
        or "pull_request" not in event.get("issue", {})
        or event.get("comment", {}).get("user", {}).get("type") != "User"
    ):
        return "ignore"
    body = command_line(event["comment"]["body"])
    if parse_command(body):
        return "command"
    return "help" if body == "/ci" or body.startswith("/ci ") else "ignore"


def same_comment(current, original):
    return all(
        current[key] == original[key]
        for key in ("id", "issue_url", "body", "created_at", "updated_at")
    ) and all(
        current["user"][key] == original["user"][key] for key in ("id", "login", "type")
    )


def help_command(event, api):
    if classify(event) != "help":
        return
    comment = api.get(f"issues/comments/{number(event['comment']['id'])}")
    pr_number = number(event["issue"]["number"])
    if (
        same_comment(comment, event["comment"])
        and comment["issue_url"]
        == f"https://api.github.com/repos/{api.repo}/issues/{pr_number}"
    ):
        reply(api, pr_number, "Unsupported CI command. " + HELP)


def trusted(api, actor, configured):
    users = {u.lower() for u in re.split(r"[\s,]+", configured) if u}
    if actor.lower() in users:
        return True
    permission = api.get(f"collaborators/{quote(actor, safe='')}/permission")[
        "permission"
    ]
    return permission in {"write", "maintain", "admin"}


def authorize(api, pr, actor, configured):
    require(
        pr["state"] == "open" and pr["base"]["repo"]["full_name"] == api.repo,
        "CI commands require an open PR in this repository.",
    )
    if trusted(api, actor, configured):
        return
    require(
        actor.lower() == pr["user"]["login"].lower() and not pr["draft"],
        "Only the authorized PR author or a maintainer can request PPU CI.",
    )
    if READY.intersection(label["name"] for label in pr["labels"]):
        return
    require(
        api.review_decision(pr["number"]) == "APPROVED",
        "Ask a maintainer for a ready label or a trusted approval first.",
    )
    latest = {}
    for review in api.pages(f"pulls/{pr['number']}/reviews"):
        if review["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}:
            latest[review["user"]["login"]] = review["state"]
    require(
        any(
            state == "APPROVED" and trusted(api, user, configured)
            for user, state in latest.items()
        ),
        "A current trusted approval is required.",
    )


def reply(api, pr_number, message):
    api.post(f"issues/{number(pr_number)}/comments", {"body": message})


def requests_for_head(api, pr):
    result = []
    for check in checks(api, pr["head"]["sha"]):
        if (
            check.get("name") == REQUEST_NAME
            and check.get("app", {}).get("id") == ACTIONS_APP_ID
        ):
            record = request_record(api, check)
            if record["pr_number"] == pr["number"]:
                result.append((check, record))
    return sorted(
        result,
        key=lambda item: (
            item[1]["controller_run_id"],
            item[1]["comment_id"],
            item[0]["id"],
        ),
        reverse=True,
    )


def save_record(api, check, record, **updates):
    return api.patch(
        f"check-runs/{check['id']}",
        {
            "output": {
                "title": REQUEST_NAME,
                "summary": json.dumps(record, sort_keys=True),
            },
            **updates,
        },
    )


def stable_pr(api, pr, actor, context):
    current = api.get(f"pulls/{pr['number']}")
    require(
        all(
            all(current[side][key] == pr[side][key] for key in ("sha", "ref"))
            and all(
                current[side]["repo"][key] == pr[side]["repo"][key]
                for key in ("id", "full_name")
            )
            for side in ("head", "base")
        ),
        "PR head or base changed. Please send the command again.",
    )
    authorize(api, current, actor, context["trusted_users"])
    return current


def require_no_automatic(api, pr):
    for run in api.pages(
        "actions/workflows/build-ppu-wheels.yaml/runs", "workflow_runs"
    ):
        active = (
            run["event"] == "pull_request"
            and run["status"] != "completed"
            and run["repository"]["full_name"] == api.repo
            and any(
                p["number"] == pr["number"] and p["head"]["sha"] == pr["head"]["sha"]
                for p in run.get("pull_requests", [])
            )
        )
        require(
            not active, f"Automatic PPU CI is already active: {run_url(api, run['id'])}"
        )


def run_command(api, pr, comment, context, command):
    existing = requests_for_head(api, pr)
    for check, record in existing:
        if record["comment_id"] == comment["id"]:
            reply(
                api,
                pr["number"],
                f"This request was already recorded: {check['details_url']}",
            )
            return
        runs = matching_runs(api, check, record)
        if check["status"] != "completed" or any(
            r["status"] != "completed" for r in runs
        ):
            link = run_url(api, runs[0]["id"]) if runs else check["details_url"]
            reply(api, pr["number"], f"PPU CI is already queued or active: {link}")
            return
    require_no_automatic(api, pr)
    behind = behind_base(api, pr)
    require(
        behind == 0 or command == "/ci run --allow-stale",
        "The PR is behind its base branch. Update it or explicitly use `/ci run --allow-stale`.",
    )
    stable_pr(api, pr, comment["user"]["login"], context)
    create_request(api, pr, comment, context, command, behind=behind)


def create_request(api, pr, comment, context, command, *, behind=0, retry=None):
    record = {
        "schema": 2,
        "pr_identity": pr_identity(pr),
        "allow_stale": command == "/ci run --allow-stale",
        "dispatch_state": "preparing",
        "repository": api.repo,
        "pr_number": pr["number"],
        "head_sha": sha(pr["head"]["sha"]),
        "base_sha": sha(pr["base"]["sha"]),
        "control_sha": sha(context["control_sha"]),
        "default_branch": context["default_branch"],
        "controller_run_id": number(context["run_id"]),
        "comment_id": number(comment["id"]),
        "actor": comment["user"]["login"],
        "command": command,
    }
    if retry:
        record.update(retry)
    try:
        check = api.post(
            "check-runs",
            {
                "name": REQUEST_NAME,
                "head_sha": record["head_sha"],
                "status": "queued",
                "external_id": f"ppu-ci-request:{record['controller_run_id']}:{record['comment_id']}",
                "details_url": run_url(api, record["controller_run_id"]),
                "output": {
                    "title": REQUEST_NAME,
                    "summary": json.dumps(record, sort_keys=True),
                },
            },
        )
    except OSError:
        # 请求写入可能成功，但此时确定尚未调用 dispatch；只恢复本次幂等身份。
        for saved_check, saved in requests_for_head(api, pr):
            if (
                saved["controller_run_id"] == record["controller_run_id"]
                and saved["comment_id"] == record["comment_id"]
            ):
                saved["dispatch_state"] = "rejected"
                save_record(
                    api, saved_check, saved, status="completed", conclusion="failure"
                )
        reply(
            api,
            pr["number"],
            "Request preparation failed before dispatch; post a new command after resolving the API failure.",
        )
        return
    sent = False
    try:
        for name in CONTEXTS:
            api.post(
                "check-runs",
                {
                    "name": name,
                    "head_sha": record["head_sha"],
                    "status": "queued",
                    "external_id": f"ppu-ci:{check['id']}:{name}",
                    "details_url": check["details_url"],
                },
            )
        record["dispatch_state"] = "uncertain"
        save_record(api, check, record)
        sent = True
        api.post(
            f"actions/workflows/{EXECUTOR}/dispatches",
            {
                "ref": context["default_branch"],
                "inputs": {
                    "pr_number": str(pr["number"]),
                    "head_sha": record["head_sha"],
                    "base_sha": record["base_sha"],
                    "request_check_id": str(check["id"]),
                },
            },
        )
    except OSError as error:
        rejected = not sent or (
            isinstance(error, HTTPError)
            and error.code in {400, 401, 403, 404, 409, 422, 429}
        )
        if rejected:
            record["dispatch_state"] = "rejected"
            save_record(api, check, record, status="completed", conclusion="failure")
            for status in checks(api, record["head_sha"]):
                if (
                    status.get("external_id")
                    == f"ppu-ci:{check['id']}:{status['name']}"
                    and status.get("app", {}).get("id") == ACTIONS_APP_ID
                ):
                    api.patch(
                        f"check-runs/{status['id']}",
                        {"status": "completed", "conclusion": "failure"},
                    )
            reply(
                api,
                pr["number"],
                "Dispatch was not accepted; post a new command after resolving the API failure.",
            )
            return
        try:
            runs = matching_runs(api, check, record)
        except OSError:
            runs = []
        if len(runs) == 1:
            record["dispatch_state"] = "accepted"
            save_record(api, check, record)
            reply(
                api,
                pr["number"],
                f"Dispatch confirmed after a lost response: {run_url(api, runs[0]['id'])}",
            )
            return
        reply(
            api,
            pr["number"],
            "Dispatch acceptance is uncertain; checks remain queued. A maintainer must inspect and confirm whether it was received before cancelling and posting `/ci run`. "
            + check["details_url"],
        )
        return
    record["dispatch_state"] = "accepted"
    save_record(api, check, record)
    warning = (
        f" Warning: this head is {behind} commits behind its base; merge protection is unchanged."
        if behind
        else ""
    )
    reply(
        api,
        pr["number"],
        f"PPU CI queued for `{record['head_sha']}`.{warning} {check['details_url']}",
    )


def requests_for_pr(api, pr):
    # 回执在任何请求写入前记录 head；它只提供查找线索，不授予取消权限。
    heads = {sha(pr["head"]["sha"])}
    for receipt in api.pages(f"issues/{number(pr['number'])}/comments"):
        if receipt.get("user", {}).get("login") != "github-actions[bot]":
            continue
        match = re.match(
            r"<!-- ppu-ci-command:[1-9][0-9]* -->\n"
            r"<!-- ppu-ci-head:([0-9a-f]{40}) -->\n",
            receipt.get("body", ""),
        )
        if match:
            heads.add(match[1])
    return {
        check["id"]: (check, record)
        for head in sorted(heads)
        for check, record in requests_for_head(
            api, {"number": pr["number"], "head": {"sha": head}}
        )
    }


def cancel_pr(api, pr, comment, context):
    requests = requests_for_pr(api, pr)
    for run in api.pages(f"actions/workflows/{EXECUTOR}/runs", "workflow_runs"):
        title = TITLE.fullmatch(run.get("display_title", ""))
        if title and int(title[1]) == pr["number"] and run["status"] != "completed":
            check = api.get(f"check-runs/{number(title[3])}")
            record = request_record(api, check)
            execution_identity(api, run, check, record)
            requests[check["id"]] = check, record
    stable_pr(api, pr, comment["user"]["login"], context)
    require(requests, "No on-demand PPU request exists for this PR.")
    cancelled_any = False
    for check, record in requests.values():
        runs = matching_runs(api, check, record)
        if check["status"] == "completed" and all(
            r["status"] == "completed" for r in runs
        ):
            # A cancelled request may still have partially written bound checks.
            if not record.get("cancelled") or all(
                status["status"] == "completed"
                and status.get("conclusion") == "cancelled"
                for status in bound_checks(api, check, record)
            ):
                continue
        # The event receipt, not a partial request write, prevents comment replay.
        cancelled_any = True
        record["cancelled"] = True
        save_record(api, check, record, status="completed", conclusion="cancelled")
        for status in bound_checks(api, check, record):
            api.patch(
                f"check-runs/{status['id']}",
                {"status": "completed", "conclusion": "cancelled"},
            )
        for run in runs:
            if run["status"] != "completed":
                api.post(f"actions/runs/{run['id']}/cancel")
    require(cancelled_any, "No active on-demand PPU request exists for this PR.")
    reply(api, pr["number"], "Cancellation requested for this PR's on-demand PPU CI.")


def retry_or_cancel(api, pr, comment, context, command):
    if command == "/ci cancel":
        cancel_pr(api, pr, comment, context)
        return
    require_no_automatic(api, pr)
    requests = requests_for_head(api, pr)
    require(requests, "No PPU execution exists for this head. Use `/ci run` first.")
    if any(r["comment_id"] == comment["id"] for _, r in requests):
        return
    source = None
    for check, record in requests:
        runs = matching_runs(api, check, record)
        if not runs and (
            record.get("dispatch_state") == "rejected"
            or (source is not None and record.get("cancelled") is True)
        ):
            continue
        require(
            runs,
            "Dispatch is pending or uncertain; inspect and confirm its status before `/ci cancel` and a new `/ci run`.",
        )
        require(
            all(r["status"] == "completed" for r in runs), "PPU CI is already active."
        )
        if source is None:
            source = check, record, runs[0]
    require(source, "No completed execution exists; use `/ci run`.")
    check, record, run = source
    require(
        pr_identity(pr) == record["pr_identity"]
        and context["control_sha"] == record["control_sha"]
        and context["default_branch"] == record["default_branch"],
        "PR head, base or control revision changed; use `/ci run`.",
    )
    require(
        run["conclusion"] != "success" or check.get("conclusion") != "success",
        "No completed failed execution to retry.",
    )
    plan = retry_plan(api, check, record, run)
    require(
        record["allow_stale"] or behind_base(api, pr) == 0,
        "The PR is behind its base; use `/ci run`.",
    )
    stable_pr(api, pr, comment["user"]["login"], context)
    create_request(
        api,
        pr,
        comment,
        context,
        "/ci retry",
        retry={
            "retry_of_request_id": check["id"],
            "retry_of_run_id": run["id"],
            "retry_plan": plan,
            "allow_stale": record["allow_stale"],
        },
    )


def handle_event(event, api, context):
    if classify(event) != "command":
        return
    command = parse_command(event["comment"]["body"])
    pr_number = number(event["issue"]["number"])
    try:
        comment = api.get(f"issues/comments/{number(event['comment']['id'])}")
        require(
            same_comment(comment, event["comment"]),
            "The comment changed; post a new command.",
        )
        require(
            comment["issue_url"]
            == f"https://api.github.com/repos/{api.repo}/issues/{pr_number}"
            and comment["user"]["type"] == "User",
            "Invalid command source.",
        )
        pr = api.get(f"pulls/{pr_number}")
        authorize(api, pr, comment["user"]["login"], context["trusted_users"])
        marker = f"<!-- ppu-ci-command:{comment['id']} -->"
        if any(
            c.get("user", {}).get("login") == "github-actions[bot]"
            and marker in c["body"]
            for c in api.pages(f"issues/{pr_number}/comments")
        ):
            reply(
                api,
                pr_number,
                "This comment has already been processed. Post a new command if needed.",
            )
            return
        # Reserve a receipt before dispatch; it prevents replay, never grants access.
        reply(
            api,
            pr_number,
            marker
            + f"\n<!-- ppu-ci-head:{sha(pr['head']['sha'])} -->"
            + f"\nProcessing `{command}`: {run_url(api, context['run_id'])}",
        )
        if command.startswith("/ci run"):
            run_command(api, pr, comment, context, command)
        else:
            retry_or_cancel(api, pr, comment, context, command)
    except Rejected as error:
        reply(api, pr_number, str(error))


def validate_execution(api, inputs, run, context):
    check = api.get(f"check-runs/{number(inputs['request_check_id'])}")
    record = request_record(api, check)
    require(
        set(inputs) == {"pr_number", "head_sha", "base_sha", "request_check_id"},
        "Unexpected dispatch inputs.",
    )
    require(
        all(str(record[k]) == inputs[k] for k in ("pr_number", "head_sha", "base_sha")),
        "Dispatch input binding mismatch.",
    )
    execution_identity(api, run, check, record)
    require(
        context["control_sha"] == record["control_sha"]
        and context["default_branch"] == record["default_branch"],
        "Control revision changed.",
    )
    unique_execution(api, check, record, run)
    require(context["run_id"] == run["id"], "Validator run identity mismatch.")
    require(not record.get("cancelled"), "This request was cancelled.")
    require(
        record.get("dispatch_state") in {"accepted", "uncertain"},
        "Dispatch was not accepted.",
    )
    pr = api.get(f"pulls/{record['pr_number']}")
    require(
        pr_identity(pr) == record["pr_identity"],
        "PR head or base changed; send `/ci run` again.",
    )
    latest = requests_for_head(api, pr)
    require(
        latest and latest[0][0]["id"] == check["id"],
        "This request has been superseded.",
    )
    actor = record["actor"]
    authorize(api, pr, actor, context["trusted_users"])
    require(
        record["allow_stale"] or behind_base(api, pr) == 0,
        "The PR is now behind its base branch.",
    )
    plan = request_plan(api, check, record)
    require_no_automatic(api, pr)
    stable_pr(api, pr, actor, context)
    bound_checks(api, check, record)
    record["run_id"] = run["id"]
    record["validation"] = {"run_id": run["id"], "run_attempt": 1, "plan": plan}
    save_record(api, check, record, status="in_progress")
    return record


def notify(event, api, context):
    if (
        event.get("action") != "labeled"
        or event.get("label", {}).get("name") not in READY
    ):
        return
    pr = api.get(f"pulls/{number(event['pull_request']['number'])}")
    if (
        pr["state"] != "open"
        or pr["draft"]
        or not READY.intersection(x["name"] for x in pr["labels"])
    ):
        return
    marker = "<!-- ppu-ci-authorized -->"
    if any(
        c.get("user", {}).get("login") == "github-actions[bot]" and marker in c["body"]
        for c in api.pages(f"issues/{pr['number']}/comments")
    ):
        return
    reply(api, pr["number"], marker + "\n" + HELP)


def main():
    with open(os.environ["GITHUB_EVENT_PATH"]) as stream:
        event = json.load(stream)
    mode = sys.argv[1]
    if mode == "classify":
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(f"kind={classify(event)}\n")
        return
    api = GitHub(os.environ["GITHUB_REPOSITORY"], os.environ["GH_TOKEN"])
    if mode == "help":
        help_command(event, api)
        return
    context = context_from_env(api)
    if mode == "command":
        handle_event(event, api, context)
    elif mode == "notify":
        notify(event, api, context)
    elif mode == "validate":
        record = validate_execution(
            api, event["inputs"], api.get(f"actions/runs/{context['run_id']}"), context
        )
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            stream.write(
                f"head_sha={record['head_sha']}\npr_number={record['pr_number']}\n"
                f"execution_plan={json.dumps(record['validation'], separators=(',', ':'))}\n"
            )
    else:
        raise Rejected("Unknown controller mode.")


if __name__ == "__main__":
    main()
