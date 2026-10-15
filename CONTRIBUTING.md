# Contributing to vLLM SAIL

Contributions to documentation, tests, PPU kernels and performance are welcome.
Check existing issues and pull requests before starting. Discuss substantial
API, dependency or architecture changes in an issue first.

## Development environment

The CPU test suite runs with Python 3.10–3.13 and does not require torch, vLLM,
the SAIL SDK or a device. From a checkout of this repository:

```bash
python3 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements/dev.txt
python -m pytest tests/ut -q
```

The package is imported directly from the checkout for these tests. The expected
`vllm_sail._version is missing` warning can appear before the package is installed.
Optional integration tests skip when their dependencies are unavailable.
`requirements/dev.txt` installs a fixed Node.js binary wheel for testing the
JavaScript guards embedded in the CI workflows. Tests invoke it through the
current Python environment (`python -m nodejs_wheel`), so neither local nor PPU
worker tests require a system `node` executable. The pin UT worker checks this
runtime before running pytest; missing runtime dependencies fail rather than
skip these tests.

For native compilation or device testing, use the separate
[PPU installation instructions](docs/getting_started/installation.md).

## Coding guidelines

- Keep changes focused and describe the behavior they change.
- Prefer explicit interfaces and shared helpers when they remove real duplication.
- Preserve lazy imports and entry-point idempotency. Core imports and CPU tests
  must remain independent of optional SDK packages.
- Use the existing registration APIs and platform hooks before adding a patch.
  Read the [runtime patch guide](vllm_sail/patch/README.md) for metadata and copied
  source rules.
- Edit plugin-owned HGGC kernels directly in `csrc/plugin/`. For upstream
  kernels, edit generation inputs and regenerate; follow
  [HGGC kernel development](docs/developer_guide/kernels.md).
- Preserve upstream attribution and license headers. New Python files should
  use `# SPDX-License-Identifier: Apache-2.0`.
- Keep user documentation focused on reproducible PPU workflows. Personal
  run logs, implementation phase records and private environment notes belong
  outside the repository.

See [Architecture](docs/developer_guide/architecture.md) for module boundaries.

## Checks before review

Run the CPU tests and formatting checks:

```bash
python -m pytest tests/ut -q
pre-commit run --files path/to/changed_file.py
```

For staged files or a full repository check:

```bash
pre-commit run
pre-commit run --all-files --hook-stage manual
```

Add tests for behavior changes and keep formatting changes focused. For native
or runtime changes, follow [Verification](docs/user_guide/verification_guide.md)
and report any checks that were not run. Verify links and commands in docs.

## Pull request workflow

1. Fork the repository and create a topic branch for the change.
2. Implement the change and update the relevant tests and documentation.
3. Run the applicable checks and inspect the diff.
4. Push the topic branch to your fork and open a pull request.
5. Address review feedback; merge through the platform after review.

Use vLLM-style PR titles: `[Type][Scope] Concise behavior-oriented summary`,
with an optional scope. For example, `[Bugfix][MoE] Honor explicit backend
selection` or `[Doc] Clarify native build prerequisites`.

Complete the [PR template](.github/pull_request_template.md), link related issues
and include test commands and results. Performance changes need baseline and
candidate measurements under the same workload. Avoid force pushes and direct
pushes to shared branches; use follow-up commits for review changes.

## Requesting PPU CI

CPU unit tests, upstream source contracts and repository security checks remain
automatic. The comment interface controls only the PPU wheel build, pin unit
tests, Qwen smoke test and PPU gate. During rollout, the existing automatic PPU
PR workflow remains enabled until the on-demand path has been verified on the
default branch. Commands reuse an active automatic run instead of duplicating it.

Post one of these commands as the entire text of a **new PR comment**:

| Command | Behavior |
| --- | --- |
| `/ci run` | Build and test the current PR head using the existing PPU PR chain. |
| `/ci run --allow-stale` | Run even if the head is behind its actual base branch; merge protection is unchanged. |
| `/ci retry` | Create a new validated run for failed stages of the latest completed on-demand execution. Reuse proven successful stages only. |
| `/ci cancel` | Cancel this PR's on-demand PPU requests, including older heads; CPU checks and other PRs are unaffected. |

An ordinary author needs an open, non-Draft PR and either a `ready` or
`ready-run-all-tests` label, or a current trusted approval with the PR's overall
review decision `APPROVED`. Historical or dismissed approvals do not authorize a
run. Maintainers with `write`, `maintain` or `admin` permission, and users listed
in the optional `CI_TRUSTED_USERS` repository variable (comma- or whitespace-
separated GitHub logins), can run open PRs directly, including Drafts. Permissions
are checked when processing a command and again in a GitHub-hosted validator
before every new run, including retries. `author_association` is not sufficient.

Adding a ready label only posts help; it does not launch tests. Approval is
checked when a command is processed. `/ci run all`, `/ci run nightly`, edited
comments, code blocks and arbitrary arguments are not supported. Manual operator
qualification and historical-wheel debugging remain separate and do not provide
PR gate evidence. Do not dispatch `PPU PR CI` directly: it requires a trusted
request created by the comment controller.

Each request binds the repository, PR number, head and base SHAs, and the trusted
control workflow revision. Checks belong to that head only. A new commit needs a
new `/ci run`; `/ci retry` never transfers failures or success from an old head.
Once automatic PPU triggering is retired, a newly pushed head has missing /
**Expected** PPU checks until explicitly tested. An old green result does not
satisfy the new head's gate. Branch synchronization, approvals and required checks
are never bypassed. The rollout still retains automatic PPU triggering for now.

Retries use a new request and workflow run, not GitHub's native **Re-run**.
Native reruns of `PPU PR CI` are rejected at the validator and actual job entry
points; use `/ci retry`. Automatic PR and independent manual workflows keep their
existing rerun behavior. Retries require unchanged head, base and control SHAs;
if any changed, use `/ci run`. An original explicit `--allow-stale` authorization
is preserved but does not relax any identity or permission check.

If the build did not succeed, retry rebuilds and reruns both tests. Otherwise it
reruns only unsuccessful pin UT / smoke stages. If only gate or reporting failed,
all three heavy stages can be reused while validation and gate run again. Reuse
requires real successful leaf jobs and a unique, unexpired build artifact bound
to its original run, artifact ID and digest. Missing or changed artifacts require
a fresh `/ci run`. Downloads use that exact ID and still verify the wheel manifest
and file hashes. Reused check summaries link to the original run and job; they
are not presented as new device measurements. Arbitrary skipped, missing,
cancelled or timed-out required jobs cannot pass the gate.

Command, validation and reporting writes share a serialized PR queue with
`queue: max` and `cancel-in-progress: false`. At most 100 pending jobs can wait;
additional jobs may be cancelled by GitHub. Ordinary comments and bot replies do
not enter this queue. Unsupported-command help and label notifications use
separate paths. A duplicate or active request returns its link.

If dispatch acceptance is uncertain, checks remain queued: ask a maintainer to
inspect the unique request/run title instead of repeatedly dispatching. A
clearly rejected new request is marked failed without changing its source;
a new comment can retry that source after the API problem is resolved. Lost
responses are matched to a unique new run, never resent automatically. After a
maintainer confirms that no run was accepted, `/ci cancel` invalidates the
pending request and a new `/ci run` can start fresh. Cancellation invalidates the
request before asking Actions to stop it. The controller records a processing
receipt before creating any request. Its head index lets cancellation find older
heads even when an accepted or uncertain dispatch has no visible executor run.
Keep these bot receipts while requests are pending; they are lookup hints, not
authorization, and every discovered request is still verified against its PR.

`PPU CI report` also reconciles unfinished terminal runs every ten minutes.
Discovery has an 80-read API budget, including checkpoint lookup and download,
and examines at most five on-demand candidates per invocation. It sweeps the
unfiltered executor history from newer to older run IDs in resumable batches,
then starts another sweep. New runs do not move an existing sweep back to the
beginning; deleted run IDs do not invalidate its position. Later sweeps revisit
failed or cancelled reports and check for partially written completion state.
This is not a completion-time guarantee: GitHub schedules can be delayed, and
both the discovery and per-PR queues have finite capacity.

A small `ppu-report-cursor-v1` artifact stores the next position. Sweep writers
are serialized separately from per-PR reporting. The checkpoint is a scheduling
hint only: its reporting-workflow provenance is verified, its JSON is never
executed, and it cannot authorize checks or PPU work. Missing or expired state
restarts a sweep with a warning; transient API failures preserve the previous
checkpoint. Failed uploads may repeat a batch. Event callbacks and explicit
`run_id` repairs do not advance this checkpoint or wait behind sweep writers.
Each repair reacquires the same PR lock and rechecks live request, run and
evidence metadata without the discovery cache. It never launches PPU work or
resends uncertain dispatches. Old heads remain isolated, superseded requests
cannot overwrite newer requests, and cancellation cannot become green.
For API failures or interrupted reporting, use **Run workflow** on `PPU CI report`
on the default branch, optionally supplying the executor `run_id` (blank resumes
the sweep). This reporting-only entry point does not rerun tests.

### Maintainer deployment checks

The comment, dispatch and reporting workflows must be present on the default
branch. Create the ready labels before advertising them. Verify that organization
policy permits the controller's `actions: write`, `checks: write` and
`issues: write`; a policy API 403 is not proof that workflow tokens have these
permissions. If policy blocks them, ask the organization maintainer to resolve it;
do not substitute a personal token or weaken branch protection.

PPU workflows explicitly use GitHub's `cache-mode: none`, propagated through the
reusable chain and enforced by scoped cache tokens. Before checking out PR code,
jobs verify the effective `ACTIONS_CACHE_MODE` inside a Node Action: the runner
injects this variable into Node Actions, not ordinary shell steps. Never set it
to a constant to satisfy the guard. This prevents default-branch cache writes
from dispatch-based PR execution. Test jobs have read-only repository and
Actions permissions, no inherited secrets and no persisted checkout credentials.

Before retiring the automatic PPU trigger in a separate PR, verify on GitHub:
authorization and notification without scheduling, fork PR execution, the exact
head SHA and all four checks, retry with reused prerequisites, cancellation and
cache enforcement. Local CPU tests cannot establish platform or device success.
After the switch, also verify that a new push does not start PPU work and that its
head cannot reuse the old green checks. Keep all existing required checks.

For workflow linting, actionlint 1.7.12 does not yet recognize the official
`cache-mode` and `concurrency.queue` keys. Compatibility exceptions are restricted
to `unexpected key "cache-mode" for "workflow" section` and
`unexpected key "queue" for "concurrency" section`. Pipeline contract tests must
still verify cache isolation, `queue: max`, shared lock identity and
`cancel-in-progress: false`. GitHub forbids combining `queue: max` with
`cancel-in-progress: true`; the automatic PR cancellation behavior is unchanged.
See [GitHub concurrency syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#concurrency).
Do not ignore other syntax errors or security findings. Existing self-hosted
runner-label warnings may be configured separately. Verify actual platform
acceptance and queue recovery before retiring the automatic trigger.

## Reporting issues

Use the bug or feature request template in the repository's Issues tab. Bug
reports should include a minimal reproducer, expected and actual results, the
full relevant error and `collect_env` output. Remove credentials, private host
addresses and confidential model paths from logs before posting.

Keep discussions respectful and focused on the work. Explain technical
disagreements with examples, measurements or source evidence.

## License

Contributions are made under the project's [Apache License 2.0](LICENSE).
Retain the applicable notices when adapting third-party code.
