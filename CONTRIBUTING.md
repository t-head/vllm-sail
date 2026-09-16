# Contributing to vLLM SAIL

Thank you for contributing. Documentation, bug reports, tests, PPU kernel
implementations and performance improvements all help make the project useful
to more developers.

## Find a place to start

- Improve an installation step or add a minimal reproducer for an issue.
- Add CPU coverage for plugin registration, configuration or packaging.
- Validate a kernel on PPU and report the exact environment and numerical result.
- Tune a workload or implement a missing backend capability.

Check existing issues and pull requests before starting. For a substantial API,
dependency or architecture change, open an issue describing the problem and
proposed design so maintainers can discuss the scope before implementation.

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

Run the CPU merge gate:

```bash
python -m pytest tests/ut -q
```

Run the repository's pinned formatting and hygiene hooks on changed files:

```bash
pre-commit run --files path/to/changed_file.py
```

The pre-commit configuration pins Ruff. Restrict unrelated formatting changes
to a separate pull request. Add tests for meaningful behavior changes; keep
test dependencies within the existing CPU or device tier.

For native or runtime changes, report the additional commands in
[Verification](docs/user_guide/verification_guide.md). Say explicitly which
device checks you could not run. Documentation changes should have working
relative links and commands consistent with the current code.

## Pull request workflow

1. Fork the repository and create a topic branch for the change.
2. Implement the change and update the relevant tests and documentation.
3. Run the applicable checks and inspect the diff.
4. Push the topic branch to your fork and open a pull request.
5. Address review feedback; merge through the platform after review.

Use a descriptive title. In the description, explain the concrete problem, the
resulting behavior and how it was validated. Link a related issue when there is
one. For performance changes, include a reproducible baseline and the same
workload on the candidate. Avoid force pushes and direct pushes to shared
branches; follow-up commits keep review history available.

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
