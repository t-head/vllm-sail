# Verification Guide

Validate the same package installation, SDK and hardware that will run the
workload. Keep the results of each check separate:

| Check | What it establishes |
| --- | --- |
| CPU unit tests | Python behavior, registration contracts and packaging invariants |
| Source regeneration checks | Generated upstream kernels match their declared inputs |
| HGGC build | Selected sources compile and link for the requested targets |
| Extension imports and strict preflight | Libraries load and representative operations are registered |
| Device correctness tests | Tested kernels match their numerical references on that device |
| Model run | The tested checkpoint and configuration complete inference |
| Benchmark | Performance under the recorded workload and environment |

Passing one check does not imply the later checks pass. A skipped test provides
no execution evidence for that feature.

## 1. Record the environment

After following [Installation](../getting_started/installation.md), run:

```bash
git rev-parse HEAD
python -c "from vllm_sail import collect_env; collect_env()"
python -c "import vllm_sail; print(vllm_sail.__file__)"
```

Record the PPU model and count, SDK version, SAIL PyTorch version, vLLM revision
and plugin revision. Confirm the printed package path points to the installation
being tested. A source checkout can shadow a separately installed native wheel.

Run the four-library import and strict preflight commands in
[Check the installation](../getting_started/installation.md#4-check-the-installation).
Resolve shared-library errors before running a model.

## 2. Run unit tests

From the plugin checkout, using its test environment:

```bash
python -m pytest tests/ut -q
```

The merge gate runs with only `requirements/dev.txt` installed, without torch,
vLLM or a device. Additional integration tests become available in a PPU
environment with their dependencies installed. Investigate failures and report
skip reasons; do not compare test counts from different environments as if they
cover the same paths.

## 3. Run device correctness tests

In the PPU environment, install `requirements/dev.txt`. Use an editable native
plugin build in the checkout, or ensure the test process imports the installed
wheel as described above. Start with activation and the affected feature:

```bash
pytest tests/e2e/test_native_activation.py -q -rs
pytest tests/e2e/test_deepseek_v4_native.py -q -rs
pytest tests/e2e/test_mxfp4_prepare.py -q -rs
pytest tests/e2e/tier_b/test_norm_quant.py -q -rs
```

The [`tests/e2e`](../../tests/e2e) directory also contains linear-attention,
indexer, cache and model-specific kernel tests. Select the relevant tests for
the change; run the full suite when qualifying an environment:

```bash
pytest tests/e2e -q -rs
```

These tests require a real PPU and the applicable SDK libraries. Verify that the
intended tests actually ran. Include dtype, shape, tolerance and device in a
correctness report, and test both architecture targets when claiming both.

## 4. Run a model

Follow the [quickstart](../getting_started/quickstart.md) with a checkpoint
appropriate for the available memory and supported precision. Confirm that:

- The PPU platform and intended compute backend are selected.
- Model loading and warm-up complete.
- The readiness endpoint succeeds and generation returns sensible output.
- Repeated generation and the intended context lengths complete successfully.

For MoE, quantized or multi-device workloads, repeat the check using the intended
expert backend, checkpoint layout and parallelism. A small dense-model smoke
test does not cover those paths.

## 5. Report correctness and performance

Include the exact command, checkpoint identifier and revision, dtype,
quantization, context length, device count and parallelism. For benchmarks, also
include the workload or dataset, input/output lengths, concurrency, warm-up,
throughput and latency measurements.

Useful tools include vLLM's `vllm bench --help` and the repository's
[`benchmarks/kernels`](../../benchmarks/kernels). Compare baseline and candidate
with the same software, hardware and workload. Explain the numerical reference
and tolerance used for kernel comparisons.

Report which checks passed, failed, skipped or were not run. Attach concise
logs and a minimal reproducer through the repository's issue or pull request
template; remove credentials and private model or host details before posting.
