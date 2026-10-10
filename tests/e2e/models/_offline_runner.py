# SPDX-License-Identifier: Apache-2.0
"""Shared offline-model E2E runner for the PPU model-precision suite.

This helper backs the ``tests/e2e/models`` category (plan PART 5). Every model
test drives vLLM's offline :class:`~vllm.LLM` directly — never a server, never
an OpenAI client — and compares greedy generation against a per-chip golden
baseline captured on real PPU hardware.

Design contract
---------------
* **Offline only.** :func:`build_llm` constructs ``vllm.LLM`` and
  :func:`run_inference` calls ``LLM.generate``. No HTTP, no ``serve``.
* **One runtime input.** The model weight path is the only value that may vary
  at run time (see :func:`resolve_checkpoint` / ``VLLM_SAIL_MODEL_ROOT``). Every
  other knob — tensor parallelism, ``max_model_len``, ``dtype``, sampling — is
  fixed here so runs are bit-for-bit reproducible.
* **TP follows the chip.** ``tensor_parallel_size`` is read from the detected
  :class:`~tests.conftest.ChipProfile` (ZW-810E -> 16, ZW-890P -> 8).
* **Per-chip golden.** 810E (int8 / w8a8-int8) and 890P (mxfp4 / w8a8-fp8) take
  different quantization paths, so baselines are archived per chip and never
  cross-compared.

Importing this module must never require ``torch`` or ``vllm``: both are
imported lazily inside the functions that need them, so collection stays
CPU-only and the guard chain in each test file decides whether to skip.

Extension point (Weeks 5-8)
---------------------------
Adding DeepSeek V4 / MiniMax M3 / Kimi K3 is a *one-file* change. Create
``test_model_<name>.py`` that:

1. declares a per-model fixed config as a module constant::

       CONFIG = ModelConfig(
           key="kimi_k3",
           checkpoint="/nas_aisw/.../Kimi-K3",
           env={"VLLM_SAIL_USE_PLA": "1"},          # chip-independent knobs
           extra_llm_kwargs={"speculative_config": None},
       )

2. runs the shared flow (build -> generate -> compare)::

       llm = offline_llm(resolve_checkpoint(CONFIG.checkpoint), CONFIG)
       result = run_inference(llm, PROMPTS, CONFIG)
       compare_golden(result, golden_store.path(f"{CONFIG.key}_{chip}.json"),
                      chip, update=golden_store.update)

``ModelConfig`` carries every per-model override; the defaults below already
match the plan's fixed-parameter table, so a model that needs nothing special
passes just ``key`` and ``checkpoint``.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Fixed parameters (plan PART 5, table 5.1). Not overridable at runtime. ---
SCHEMA_VERSION = 1
DEFAULT_MAX_MODEL_LEN = 4096
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_ENFORCE_EAGER = True
DEFAULT_SEED = 0
DEFAULT_TRUST_REMOTE_CODE = True

# Greedy sampling: deterministic decode, top-k logprobs captured for compare.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 64
DEFAULT_LOGPROBS = 5
DEFAULT_PROMPT_LOGPROBS = 0

# Golden logprob tolerance. Greedy decode is deterministic, so the tolerance is
# tight: a value passes when |actual - expected| <= atol + rtol * |expected|.
GOLDEN_LOGPROB_ATOL = 1e-3
GOLDEN_LOGPROB_RTOL = 1e-2

#: Environment variable that may override the fixed checkpoint path at runtime.
MODEL_ROOT_ENV = "VLLM_SAIL_MODEL_ROOT"

# Capability -> chip key. Mirrors tests/conftest.py CHIP_PROFILES so golden
# files can be named without importing the conftest.
CAPABILITY_TO_CHIP: dict[tuple[int, int], str] = {(8, 0): "810e", (8, 9): "890p"}
DEFAULT_CHIP = "810e"

#: Fixed representative prompt set (plan: 8 prompts, Chinese + English + code +
#: long context). Shared by every model test so goldens stay comparable.
PROMPTS: tuple[str, ...] = (
    "Explain in one sentence what a single transformer attention head computes.",
    "用一句话解释什么是张量并行（tensor parallelism）。",
    "Write a Python function fib(n) that returns the nth Fibonacci number "
    "using memoization. Include a docstring.",
    "下面这段 Python 代码有什么问题？请指出并给出修复版本：\n"
    "def divide(a, b):\n    return a / b\n",
    "You are reviewing a distributed inference engine. The scheduler batches "
    "requests, the KV cache is paged into fixed-size blocks, and tensor "
    "parallelism splits each attention head across devices. Given a workload "
    "of many short prompts with a few very long ones, describe two sources of "
    "load imbalance and one mitigation for each. Keep the answer under 120 "
    "words and use a numbered list.",
    "A train travels 120 km at 60 km/h, then 180 km at 90 km/h. What is the "
    "average speed for the whole trip in km/h? Show the calculation briefly.",
    "请用中文写一首关于秋天的四行短诗，要求押韵，不要出现“秋天”两个字。",
    "Return exactly a JSON array of the three primary colors of light, "
    'lowercase, with no extra text. Example format: ["red", "green", "blue"].',
)


@dataclass(frozen=True)
class ModelConfig:
    """Per-model fixed configuration for the offline E2E flow.

    Defaults reproduce the plan's shared parameter table; a model overrides
    only what it truly needs. ``env`` is applied to ``os.environ`` before the
    ``LLM`` is built (chip-independent knobs such as ``VLLM_SAIL_USE_PLA``);
    ``extra_llm_kwargs`` is merged last into the ``LLM(...)`` call.
    """

    key: str
    checkpoint: str
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    dtype: str = DEFAULT_DTYPE
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    enforce_eager: bool = DEFAULT_ENFORCE_EAGER
    seed: int = DEFAULT_SEED
    trust_remote_code: bool = DEFAULT_TRUST_REMOTE_CODE
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    logprobs: int = DEFAULT_LOGPROBS
    prompt_logprobs: int = DEFAULT_PROMPT_LOGPROBS
    #: Per-model golden logprob tolerance. Defaults reproduce the module-wide
    #: ``GOLDEN_LOGPROB_ATOL`` / ``GOLDEN_LOGPROB_RTOL`` so a model that needs
    #: nothing special compares exactly like before; a model with known numeric
    #: drift (e.g. Kimi K3 linear attention accumulating error over long
    #: sequences) may relax them locally without touching the shared constants
    #: or any other model's golden.
    logprob_atol: float = GOLDEN_LOGPROB_ATOL
    logprob_rtol: float = GOLDEN_LOGPROB_RTOL
    env: Mapping[str, str] = field(default_factory=dict)
    extra_llm_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseResult:
    """Serializable generation result for a single prompt."""

    prompt: str
    token_ids: list[int]
    text: str
    completion_logprobs: list[dict[int, float]]
    prompt_logprobs: list[dict[int, float]]


@dataclass(frozen=True)
class InferenceResult:
    """Serializable batch result handed to :func:`compare_golden`."""

    prompts: list[str]
    cases: list[CaseResult]


def resolve_checkpoint(default: str) -> str:
    """Return the runtime weight path: ``VLLM_SAIL_MODEL_ROOT`` or ``default``."""
    return os.environ.get(MODEL_ROOT_ENV) or default


def chip_key_for_capability(capability: Any) -> str:
    """Map a device capability onto the golden chip key (``810e`` / ``890p``)."""
    if capability is None:
        return DEFAULT_CHIP
    key = (int(capability[0]), int(capability[1]))
    return CAPABILITY_TO_CHIP.get(key, DEFAULT_CHIP)


def build_llm(model_path: str, chip_profile: Any, config: ModelConfig | None = None):
    """Construct the offline ``vllm.LLM`` with every parameter fixed.

    ``tensor_parallel_size`` comes from ``chip_profile.tensor_parallel_size``
    (16 on ZW-810E, 8 on ZW-890P); all other knobs come from ``config`` or the
    module defaults. ``vllm`` is imported lazily so importing this module stays
    torch/vLLM-free.
    """
    if config is not None and config.env:
        for name, value in config.env.items():
            os.environ[name] = value

    tensor_parallel_size = int(chip_profile.tensor_parallel_size)
    kwargs: dict[str, Any] = {
        "max_model_len": config.max_model_len if config else DEFAULT_MAX_MODEL_LEN,
        "dtype": config.dtype if config else DEFAULT_DTYPE,
        "gpu_memory_utilization": (
            config.gpu_memory_utilization if config else DEFAULT_GPU_MEMORY_UTILIZATION
        ),
        "enforce_eager": config.enforce_eager if config else DEFAULT_ENFORCE_EAGER,
        "seed": config.seed if config else DEFAULT_SEED,
        "trust_remote_code": (
            config.trust_remote_code if config else DEFAULT_TRUST_REMOTE_CODE
        ),
    }
    if config is not None and config.extra_llm_kwargs:
        kwargs.update(config.extra_llm_kwargs)

    from vllm import LLM

    return LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, **kwargs)


def _sampling_params(config: ModelConfig | None):
    """Build the fixed greedy ``SamplingParams`` (imports vLLM lazily)."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=config.temperature if config else DEFAULT_TEMPERATURE,
        max_tokens=config.max_tokens if config else DEFAULT_MAX_TOKENS,
        logprobs=config.logprobs if config else DEFAULT_LOGPROBS,
        prompt_logprobs=config.prompt_logprobs if config else DEFAULT_PROMPT_LOGPROBS,
        seed=config.seed if config else DEFAULT_SEED,
    )


def _extract_logprobs(raw: Any) -> list[dict[int, float]]:
    """Convert vLLM ``list[dict[int, Logprob] | None]`` into plain JSON-safe data."""
    steps: list[dict[int, float]] = []
    if not raw:
        return steps
    for step in raw:
        if not step:
            steps.append({})
            continue
        steps.append(
            {int(token_id): float(lp.logprob) for token_id, lp in step.items()}
        )
    return steps


def run_inference(
    llm: Any, prompts: Sequence[str] = PROMPTS, config: ModelConfig | None = None
) -> InferenceResult:
    """Run greedy offline generation and capture token ids + logprobs."""
    params = _sampling_params(config)
    prompt_list = list(prompts)
    raw_outputs = llm.generate(prompt_list, params)

    cases: list[CaseResult] = []
    for index, request_output in enumerate(raw_outputs):
        completion = request_output.outputs[0]
        cases.append(
            CaseResult(
                prompt=prompt_list[index],
                token_ids=[int(token) for token in completion.token_ids],
                text=completion.text,
                completion_logprobs=_extract_logprobs(completion.logprobs),
                prompt_logprobs=_extract_logprobs(request_output.prompt_logprobs),
            )
        )
    return InferenceResult(prompts=prompt_list, cases=cases)


def _fixed_param_snapshot(
    atol: float = GOLDEN_LOGPROB_ATOL, rtol: float = GOLDEN_LOGPROB_RTOL
) -> dict[str, Any]:
    """Record the fixed knobs + tolerance inside every golden for traceability."""
    return {
        "max_model_len": DEFAULT_MAX_MODEL_LEN,
        "dtype": DEFAULT_DTYPE,
        "gpu_memory_utilization": DEFAULT_GPU_MEMORY_UTILIZATION,
        "enforce_eager": DEFAULT_ENFORCE_EAGER,
        "seed": DEFAULT_SEED,
        "trust_remote_code": DEFAULT_TRUST_REMOTE_CODE,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "logprobs": DEFAULT_LOGPROBS,
        "prompt_logprobs": DEFAULT_PROMPT_LOGPROBS,
        "tolerance": {"atol": atol, "rtol": rtol},
    }


def _model_from_path(golden_path: Path, chip: str) -> str:
    """Derive the model key from a ``<model>_<chip>.json`` golden filename."""
    stem = golden_path.stem
    suffix = f"_{chip}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _case_to_json(case: CaseResult) -> dict[str, Any]:
    return {
        "prompt": case.prompt,
        "token_ids": list(case.token_ids),
        "text": case.text,
        "completion_logprobs": [
            {str(tid): lp for tid, lp in step.items()}
            for step in case.completion_logprobs
        ],
        "prompt_logprobs": [
            {str(tid): lp for tid, lp in step.items()} for step in case.prompt_logprobs
        ],
    }


def _record_to_json(
    outputs: InferenceResult,
    golden_path: Path,
    chip: str,
    atol: float = GOLDEN_LOGPROB_ATOL,
    rtol: float = GOLDEN_LOGPROB_RTOL,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "model": _model_from_path(golden_path, chip),
        "chip": chip,
        "fixed_params": _fixed_param_snapshot(atol, rtol),
        "prompts": list(outputs.prompts),
        "cases": [_case_to_json(case) for case in outputs.cases],
    }


def _write_golden(record: Mapping[str, Any], golden_path: Path) -> None:
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def _assert_chip(expected: Mapping[str, Any], chip: str, golden_path: Path) -> None:
    golden_chip = expected.get("chip")
    if golden_chip != chip:
        raise AssertionError(
            f"golden chip mismatch: {golden_path} records chip={golden_chip!r} but the "
            f"current device is {chip!r}; per-chip goldens must never be cross-compared."
        )


def _logprob_deviation(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    atol: float = GOLDEN_LOGPROB_ATOL,
    rtol: float = GOLDEN_LOGPROB_RTOL,
) -> tuple[float, list[str]]:
    """Return (max deviation, violations) comparing two JSON logprob mappings."""
    max_dev = 0.0
    violations: list[str] = []
    for token_id, expected_lp in expected.items():
        if token_id not in actual:
            violations.append(f"missing logprob for token {token_id}")
            continue
        actual_lp = actual[token_id]
        deviation = abs(actual_lp - expected_lp)
        max_dev = max(max_dev, deviation)
        allowed = atol + rtol * abs(expected_lp)
        if deviation > allowed:
            violations.append(
                f"token {token_id}: actual={actual_lp:.6f} expected={expected_lp:.6f} "
                f"dev={deviation:.3e} > allowed={allowed:.3e}"
            )
    return max_dev, violations


def _compare_steps(
    actual_steps: Sequence[Mapping[str, Any]],
    expected_steps: Sequence[Mapping[str, Any]],
    label: str,
    case_index: int,
    atol: float = GOLDEN_LOGPROB_ATOL,
    rtol: float = GOLDEN_LOGPROB_RTOL,
) -> tuple[float, list[str]]:
    max_dev = 0.0
    problems: list[str] = []
    if len(actual_steps) != len(expected_steps):
        problems.append(
            f"case {case_index} {label}: step count {len(actual_steps)} != "
            f"golden {len(expected_steps)}"
        )
        return max_dev, problems
    for position in range(len(expected_steps)):
        deviation, violations = _logprob_deviation(
            actual_steps[position], expected_steps[position], atol, rtol
        )
        max_dev = max(max_dev, deviation)
        problems.extend(
            f"case {case_index} {label} pos {position}: {v}" for v in violations
        )
    return max_dev, problems


def _compare_cases(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    atol: float = GOLDEN_LOGPROB_ATOL,
    rtol: float = GOLDEN_LOGPROB_RTOL,
) -> float:
    """Compare generated cases against the golden; return the max logprob deviation."""
    actual_cases = record["cases"]
    expected_cases = expected.get("cases", [])
    if len(actual_cases) != len(expected_cases):
        raise AssertionError(
            f"golden case count mismatch: produced {len(actual_cases)}, "
            f"golden has {len(expected_cases)}."
        )

    max_dev = 0.0
    problems: list[str] = []
    for index in range(len(expected_cases)):
        actual = actual_cases[index]
        golden = expected_cases[index]
        if actual["token_ids"] != golden.get("token_ids"):
            problems.append(
                f"case {index}: greedy token id sequence diverged from golden "
                f"(actual {actual['token_ids'][:8]}... vs golden "
                f"{list(golden.get('token_ids', []))[:8]}...)"
            )
        for kind in ("completion_logprobs", "prompt_logprobs"):
            deviation, step_problems = _compare_steps(
                actual.get(kind, []), golden.get(kind, []), kind, index, atol, rtol
            )
            max_dev = max(max_dev, deviation)
            problems.extend(step_problems)

    if problems:
        detail = "\n  ".join(problems[:20])
        raise AssertionError(
            "golden comparison failed "
            f"(max logprob deviation={max_dev:.3e}, atol={atol}, "
            f"rtol={rtol}); {len(problems)} problem(s):\n  {detail}"
        )
    return max_dev


def compare_golden(
    outputs: InferenceResult,
    golden_path: Any,
    chip: str,
    update: bool = False,
    atol: Any = None,
    rtol: Any = None,
) -> str:
    """Compare ``outputs`` against the per-chip golden, or (re)write it.

    Returns one of ``"updated"`` (``--update-golden``), ``"created"`` (no golden
    yet: written and flagged pending-review, not a failure) or ``"compared"``.
    Raises :class:`AssertionError` with the maximum logprob deviation when the
    comparison fails.

    ``atol`` / ``rtol`` optionally override the module-wide logprob tolerance
    for a single model (see :class:`ModelConfig`); ``None`` keeps the shared
    ``GOLDEN_LOGPROB_ATOL`` / ``GOLDEN_LOGPROB_RTOL`` so existing callers are
    unaffected.
    """
    effective_atol = GOLDEN_LOGPROB_ATOL if atol is None else float(atol)
    effective_rtol = GOLDEN_LOGPROB_RTOL if rtol is None else float(rtol)
    golden_path = Path(golden_path)
    record = _record_to_json(outputs, golden_path, chip, effective_atol, effective_rtol)

    if update:
        _write_golden(record, golden_path)
        return "updated"

    if not golden_path.exists():
        _write_golden(record, golden_path)
        warnings.warn(
            f"no golden baseline for chip {chip!r}; wrote a pending-review golden to "
            f"{golden_path}. Re-run with a reviewed baseline to enforce comparison.",
            stacklevel=2,
        )
        return "created"

    expected = json.loads(golden_path.read_text())
    _assert_chip(expected, chip, golden_path)
    _compare_cases(record, expected, effective_atol, effective_rtol)
    return "compared"
