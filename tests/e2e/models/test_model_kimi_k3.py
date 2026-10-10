# SPDX-License-Identifier: Apache-2.0
"""Offline model E2E precision test for Kimi K3 (plan PART 5).

Drives vLLM's offline ``LLM`` directly (no server, no OpenAI client) against the
fixed Kimi K3 checkpoint and compares greedy generation with a per-chip golden
baseline. Kimi K3 registers no new PPU model class: it runs the upstream
architecture plus a runtime patch that routes KDA (Kimi Delta Attention) onto the
PPU PLA backend. This test is the end-to-end guard that the patch path — not a
registered subclass — produces numerics matching the captured baseline.

The only runtime input is the weight path (``VLLM_SAIL_MODEL_ROOT`` overrides the
fixed checkpoint). Tensor parallelism follows the detected chip (ZW-810E -> 16,
ZW-890P -> 8); every other parameter is fixed in ``_offline_runner`` /
``KIMI_K3_CONFIG``. ``VLLM_SAIL_USE_PLA=1`` is pinned in the config ``env`` so
KDA dispatch deterministically hits the PPU PLA backend.

Tolerance note: linear attention accumulates error over long sequences, so the
plan relaxes the golden logprob tolerance for K3 slightly. The relaxed values are
carried per-model on ``KIMI_K3_CONFIG`` (never on the shared module constants),
leaving every other model's comparison untouched.
"""

from __future__ import annotations

import pytest

# --- Guard chain: skip the whole module without a real PPU device. ------------
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("requires a real PPU/CUDA device", allow_module_level=True)
pytest.importorskip("vllm")

from vllm.platforms import current_platform  # noqa: E402

if not current_platform.is_ppu():
    pytest.skip("requires a PPU platform (is_ppu() is False)", allow_module_level=True)

from tests.e2e.models import _offline_runner  # noqa: E402
from tests.e2e.models._offline_runner import (  # noqa: E402
    ModelConfig,
    resolve_checkpoint,
)

pytestmark = [pytest.mark.ppu, pytest.mark.model_e2e]

#: Fixed checkpoint. Overridable at runtime only via ``VLLM_SAIL_MODEL_ROOT``.
#:
#: ASSUMPTION: the implementation plan (outputs/04_CI实施方案.html PART 5) names
#: Kimi K3 but does not record its NAS path. This follows the same
#: ``/nas_aisw/datasets/checkpoints/LLM/<Vendor>/<version>/<Model>`` convention
#: as the verified Qwen3.8-Next entry in ``scripts/ci/ppu_e2e_models.json``
#: (Kimi is published by Moonshot AI). Confirm/adjust against the internal
#: model-catalog sheet before the first golden capture; ``VLLM_SAIL_MODEL_ROOT``
#: overrides it without a code change.
KIMI_K3_CHECKPOINT = "/nas_aisw/datasets/checkpoints/LLM/Moonshot/v3/Kimi-K3"

#: Relaxed golden logprob tolerance for K3 only (plan PART 5: linear attention
#: accumulates error over long sequences). ASSUMPTION: these values are a
#: starting point (5x the shared atol, 2x the shared rtol) and should be tuned
#: against the observed drift after the first real-hardware golden capture. They
#: live on the config so the shared module constants — and every other model's
#: comparison — stay at the tight defaults.
KIMI_K3_LOGPROB_ATOL = 5e-3
KIMI_K3_LOGPROB_RTOL = 2e-2

#: Per-model fixed config. Every field left at its default matches the plan's
#: shared parameter table (max_model_len=4096, dtype=bfloat16,
#: gpu_memory_utilization=0.90, enforce_eager=True, seed=0, greedy sampling).
#: ``env`` pins ``VLLM_SAIL_USE_PLA=1`` so KDA dispatch hits the PPU PLA backend.
KIMI_K3_CONFIG = ModelConfig(
    key="kimi_k3",
    checkpoint=KIMI_K3_CHECKPOINT,
    trust_remote_code=True,
    logprob_atol=KIMI_K3_LOGPROB_ATOL,
    logprob_rtol=KIMI_K3_LOGPROB_RTOL,
    env={"VLLM_SAIL_USE_PLA": "1"},
)


def test_kimi_k3_offline_precision(offline_llm, golden_store, chip_capability):
    """Build -> greedy generate the fixed prompt set -> compare per-chip golden."""
    chip = _offline_runner.chip_key_for_capability(chip_capability)
    checkpoint = resolve_checkpoint(KIMI_K3_CONFIG.checkpoint)

    llm = offline_llm(checkpoint, KIMI_K3_CONFIG)
    result = _offline_runner.run_inference(llm, _offline_runner.PROMPTS, KIMI_K3_CONFIG)

    # Sanity: greedy decode must produce a token sequence for every prompt.
    assert len(result.cases) == len(_offline_runner.PROMPTS)
    for index, case in enumerate(result.cases):
        assert case.token_ids, f"prompt {index} produced no tokens"

    golden_path = golden_store.path(f"{KIMI_K3_CONFIG.key}_{chip}.json")
    status = _offline_runner.compare_golden(
        result,
        golden_path,
        chip,
        update=golden_store.update,
        atol=KIMI_K3_CONFIG.logprob_atol,
        rtol=KIMI_K3_CONFIG.logprob_rtol,
    )
    assert status in {"compared", "updated", "created"}
