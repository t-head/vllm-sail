# SPDX-License-Identifier: Apache-2.0
"""Offline model E2E precision test for Qwen3.8-Next (plan PART 5).

Drives vLLM's offline ``LLM`` directly (no server, no OpenAI client) against
the fixed Qwen3.8-Next checkpoint and compares greedy generation with a
per-chip golden baseline. Qwen3.8-Next registers no PPU model class: it runs on
the upstream architecture plus runtime patches (NVTX profiling scaffold +
per-chip MoE backends, hybrid GDN -> PLA). This test is the end-to-end guard
that those patches do not perturb numerics.

The only runtime input is the weight path (``VLLM_SAIL_MODEL_ROOT`` overrides
the fixed checkpoint). Tensor parallelism follows the detected chip
(ZW-810E -> 16, ZW-890P -> 8); every other parameter is fixed in
``_offline_runner`` / ``QWEN3_NEXT_CONFIG``.
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
QWEN3_NEXT_CHECKPOINT = "/nas_aisw/datasets/checkpoints/LLM/Qwen/v3.8/Qwen3.8-27B"

#: Per-model fixed config. Every field left at its default matches the plan's
#: shared parameter table (max_model_len=4096, dtype=bfloat16,
#: gpu_memory_utilization=0.90, enforce_eager=True, seed=0, greedy sampling).
QWEN3_NEXT_CONFIG = ModelConfig(
    key="qwen3_next",
    checkpoint=QWEN3_NEXT_CHECKPOINT,
    trust_remote_code=True,
)


def test_qwen3_next_offline_precision(offline_llm, golden_store, chip_capability):
    """Build -> greedy generate the fixed prompt set -> compare per-chip golden."""
    chip = _offline_runner.chip_key_for_capability(chip_capability)
    checkpoint = resolve_checkpoint(QWEN3_NEXT_CONFIG.checkpoint)

    llm = offline_llm(checkpoint, QWEN3_NEXT_CONFIG)
    result = _offline_runner.run_inference(
        llm, _offline_runner.PROMPTS, QWEN3_NEXT_CONFIG
    )

    # Sanity: greedy decode must produce a token sequence for every prompt.
    assert len(result.cases) == len(_offline_runner.PROMPTS)
    for index, case in enumerate(result.cases):
        assert case.token_ids, f"prompt {index} produced no tokens"

    golden_path = golden_store.path(f"{QWEN3_NEXT_CONFIG.key}_{chip}.json")
    status = _offline_runner.compare_golden(
        result, golden_path, chip, update=golden_store.update
    )
    assert status in {"compared", "updated", "created"}
