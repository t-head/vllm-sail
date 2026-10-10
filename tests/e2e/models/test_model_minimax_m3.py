# SPDX-License-Identifier: Apache-2.0
"""Offline model E2E precision test for MiniMax M3 (plan PART 5).

Drives vLLM's offline ``LLM`` directly (no server, no OpenAI client) against the
fixed MiniMax M3 checkpoint and compares greedy generation with a per-chip
golden baseline. MiniMax M3 registers a PPU subclass that implements
``SupportsQuant`` and drives a sparse-MoE attention preprocessing path, so this
test is the end-to-end guard that the registered subclass, its topk-buffer
layout and its quant mapping do not regress numerics on real hardware.

The only runtime input is the weight path (``VLLM_SAIL_MODEL_ROOT`` overrides the
fixed checkpoint). Tensor parallelism follows the detected chip (ZW-810E -> 16,
ZW-890P -> 8); every other parameter is fixed in ``_offline_runner`` /
``MINIMAX_M3_CONFIG``. The M3 topk-buffer layout and quant mapping are decided by
the model config + PPU patch, so no extra LLM knob is required here.
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
#: MiniMax M3 but does not record its NAS path. This follows the same
#: ``/nas_aisw/datasets/checkpoints/LLM/<Vendor>/<version>/<Model>`` convention
#: as the verified Qwen3.8-Next entry in ``scripts/ci/ppu_e2e_models.json``.
#: Confirm/adjust against the internal model-catalog sheet before the first
#: golden capture; ``VLLM_SAIL_MODEL_ROOT`` overrides it without a code change.
MINIMAX_M3_CHECKPOINT = "/nas_aisw/datasets/checkpoints/LLM/MiniMax/m3/MiniMax-M3"

#: Per-model fixed config. Every field left at its default matches the plan's
#: shared parameter table (max_model_len=4096, dtype=bfloat16,
#: gpu_memory_utilization=0.90, enforce_eager=True, seed=0, greedy sampling).
MINIMAX_M3_CONFIG = ModelConfig(
    key="minimax_m3",
    checkpoint=MINIMAX_M3_CHECKPOINT,
    trust_remote_code=True,
)


def test_minimax_m3_offline_precision(offline_llm, golden_store, chip_capability):
    """Build -> greedy generate the fixed prompt set -> compare per-chip golden."""
    chip = _offline_runner.chip_key_for_capability(chip_capability)
    checkpoint = resolve_checkpoint(MINIMAX_M3_CONFIG.checkpoint)

    llm = offline_llm(checkpoint, MINIMAX_M3_CONFIG)
    result = _offline_runner.run_inference(
        llm, _offline_runner.PROMPTS, MINIMAX_M3_CONFIG
    )

    # Sanity: greedy decode must produce a token sequence for every prompt.
    assert len(result.cases) == len(_offline_runner.PROMPTS)
    for index, case in enumerate(result.cases):
        assert case.token_ids, f"prompt {index} produced no tokens"

    golden_path = golden_store.path(f"{MINIMAX_M3_CONFIG.key}_{chip}.json")
    status = _offline_runner.compare_golden(
        result,
        golden_path,
        chip,
        update=golden_store.update,
        atol=MINIMAX_M3_CONFIG.logprob_atol,
        rtol=MINIMAX_M3_CONFIG.logprob_rtol,
    )
    assert status in {"compared", "updated", "created"}
