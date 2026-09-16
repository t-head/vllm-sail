# SPDX-License-Identifier: Apache-2.0
"""Run the PPU DeepGEMM warmup from ``kernel_warmup``.

Upstream ``warmup/kernel_warmup.py`` ends with a DeepGEMM warmup block gated
on ``is_deep_gemm_supported()``, which upstream defines via
``support_deep_gemm()`` — Hopper and Blackwell only. PPU devices report
sm80-class capability, so that block never fires on PPU, and PPU needs its own
warmup anyway: ``vllm_sail.model_executor.warmup.deep_gemm_warmup`` exercises
the PPU DeepGEMM wrapper's int8/bf16/fp4 grouped GEMMs and its tuned-config
lookup, none of which CUDA's warmup reaches.

The in-tree fork edits ``kernel_warmup`` to swap both the import and the gate.
Here the patch delegates: the upstream body runs unchanged (its DeepGEMM block
stays inert on PPU for the capability reason above), and the PPU warmup is
appended with the fork's gate — never when ``VLLM_DEEP_GEMM_WARMUP=skip``,
only when the PPU DeepGEMM wrapper reports support, and only when at least one
of the MoE / dense backend selectors still allows DeepGEMM.

The target module imports the full worker/warmup stack at module scope. On
hosts where that chain cannot run (e.g. a minimal test venv without vLLM's
engine dependencies) there is also no warmup entry point for the patch to act
on, so this module skips itself with a warning instead of taking down the rest
of the plugin install. On a real PPU host the import always succeeds — vLLM's
own runtime requires the same dependencies.
"""

from __future__ import annotations

import importlib

from vllm.logger import init_logger

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_MODULE = "vllm.model_executor.warmup.kernel_warmup"


def _ppu_deep_gemm_warmup_enabled() -> bool:
    """The fork's gate for running DeepGEMM warmup on PPU."""
    import vllm.envs as envs

    import vllm_sail.envs as ppu_envs
    from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

    if envs.VLLM_DEEP_GEMM_WARMUP == "skip" or not is_deep_gemm_supported():
        return False
    moe_backend = ppu_envs.VLLM_SAIL_MOE_BACKEND
    dense_backend = ppu_envs.VLLM_SAIL_DENSE_BACKEND
    return (
        (not moe_backend or moe_backend == "deepgemm")
        or (not dense_backend or dense_backend == "deepgemm")
        or (ppu_envs.VLLM_SAIL_DENSE_BF16_DEEPGEMM)
    )


try:
    _upstream_kernel_warmup = importlib.import_module(_MODULE).kernel_warmup
except ImportError as exc:
    logger.warning(
        "vllm-sail: skipping the %s patch because the target module did not "
        "import (%s). This is expected in minimal test environments; on a PPU "
        "host the import succeeds.",
        _MODULE,
        exc,
    )
else:

    @patch(
        _MODULE,
        "kernel_warmup",
        reason=(
            "Upstream's DeepGEMM warmup is gated on support_deep_gemm() "
            "(sm90/sm100/sm120), so it never runs on PPU; and CUDA's warmup "
            "would not exercise the PPU DeepGEMM wrapper's kernels or tuned "
            "configs anyway. Appends the PPU DeepGEMM warmup with the fork's "
            "env gates after the upstream body."
        ),
        affected_versions=">=0.27.0,<0.28.0",
        remove_when=(
            "upstream gives kernel_warmup a per-platform warmup registry (or "
            "vllm.utils.deep_gemm dispatches on current_platform), so the PPU "
            "warmup can register itself."
        ),
    )
    def kernel_warmup(worker, *, process_local_only: bool = False):
        result = _upstream_kernel_warmup(worker, process_local_only=process_local_only)

        from vllm.platforms import current_platform

        if process_local_only or not current_platform.is_ppu():
            return result

        if _ppu_deep_gemm_warmup_enabled():
            from vllm_sail.model_executor.warmup.deep_gemm_warmup import (
                deep_gemm_warmup,
            )

            deep_gemm_warmup(
                worker.get_model(),
                worker.scheduler_config.max_num_batched_tokens,
            )
        return result
