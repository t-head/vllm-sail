# SPDX-License-Identifier: Apache-2.0
"""Exclude CuTeDSL on PPU and run the PPU DeepGEMM warmup.

Upstream ``warmup/kernel_warmup.py`` ends with a DeepGEMM warmup block gated
on ``is_deep_gemm_supported()``, which upstream defines via
``support_deep_gemm()`` — Hopper and Blackwell only. PPU devices report
sm80-class capability, so that block never fires on PPU, and PPU needs its own
warmup anyway: ``vllm_sail.model_executor.warmup.deep_gemm_warmup`` exercises
the PPU DeepGEMM wrapper's int8/bf16/fp4 grouped GEMMs and its tuned-config
lookup, none of which CUDA's warmup reaches.

PPU shares CUDA identity but cannot compile CuTeDSL. Before delegating to
upstream, discard CuTeDSL registrations and guard the legacy warmup provider
and its imported alias. Other registered JIT kernels retain their warmup.
The PPU DeepGEMM warmup is appended with the fork's gate — never when
``VLLM_DEEP_GEMM_WARMUP=skip``,
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
_CUTEDSL_MODULE = "vllm.model_executor.warmup.cutedsl_warmup"


def _exclude_ppu_cutedsl_registrations(worker) -> None:
    from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import (
        VllmCuTeDSLJitKernel,
    )

    registry = getattr(worker.model_runner, "jit_warmup_registry", None)
    if registry is None:
        return
    # vLLM 0.30 has no public backend filter. Keep the existing mapping and
    # registration arguments for every supported kernel; remove only CuTeDSL.
    registrations = registry._registrations
    unsupported = [
        kernel for kernel in registrations if isinstance(kernel, VllmCuTeDSLJitKernel)
    ]
    for kernel in unsupported:
        del registrations[kernel]
    if unsupported:
        logger.warning(
            "Skipping unsupported CuTeDSL JIT warmup on PPU: %s",
            ", ".join(type(kernel).__name__ for kernel in unsupported),
        )


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
    _upstream_cutedsl_warmup = importlib.import_module(_CUTEDSL_MODULE).cutedsl_warmup

    @patch(
        _MODULE,
        "cutedsl_warmup",
        reason="Rebind the preloaded legacy CuTeDSL warmup alias to the PPU guard.",
        affected_versions=">=0.30.0,<0.31.0",
        remove_when="The legacy CuTeDSL warmup gates execution on backend capabilities.",
    )
    @patch(
        _CUTEDSL_MODULE,
        "cutedsl_warmup",
        reason="Legacy CuTeDSL warmup checks is_cuda(), which also accepts PPU.",
        affected_versions=">=0.30.0,<0.31.0",
        remove_when="The legacy CuTeDSL warmup gates execution on backend capabilities.",
    )
    def cutedsl_warmup() -> None:
        from vllm.platforms import current_platform

        if not current_platform.is_ppu():
            return _upstream_cutedsl_warmup()

    @patch(
        _MODULE,
        "kernel_warmup",
        reason=(
            "Exclude unsupported CuTeDSL JIT registrations on PPU. "
            "Upstream's DeepGEMM warmup is gated on support_deep_gemm() "
            "(sm90/sm100/sm120), so it never runs on PPU; and CUDA's warmup "
            "would not exercise the PPU DeepGEMM wrapper's kernels or tuned "
            "configs anyway. Appends the PPU DeepGEMM warmup with the fork's "
            "env gates after the upstream body."
        ),
        affected_versions=">=0.30.0,<0.31.0",
        remove_when=(
            "upstream gives kernel_warmup a per-platform warmup registry (or "
            "vllm.utils.deep_gemm dispatches on current_platform), so the PPU "
            "warmup can register itself."
        ),
    )
    def kernel_warmup(worker, *, process_local_only: bool = False):
        from vllm.platforms import current_platform

        if current_platform.is_ppu():
            _exclude_ppu_cutedsl_registrations(worker)
        result = _upstream_kernel_warmup(worker, process_local_only=process_local_only)

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
