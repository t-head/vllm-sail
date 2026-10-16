# SPDX-License-Identifier: Apache-2.0
"""Exclude unsupported warmup on PPU and run the PPU DeepGEMM warmup.

Upstream ``warmup/kernel_warmup.py`` ends with a DeepGEMM warmup block gated
on ``is_deep_gemm_supported()``, which upstream defines via
``support_deep_gemm()`` — Hopper and Blackwell only. PPU devices report
sm80-class capability, so that block never fires on PPU, and PPU needs its own
warmup anyway: ``vllm_sail.model_executor.warmup.deep_gemm_warmup`` exercises
the PPU DeepGEMM wrapper's int8/bf16/fp4 grouped GEMMs and its tuned-config
lookup, none of which CUDA's warmup reaches.

PPU shares CUDA identity but cannot compile CuTeDSL. Before delegating to
upstream, discard CuTeDSL registrations and guard the legacy warmup provider
and its imported alias. PPU also inherits CUDA's unconditional FP8 warmup
registrations, but its Triton backend supports no fp8 dtypes on sm80-class
devices (``current_platform.supports_fp8()`` is CUDA's
``has_device_capability(89)``). The fork guards this in two layers:
``VllmTritonJitKernel.compile_many`` drops compile keys whose inputs carry an
FP8 dtype before any compile is attempted, and ``VllmTritonJitKernel.compile``
skips — with a warning — compile errors naming an unsupported FP8 dtype from
kernels that cast to FP8 internally. Other errors, devices with FP8 support,
and other registered JIT kernels keep upstream warmup behaviour.
The PPU DeepGEMM warmup is appended with the fork's gate — never when
``VLLM_DEEP_GEMM_WARMUP=skip``,
only when the PPU DeepGEMM wrapper reports support, and only when at least one
of the MoE / dense backend selectors still allows DeepGEMM.

The target modules import the full worker/warmup stack at module scope. On
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
_TRITON_MODULE = "vllm.model_executor.warmup.jit_warmup_triton_helper"


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


def _is_unsupported_fp8_compile_error(exc: Exception) -> bool:
    """Triton reports unsupported fp8 dtypes as a plain-text architecture error."""
    message = str(exc)
    return "not supported in this architecture" in message and "fp8" in message


def _compile_key_uses_fp8(compile_key) -> bool:
    """FP8 appears in warmup inputs as torch or Triton dtype spellings."""
    for _, value in getattr(compile_key, "inputs", ()):
        name = str(getattr(value, "dtype", ""))
        if "fp8" in name or "float8" in name:
            return True
    return False


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
        affected_versions=">=0.31.0,<0.32.0",
        remove_when="The legacy CuTeDSL warmup gates execution on backend capabilities.",
    )
    @patch(
        _CUTEDSL_MODULE,
        "cutedsl_warmup",
        reason="Legacy CuTeDSL warmup checks is_cuda(), which also accepts PPU.",
        affected_versions=">=0.31.0,<0.32.0",
        remove_when="The legacy CuTeDSL warmup gates execution on backend capabilities.",
    )
    def cutedsl_warmup() -> None:
        from vllm.platforms import current_platform

        if not current_platform.is_ppu():
            return _upstream_cutedsl_warmup()

    try:
        _triton_helper = importlib.import_module(_TRITON_MODULE)
        _upstream_triton_compile_many = _triton_helper.VllmTritonJitKernel.compile_many
        _upstream_triton_compile = _triton_helper.VllmTritonJitKernel.compile
    except ImportError as exc:
        logger.warning(
            "vllm-sail: skipping the %s FP8 warmup guard because the target "
            "module did not import (%s).",
            _TRITON_MODULE,
            exc,
        )
    else:

        @patch(
            _TRITON_MODULE,
            "VllmTritonJitKernel.compile_many",
            reason=(
                "PPU inherits CUDA's unconditional FP8 JIT warmup "
                "registrations, but its Triton backend rejects fp8 dtypes. On "
                "PPU without FP8 support, drop warmup compile keys whose "
                "inputs carry an FP8 dtype before compiling, so unsupported "
                "variants are never attempted and sibling keys still warm."
            ),
            affected_versions=">=0.31.0,<0.32.0",
            remove_when=(
                "upstream gates Triton FP8 warmup registrations on device FP8 "
                "support, or the PPU Triton backend accepts fp8 dtypes."
            ),
        )
        def vllm_triton_jit_kernel_compile_many(self, compile_keys):
            from vllm.platforms import current_platform

            keys = list(compile_keys)
            if current_platform.is_ppu() and not current_platform.supports_fp8():
                supported = [key for key in keys if not _compile_key_uses_fp8(key)]
                if len(supported) != len(keys):
                    logger.info(
                        "Skipping %d FP8 JIT warmup compile keys on this device "
                        "for %s.",
                        len(keys) - len(supported),
                        type(self).__name__,
                    )
                    keys = supported
            return _upstream_triton_compile_many(self, keys)

        @patch(
            _TRITON_MODULE,
            "VllmTritonJitKernel.compile",
            reason=(
                "PPU inherits CUDA's unconditional FP8 JIT warmup "
                "registrations, but its Triton backend rejects fp8 dtypes. "
                "Kernels that cast to FP8 internally keep non-FP8 warmup "
                "inputs, so the compile_many dtype gate cannot see them; skip "
                "(with a warning) compile errors naming an unsupported FP8 "
                "dtype when current_platform.supports_fp8() is False, instead "
                "of aborting engine startup. Other errors still propagate."
            ),
            affected_versions=">=0.31.0,<0.32.0",
            remove_when=(
                "upstream gates Triton FP8 warmup registrations on device FP8 "
                "support, or the PPU Triton backend accepts fp8 dtypes."
            ),
        )
        def vllm_triton_jit_kernel_compile(self, compile_key):
            try:
                return _upstream_triton_compile(self, compile_key)
            except Exception as exc:
                from vllm.platforms import current_platform

                unsupported_fp8 = (
                    not current_platform.supports_fp8()
                    and _is_unsupported_fp8_compile_error(exc)
                )
                if not unsupported_fp8:
                    raise
                logger.warning(
                    "Skipping uncompilable FP8 JIT warmup on this device: %s (%s).",
                    type(self).__name__,
                    exc,
                )

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
        affected_versions=">=0.31.0,<0.32.0",
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
