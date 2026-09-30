# SPDX-License-Identifier: Apache-2.0
"""PPU MXFP8 dense fallback using upstream's BF16 emulation lifecycle."""

from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.platforms import current_platform


class PPUEmulationMxfp8LinearKernel(EmulationMxfp8LinearKernel):
    """Preserve E8M0 / 32-element scales instead of selecting CUDA Marlin.

    ModelOpt expands checkpoint block rows before this kernel runs. Upstream
    emulation handles dequantization, bias and activation dtype conversion.
    This is a W8A16 fallback; MXFP4 experts keep their independent MoE backend.
    """

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_ppu():
            return False, "PPU MXFP8 emulation requires PPU."
        return True, None
