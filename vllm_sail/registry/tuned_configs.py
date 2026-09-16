# SPDX-License-Identifier: Apache-2.0
"""Make the plugin's tuned kernel configs reachable.

## The problem

The plugin ships 71 tuned-config JSON files (fused-MoE Triton configs and block
-quantized GEMM configs) for ``PPU-ZW810``, ``PPU-ZW810E`` and ``ZW-M890P``. In
the in-tree fork these sat *inside* vLLM's own package, so upstream's loaders
found them by resolving ``configs/`` relative to their own ``__file__``:

* ``fused_moe/fused_moe.py`` -> ``os.path.dirname(realpath(__file__)) / "configs"``
* ``quantization/utils/int8_utils.py`` and ``fp8_utils.py`` -> same, for
  ``utils/configs``

Ported into ``vllm_sail/``, those files are outside every directory upstream looks
in, so **every config silently misses** and the kernels fall back to untuned
defaults. Silently: the miss path only emits a `warning_once`. That makes it a
performance bug that is easy to ship and hard to notice, which is why it gets an
explicit test.

## The fix, in two halves

**Fused-MoE configs** need no patch. Upstream already checks a user-defined
folder *before* its own (``fused_moe.py``: ``envs.VLLM_TUNED_CONFIG_FOLDER``), so
we point that at our directory when the user has not set it themselves. A user
value always wins -- we only fill in a default.

**Block-quant configs** have no such hook, so ``get_w8a8_block_int8_configs`` and
its fp8 counterpart are patched to search the plugin directory as well. Those
patches live in ``vllm_sail/patch/enhancement/tuned_config_lookup.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from vllm.logger import init_logger

logger = init_logger(__name__)

#: Directory holding the fused-MoE Triton configs (``E=*,device_name=*.json``).
FUSED_MOE_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent
    / "model_executor"
    / "layers"
    / "fused_moe"
    / "configs"
)

#: Directory holding block-quantized GEMM configs (``N=*,K=*,device_name=*.json``).
BLOCK_QUANT_CONFIG_DIR = (
    Path(__file__).resolve().parent.parent
    / "model_executor"
    / "layers"
    / "quantization"
    / "utils"
    / "configs"
)

_ENV_NAME = "VLLM_TUNED_CONFIG_FOLDER"

_registered = False


def register() -> None:
    """Point vLLM's tuned-config lookup at the plugin's configs. Idempotent."""
    global _registered
    if _registered:
        return
    _registered = True

    if not FUSED_MOE_CONFIG_DIR.is_dir():
        logger.warning(
            "vllm-sail tuned fused-MoE config directory %s is missing; MoE "
            "kernels will use untuned defaults.",
            FUSED_MOE_CONFIG_DIR,
        )
        return

    existing = os.environ.get(_ENV_NAME)
    if existing:
        # The user asked for a specific folder. Respect it, but say clearly that
        # the shipped PPU configs are now unreachable -- otherwise a stray env
        # var looks like a mysterious performance regression.
        if os.path.realpath(existing) != str(FUSED_MOE_CONFIG_DIR):
            logger.warning(
                "%s is already set to %s, so vllm-sail's bundled tuned MoE "
                "configs in %s will not be used. Unset it to use them.",
                _ENV_NAME,
                existing,
                FUSED_MOE_CONFIG_DIR,
            )
        return

    os.environ[_ENV_NAME] = str(FUSED_MOE_CONFIG_DIR)
    logger.info(
        "vllm-sail: %s defaulted to %s (%d tuned MoE configs).",
        _ENV_NAME,
        FUSED_MOE_CONFIG_DIR,
        len(list(FUSED_MOE_CONFIG_DIR.glob("*.json"))),
    )


def block_quant_config_path(json_file_name: str) -> Path:
    """Absolute path of a plugin-shipped block-quant config file."""
    return BLOCK_QUANT_CONFIG_DIR / json_file_name
