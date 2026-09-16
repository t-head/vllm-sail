# SPDX-License-Identifier: Apache-2.0
"""Make vLLM's CUDA platform importable in an empty-target installation.

``vllm.platforms.cuda`` unconditionally imports ``vllm._C_stable_libtorch`` at
module import time.  That extension is intentionally absent when vLLM is built
with ``VLLM_TARGET_DEVICE=empty``; PPU registers the equivalent op namespaces
later from its own extensions.  The platform plugin hook runs before the CUDA
platform is imported, so it can supply this side-effect-only module name.

The shim is installed only when no loaded or discoverable real extension owns
the name.  A present-but-broken extension is therefore never hidden.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
import types
from typing import Literal

__all__ = ["UPSTREAM_EXTENSION", "install"]

UPSTREAM_EXTENSION = "vllm._C_stable_libtorch"

logger = logging.getLogger(__name__)


def install() -> Literal["shimmed", "loaded", "available"]:
    """Provide vLLM's side-effect-only extension name when it is truly absent."""
    if UPSTREAM_EXTENSION in sys.modules:
        return "loaded"

    try:
        available = importlib.util.find_spec(UPSTREAM_EXTENSION)
    except (ImportError, AttributeError, ValueError):
        # A missing or partially initialized parent package cannot contain a
        # discoverable extension. vLLM itself is loaded in the real plugin path.
        available = None
    if available is not None:
        return "available"

    module = types.ModuleType(
        UPSTREAM_EXTENSION,
        "Empty-target import shim installed by vllm-sail; op registration is "
        "provided later by vllm_sail.native.install().",
    )
    module.__package__ = "vllm"
    module.__spec__ = importlib.machinery.ModuleSpec(UPSTREAM_EXTENSION, loader=None)
    module.__vllm_sail_empty_shim__ = True
    sys.modules[UPSTREAM_EXTENSION] = module
    parent = sys.modules.get("vllm")
    if parent is not None:
        parent._C_stable_libtorch = module
    logger.info(
        "vllm-sail: installed an empty-target import shim for %s; PPU kernels "
        "will register the op namespaces during PPUPlatform.import_kernels()",
        UPSTREAM_EXTENSION,
    )
    return "shimmed"
