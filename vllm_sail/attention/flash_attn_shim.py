# SPDX-License-Identifier: Apache-2.0
"""Redirect both upstream FlashAttention import paths before platform resolution.

An empty-target vLLM has no CUDA FA binaries. Seeding only its interface still
executes upstream's package initializer and binary availability guard. Install
our package AND interface, without importing torch or SDK wheels in this hook.
The real PPU implementation loads when callers query support or invoke FA.
"""

from __future__ import annotations

import sys

from vllm.logger import init_logger

# Inherit vLLM's configured handler so the redirect appears in startup logs.
logger = init_logger("vllm.ppu.flash_attn_shim")

_UPSTREAM_PACKAGE = "vllm.vllm_flash_attn"
_UPSTREAM_MODULE = f"{_UPSTREAM_PACKAGE}.flash_attn_interface"
_installed = False


def install() -> None:
    """Install the import-safe PPU API; preserve existing module references."""
    global _installed
    if _installed:
        return

    from vllm_sail.attention import flash_attn as package
    from vllm_sail.attention.flash_attn import flash_attn_interface as interface

    for name, replacement in (
        (_UPSTREAM_PACKAGE, package),
        (_UPSTREAM_MODULE, interface),
    ):
        target = sys.modules.get(name)
        if target is None:
            sys.modules[name] = replacement
            continue
        if target is replacement:
            continue
        # A failed upstream package import can leave its interface cached. Also
        # update an already-loaded package: consumers import functions from both.
        for export in replacement.__all__:
            setattr(target, export, getattr(replacement, export))
        for export in interface._AVAILABILITY_EXPORTS:
            target.__dict__.pop(export, None)
        target.__getattr__ = interface.__getattr__
        logger.warning(
            "vllm-sail: %s was imported before platform registration; rebound "
            "its FA API to PPU. Previously copied function references may be "
            "stale; register the platform before importing attention backends.",
            name,
        )

    installed_package = sys.modules[_UPSTREAM_PACKAGE]
    installed_package.flash_attn_interface = sys.modules[_UPSTREAM_MODULE]
    parent = sys.modules.get("vllm")
    if parent is not None:
        parent.vllm_flash_attn = installed_package
    _installed = True
    logger.info(
        "vllm-sail: %s and its interface use the PPU FlashAttention API; "
        "SDK wheels load on use, without vLLM's _vllm_fa2_C/_vllm_fa3_C imports.",
        _UPSTREAM_PACKAGE,
    )
