# SPDX-License-Identifier: Apache-2.0
"""Lazy PPU KDA SDK access; never probe the NVIDIA native-op stub."""

from __future__ import annotations

from importlib import import_module

from vllm.platforms import current_platform

from vllm_sail import envs


def get_pla_kda_kernel(mode):
    if not current_platform.is_ppu() or not envs.VLLM_SAIL_USE_PLA:
        return None
    module, name = {
        "decode": ("pla.decode.kda", "fused_kda_decode_mega_forward"),
        "prefill": ("pla.prefill.flashkdapro", "flashkda_fwd"),
    }[mode]
    try:
        return getattr(import_module(module), name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"PPU KDA requires {module}.{name}; install a compatible PLA SDK "
            "or set VLLM_SAIL_USE_PLA=0 to use Triton."
        ) from exc
