# SPDX-License-Identifier: Apache-2.0
"""Residual fork M-file patches owned by no other phase.

These cover the small PPU hunks the Phase 0 M-file classification left without
an owner:

* ``all2all_deepep_shrink``     — vllm/distributed/device_communicators/all2all.py
* ``gpu_worker_max_split``      — vllm/v1/worker/gpu_worker.py
* ``compilation_attention_ops`` — vllm/config/compilation.py
* ``kernel_moe_backend_literal``— vllm/config/kernel.py
* ``minimax_fused_ar_rms_qk``   — vllm/model_executor/layers/minimax_rms_norm/
  rms_norm_tp.py

``vllm/v1/kv_cache_interface.py`` was residual scope until Phase 4 took it
(``patch/enhancement/attention/kv_cache_interface.py``); nothing here patches
it. ``vllm/benchmarks/serve.py`` is deliberately NOT patched: its fork hunk is
benchmark ergonomics only (see the report).

Wired into ``vllm_sail/patch/enhancement/__init__.py``: importing this package
applies every residual patch, so that single import is the complete
registration.

Design note: unlike the sibling patch modules (which install at their own import
time), the leaf modules here only *define* their replacements and expose
``install()``; installation happens when THIS package is imported. That keeps
every leaf module importable on a bare CPU runner (no vLLM, no torch side
effects), which the metadata-shape unit tests rely on.
"""

from __future__ import annotations

from vllm_sail.patch.enhancement.residual import (
    all2all_deepep_shrink,
    compilation_attention_ops,
    gpu_worker_max_split,
    kernel_moe_backend_literal,
    minimax_fused_ar_rms_qk,
)

__all__ = [
    "all2all_deepep_shrink",
    "compilation_attention_ops",
    "gpu_worker_max_split",
    "install",
    "kernel_moe_backend_literal",
    "minimax_fused_ar_rms_qk",
]

#: Installation order is not significant; each module patches an independent
#: upstream target.
_MODULES = (
    all2all_deepep_shrink,
    compilation_attention_ops,
    gpu_worker_max_split,
    kernel_moe_backend_literal,
    minimax_fused_ar_rms_qk,
)

_installed = False


def install() -> None:
    """Apply every residual patch. Requires vLLM to be importable. Idempotent."""
    global _installed
    if _installed:
        return
    for module in _MODULES:
        module.install()
    _installed = True


install()
