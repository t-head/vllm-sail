# SPDX-License-Identifier: Apache-2.0
"""The PPU native layer: kernel corpus, build toolchain, and runtime honesty.

This package is the only place that knows PPU kernels are compiled by ``hgcc``
against HGGC headers rather than by ``nvcc`` against CUDA headers. Everything
above it -- the platform, the attention backends, the model overrides -- stays
CUDA-shaped on purpose (``docs/developer_guide/kernels.md``).

Modules:

* :mod:`vllm_sail.native.bootstrap`   empty-vLLM platform import handshake
* :mod:`vllm_sail.native.manifest`    the kernel corpus, as declared
* :mod:`vllm_sail.native.plugin_manifest` plugin-owned HGGC sources and bindings
* :mod:`vllm_sail.native.toolchain`   the build gate, importable by ``setup.py``
* :mod:`vllm_sail.native.extensions`  claiming the upstream op namespaces
* :mod:`vllm_sail.native.stubs`       honest failures for excluded kernels
"""

from __future__ import annotations

__all__ = ["install"]


def install() -> None:
    """Load the ported kernels, then stub whatever is still missing.

    Order matters: stubs must never shadow a real implementation, so the
    extensions register first and :func:`vllm_sail.native.stubs.install` skips
    any op that already exists.
    """
    from vllm_sail.native import extensions, portable, stubs, triton_ops

    extensions.import_kernels()
    portable.install()
    triton_ops.install()
    stubs.install()
