# SPDX-License-Identifier: Apache-2.0
"""Import the ported upstream kernels into the op namespaces vLLM already uses.

A vLLM built with ``VLLM_TARGET_DEVICE=empty`` ships no compiled extensions, so
``torch.ops._C``, ``_C_cache_ops``, ``_C_cuda_utils``, ``_C_custom_ar`` and
``_moe_C`` are unclaimed. Importing the plugin's own shared objects registers the
ported kernels into exactly those namespaces, and every wrapper in
``vllm/_custom_ops.py`` then resolves normally -- with zero patches. This is the
handshake described in ``docs/developer_guide/kernels.md``.

The counterpart risk is installing beside a *kernel-ful* vLLM, where both
libraries would register the same schemas and torch would raise during the
``.so`` import. Registration is a static-initializer side effect and cannot be
partially skipped, so the check has to happen *before* the import.
"""

from __future__ import annotations

import importlib
import logging

__all__ = ["import_kernels"]

logger = logging.getLogger(__name__)

# Extension module -> (op namespace, an op that namespace is certain to carry).
# The sentinel answers "has someone already claimed this namespace?" without
# importing anything.
_EXTENSIONS: tuple[tuple[str, str, str], ...] = (
    ("vllm_sail._upstream_C", "_C", "rms_norm"),
    ("vllm_sail._upstream_moe_C", "_moe_C", "moe_sum"),
)

# A preflight checks representative Tier A device kernels, not just namespace
# ownership: rms_norm alone does not establish that activations were registered.
_REQUIRED_OPS = {
    "_C": ("rms_norm", "silu_and_mul"),
    "_moe_C": ("moe_sum",),
}
_BUILD_HELP = (
    "Use VLLM_TARGET_DEVICE=empty vLLM and activate SAIL PyTorch plus the SAIL SDK; "
    "unset VLLM_SAIL_SKIP_EXT, then rebuild/install vllm-sail from its source root "
    "with `python -m pip install -e . --no-build-isolation --no-deps`. "
    "Check vllm_sail.__file__ for a source checkout shadowing the installed wheel. "
    "See docs/getting_started/installation.md."
)

_imported: set[str] = set()


def _verify_ops(torch, module: str, namespace: str) -> None:
    ops = getattr(torch.ops, namespace)
    missing = [name for name in _REQUIRED_OPS[namespace] if not hasattr(ops, name)]
    if missing:
        raise RuntimeError(
            f"vllm-sail: {module} did not provide required Tier A ops in "
            f"torch.ops.{namespace}: {', '.join(missing)}. Another library may "
            f"have partially claimed the namespace. {_BUILD_HELP}"
        )
    for name in _REQUIRED_OPS[namespace]:
        if not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"{namespace}::{name}", "CUDA"
        ):
            raise RuntimeError(
                f"vllm-sail: torch.ops.{namespace}.{name} has a schema but no "
                f"CUDA implementation (PPU uses the CUDA dispatch key). "
                f"{_BUILD_HELP}"
            )


def import_kernels(*, strict: bool = False) -> tuple[str, ...]:
    """Import the ported-kernel extensions. Returns the modules actually loaded.

    Safe to call repeatedly and safe to call on an install with no compiled
    extensions: a missing extension is logged, never raised, matching the
    tolerance of vLLM's own ``Platform.import_kernels``. ``strict=True`` is an
    e2e preflight: fail on loading errors and verify the minimum Tier A schemas
    and CUDA dispatch registrations. It does not launch any device kernels.
    """
    import torch

    loaded: list[str] = []
    for module, namespace, sentinel in _EXTENSIONS:
        if module in _imported:
            if strict:
                _verify_ops(torch, module, namespace)
            continue
        claimed = getattr(torch.ops, namespace, None)
        if claimed is not None and hasattr(claimed, sentinel):
            logger.warning(
                "vllm-sail: torch.ops.%s is already registered (found %r), so %s "
                "was not imported. This means vLLM was built with its own "
                "kernels; vllm-sail expects VLLM_TARGET_DEVICE=empty.",
                namespace,
                sentinel,
                module,
            )
            _imported.add(module)
            if strict:
                _verify_ops(torch, module, namespace)
            continue
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            if strict:
                raise RuntimeError(
                    f"vllm-sail: cannot load {module}: {exc}. {_BUILD_HELP}"
                ) from exc
            logger.warning(
                "vllm-sail: %s is unavailable (%s). Upstream ops in torch.ops.%s "
                "will not resolve. %s",
                module,
                exc,
                namespace,
                _BUILD_HELP,
            )
            # An unsuccessful import must not count as a loaded extension.
            # In particular a later explicit preflight must retry it and retain
            # the original loader/ABI error rather than report only missing ops.
            continue
        else:
            loaded.append(module)
        _imported.add(module)
        if strict:
            _verify_ops(torch, module, namespace)
    return tuple(loaded)
