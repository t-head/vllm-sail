# SPDX-License-Identifier: Apache-2.0
"""Runtime honesty for kernels PPU does not have.

This module prevents two failure modes:

1. ``AttributeError: '_OpNamespace' object has no attribute 'cutlass_scaled_mm'``
   surfacing as an opaque crash deep inside a quantization path.
2. A capability probe that *raises* where vLLM expects a bool, breaking backend
   selection instead of steering it.

The op list is derived from the kernel manifest, so a tier-``x`` op cannot be
silently forgotten and promoting a kernel out of tier ``x`` removes its stub
automatically. The *schemas* come from the generated ``excluded_ops.toml`` copied
into installed wheels from ``csrc/upstream``. ``tools/port_upstream_kernels.py``
emits it while pruning upstream's binding TUs -- registering an op requires its
exact signature, and copying signatures by hand is precisely the kind of drift
the port tool exists to eliminate.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from vllm_sail.native import manifest as _manifest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

logger = logging.getLogger(__name__)

__all__ = [
    "ExcludedOp",
    "default_schema_path",
    "install",
    "load_schemas",
    "unsupported_ops",
]


@dataclass(frozen=True)
class ExcludedOp:
    """One op that upstream defines but the PPU corpus does not compile."""

    namespace: str
    name: str
    schema: str
    kind: str
    tier: str
    reason: str

    @property
    def is_probe(self) -> bool:
        return self.kind == "probe"


def default_schema_path() -> Path:
    from vllm_sail.native import resources

    return resources.resolve(Path(__file__).resolve().parent, "excluded_ops.toml")


def unsupported_ops() -> tuple[str, ...]:
    """Uncompiled ops without a plugin-provided Triton implementation.

    Available even when the corpus has not been ported yet, which is why
    diagnostics (``collect_env``, the docs table) use this rather than the
    generated schema file.
    """
    from vllm_sail.native.triton_ops import IMPLEMENTATIONS

    implemented = {name for _, name in IMPLEMENTATIONS}
    return tuple(
        name for name in _manifest.load().excluded_ops() if name not in implemented
    )


def load_schemas(path: str | Path | None = None) -> tuple[ExcludedOp, ...]:
    """Read the generated schema file. Empty tuple when it has not been generated."""
    src = Path(path) if path is not None else default_schema_path()
    if not src.is_file():
        return ()
    raw = tomllib.loads(src.read_text(encoding="utf-8"))
    return tuple(
        ExcludedOp(
            namespace=item["namespace"],
            name=item["name"],
            schema=item["schema"],
            kind=item.get("kind", "compute"),
            tier=item.get("tier", "x"),
            reason=item.get("reason", ""),
        )
        for item in raw.get("op") or ()
    )


def install(path: str | Path | None = None) -> tuple[str, ...]:
    """Register a stub for every excluded op. Returns the ops actually stubbed.

    Skips any op whose namespace already provides it: that means a kernel-ful
    vLLM owns the namespace, and its real implementation must win.
    """
    import torch

    ops = load_schemas(path)
    if not ops:
        missing = unsupported_ops()
        if missing:
            logger.warning(
                "vllm-sail: %d upstream ops have no PPU implementation but no "
                "stub schemas are available (excluded_ops.toml is missing -- run "
                "tools/port_upstream_kernels.py and rebuild the package). Calls will fail "
                "with AttributeError instead of a clear message: %s",
                len(missing),
                ", ".join(missing),
            )
        return ()

    installed: list[str] = []
    for op in ops:
        if _op_exists(torch.ops, op.namespace, op.name):
            logger.debug(
                "vllm-sail: %s::%s is already registered; leaving it alone",
                op.namespace,
                op.name,
            )
            continue
        _register(torch, op)
        installed.append(f"{op.namespace}::{op.name}")

    logger.warning(
        "vllm-sail: stubbed %d unsupported upstream op(s); capability probes "
        "report False and compute ops raise NotImplementedError",
        len(installed),
    )
    return tuple(installed)


def _op_exists(ops, namespace: str, name: str) -> bool:
    """Whether the exact default or named overload is already registered."""
    namespace_obj = getattr(ops, namespace, None)
    if namespace_obj is None:
        return False
    packet_name, separator, overload = name.partition(".")
    packet = getattr(namespace_obj, packet_name, None)
    if packet is None:
        return False
    return hasattr(packet, overload if separator else "default")


def _register(torch, op: ExcludedOp) -> None:
    lib = torch.library.Library(op.namespace, "FRAGMENT")
    lib.define(op.schema)
    lib.impl(
        op.name,
        _probe if op.is_probe else _unsupported(op),
        "CompositeExplicitAutograd",
    )
    # The Library object owns the registration for its lifetime, so it must
    # outlive this function.
    _KEEPALIVE.append(lib)


_KEEPALIVE: list = []


def _probe(*args, **kwargs) -> bool:
    return False


def _unsupported(op: ExcludedOp):
    def raise_unsupported(*args, **kwargs):
        raise NotImplementedError(
            f"{op.namespace}::{op.name} has no PPU implementation "
            f"(hazard: {op.reason or 'unsupported'}; kernel manifest tier "
            f"{op.tier!r}). See docs/developer_guide/kernels.md."
        )

    return raise_unsupported
