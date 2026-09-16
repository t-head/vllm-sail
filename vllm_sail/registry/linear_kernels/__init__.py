# SPDX-License-Identifier: Apache-2.0
"""Register PPU dense-GEMM kernels with vLLM's linear-kernel selector.

This is the plugin's cleanest extension point: upstream exposes
``vllm.model_executor.kernels.linear.register_linear_kernel(kernel_class,
platform, kernel_type)`` and keys the candidate lists by ``PlatformEnum``. All
five PPU ``scaled_mm`` kernels register themselves here, so the in-tree fork's
45-line edit to ``kernels/linear/__init__.py`` needs **no patch at all**.

(vLLM-MetaX has this wired but commented out in
``vllm_metax/registry/linear_kernels/fp8.py`` -- the extension point exists and
works; it just was never used.)

## Ordering

``register_linear_kernel`` **appends**, but selection walks the list in order and
takes the first supported kernel -- so an appended PPU kernel would lose to
upstream's CUDA kernels that precede it. The fork gets the order it wants by
literally writing ``PlatformEnum.PPU: [PPUDeepGemmFP8..., ...]`` as the whole
list.

Since ``PPUPlatform`` declares ``PlatformEnum.CUDA`` (see
``vllm_sail/platform.py``), we share CUDA's bucket and must **prepend** instead.
:func:`_register_first` does that against the same module-level dicts
``register_linear_kernel`` mutates. It is a documented upstream API gap: the
right fix is a ``priority=`` argument on ``register_linear_kernel``.
"""

from __future__ import annotations

#: Maps the ``kernel_type`` accepted by ``register_linear_kernel`` to the
#: module-level dict it mutates. Kept in one place so a rename upstream produces
#: one clear failure here rather than silent non-registration.
_REGISTRY_BY_KERNEL_TYPE = {
    "mp": "_POSSIBLE_KERNELS",
    "int8": "_POSSIBLE_INT8_KERNELS",
    "fp8": "_POSSIBLE_FP8_KERNELS",
    "fp8_block": "_POSSIBLE_FP8_BLOCK_KERNELS",
}

_registered = False


def _register_first(
    kernel_class: type,
    kernel_type: str,
    platform: object | None = None,
) -> None:
    """Insert ``kernel_class`` at the *front* of a kernel candidate list.

    Mirrors ``register_linear_kernel`` but prepends. Raises if the upstream dict
    it targets has been renamed, so a silent loss of PPU kernel selection is
    impossible.
    """
    from vllm.model_executor.kernels import linear as linear_kernels
    from vllm.platforms.interface import PlatformEnum

    if platform is None:
        platform = PlatformEnum.CUDA

    attribute = _REGISTRY_BY_KERNEL_TYPE.get(kernel_type)
    if attribute is None:
        raise ValueError(
            f"Unknown kernel_type {kernel_type!r}; expected one of "
            f"{', '.join(_REGISTRY_BY_KERNEL_TYPE)}."
        )

    registry = getattr(linear_kernels, attribute, None)
    if registry is None:
        raise AttributeError(
            f"vllm.model_executor.kernels.linear.{attribute} does not exist. "
            "Upstream renamed or restructured the linear-kernel registry; PPU "
            "kernels would otherwise be silently unregistered. Update "
            "_REGISTRY_BY_KERNEL_TYPE in vllm_sail/registry/linear_kernels/."
        )

    candidates = registry.setdefault(platform, [])
    if kernel_class in candidates:
        return
    candidates.insert(0, kernel_class)


def _selection_plan() -> list[tuple[str, list[type]]]:
    """PPU kernel preference order per kernel type.

    Mirrors the ``PlatformEnum.PPU`` lists the fork writes into
    ``kernels/linear/__init__.py``. Imported lazily because these modules pull in
    torch and the PPU SDK.
    """
    from vllm_sail.model_executor.kernels.linear.scaled_mm.ppu import (
        PPUCutlassFp8BlockScaledMMKernel,
        PPUCutlassFP8ScaledMMLinearKernel,
        PPUDeepGemmFp8BlockScaledMMKernel,
        PPUDeepGemmFP8ScaledMMLinearKernel,
        PPUInt8ScaledMMLinearKernel,
    )

    return [
        ("int8", [PPUInt8ScaledMMLinearKernel]),
        (
            "fp8",
            [
                PPUDeepGemmFP8ScaledMMLinearKernel,
                PPUCutlassFP8ScaledMMLinearKernel,
            ],
        ),
        (
            "fp8_block",
            [
                PPUDeepGemmFp8BlockScaledMMKernel,
                PPUCutlassFp8BlockScaledMMKernel,
            ],
        ),
    ]


def register() -> None:
    """Register every PPU dense-GEMM kernel. Idempotent."""
    global _registered
    if _registered:
        return

    from vllm.logger import init_logger

    plan = _selection_plan()
    for kernel_type, kernel_classes in plan:
        # Reversed so the first entry in each list ends up first overall.
        for kernel_class in reversed(kernel_classes):
            _register_first(kernel_class, kernel_type)

    _registered = True
    init_logger(__name__).debug(
        "Registered PPU linear kernels: %s",
        ", ".join(cls.__name__ for _, classes in plan for cls in classes),
    )
