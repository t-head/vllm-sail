# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821, I001
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Register PPU standard and batched INT4 W4A16 experts."""

from __future__ import annotations

import inspect

from vllm.model_executor.layers.fused_moe.oracle import int_wna16 as oracle

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch
from vllm_sail.registry.moe_backends._extend import extend_enum

_MODULE = "vllm.model_executor.layers.fused_moe.oracle.int_wna16"
_META = dict(
    reason="The WNA16 oracle needs the plugin's PPU DeepGEMM experts and packed weight layout.",
    affected_versions=">=0.27.0,<0.28.0",
    remove_when="The MoE oracle supports registering plugin WNA16 backends.",
)
WNA16MoEBackend = oracle.WNA16MoEBackend
for _name in ("PPU_DEEPGEMM", "BATCHED_PPU_DEEPGEMM"):
    extend_enum(WNA16MoEBackend, _name, _name)
_PPU = (WNA16MoEBackend.PPU_DEEPGEMM, WNA16MoEBackend.BATCHED_PPU_DEEPGEMM)


def backend_to_kernel_cls(
    backend: WNA16MoEBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Return the experts class for the given backend, or None for NONE."""
    if backend == WNA16MoEBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        )

        return [
            BatchedHummingGroupedExperts,
            HummingGroupedExperts,
            HummingIndexedExperts,
        ]
    elif backend == WNA16MoEBackend.MARLIN:
        return [MarlinExperts]
    elif backend == WNA16MoEBackend.BATCHED_MARLIN:
        return [BatchedMarlinExperts]
    # PPU MODIFICATION: begin
    elif backend == WNA16MoEBackend.PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
            PPUDeepGemmExperts,
        )

        return [PPUDeepGemmExperts]
    elif backend == WNA16MoEBackend.BATCHED_PPU_DEEPGEMM:
        from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            PPUBatchedDeepGemmExperts,
        )

        return [PPUBatchedDeepGemmExperts]
    # PPU MODIFICATION: end
    elif backend == WNA16MoEBackend.FLASHINFER_TRTLLM:
        return [TrtLlmMxint4ExpertsMonolithic]
    elif backend == WNA16MoEBackend.TRITON:
        return [TritonWNA16Experts]
    elif backend == WNA16MoEBackend.XPU:
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
            XPUExpertsWNA16,
        )

        return [XPUExpertsWNA16]
    elif backend == WNA16MoEBackend.CPU:
        from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
            CPUExpertsInt4,
        )

        return [CPUExpertsInt4]
    elif backend == WNA16MoEBackend.EMULATION:
        from vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe import (
            Int4EmulationTritonExperts,
        )

        return [Int4EmulationTritonExperts]
    else:
        raise ValueError(f"Unknown WNA16 MoE backend: {backend.value}")


backend_to_kernel_cls = patch(_MODULE, "backend_to_kernel_cls", **_META)(
    bind_body(backend_to_kernel_cls, oracle)
)


def _get_priority_backends() -> list[WNA16MoEBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    if current_platform.is_cpu():
        return [WNA16MoEBackend.CPU]
    if current_platform.is_xpu():
        return [WNA16MoEBackend.XPU]
    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        return [
            WNA16MoEBackend.PPU_DEEPGEMM,
            WNA16MoEBackend.BATCHED_PPU_DEEPGEMM,
            WNA16MoEBackend.MARLIN,
            WNA16MoEBackend.BATCHED_MARLIN,
            WNA16MoEBackend.TRITON,
            WNA16MoEBackend.HUMMING,
            WNA16MoEBackend.EMULATION,
        ]
    # PPU MODIFICATION: end

    return [
        WNA16MoEBackend.FLASHINFER_TRTLLM,
        WNA16MoEBackend.MARLIN,
        WNA16MoEBackend.BATCHED_MARLIN,
        WNA16MoEBackend.TRITON,
        WNA16MoEBackend.HUMMING,
        WNA16MoEBackend.EMULATION,
    ]


_get_priority_backends = patch(_MODULE, "_get_priority_backends", **_META)(
    bind_body(_get_priority_backends, oracle)
)


def map_wna16_backend(runner_backend: MoEBackend) -> WNA16MoEBackend:
    """Map user's MoEBackend to WNA16MoEBackend."""
    mapping = {
        "triton": WNA16MoEBackend.TRITON,
        "marlin": WNA16MoEBackend.MARLIN,
        # PPU MODIFICATION: begin
        "ppu_deep_gemm": WNA16MoEBackend.PPU_DEEPGEMM,
        "ppu_deep_gemm_w4a16": WNA16MoEBackend.PPU_DEEPGEMM,
        # PPU MODIFICATION: end
        "humming": WNA16MoEBackend.HUMMING,
        "flashinfer_trtllm": WNA16MoEBackend.FLASHINFER_TRTLLM,
        "emulation": WNA16MoEBackend.EMULATION,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for WNA16 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


map_wna16_backend = patch(_MODULE, "map_wna16_backend", **_META)(
    bind_body(map_wna16_backend, oracle)
)


def select_wna16_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey,
    quant_config: QuantizationConfig | QuantizationArgs,
    may_have_zp: bool,
    may_have_bias: bool,
    allow_tile_padding: bool = False,
) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
    """Select the WNA16 MoE backend.

    Args:
        config: the shared ``FusedMoEConfig`` for this layer.
        weight_key: The QuantKey describing the weight quantization.
                    Must have int4 or int8 type.
        quant_config: Quantization structure and checkpoint format description.
        may_have_zp: Whether the integration can provide weight zero points.
        may_have_bias: Whether the integration can provide expert bias.

    Returns:
        A tuple of (``WNA16MoEBackend``, experts class or ``None``).
    """

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: WNA16MoEBackend):
        return f"Using '{backend.value}' WNA16 MoE backend."

    def _make_log_unsupported(backend: WNA16MoEBackend, reason: str | None) -> str:
        if reason:
            return (
                f"WNA16 MoE backend '{backend.value}' does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"WNA16 MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: WNA16MoEBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
        reason: str | None = None
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_wna16_backend(runner_backend)
        reason = _backend_incompatibility_reason(
            requested_backend,
            config,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
        if reason is not None:
            raise ValueError(_make_log_unsupported(requested_backend, reason))
        # PPU MODIFICATION: begin
        if (
            activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and requested_backend == WNA16MoEBackend.PPU_DEEPGEMM
        ):
            requested_backend = WNA16MoEBackend.BATCHED_PPU_DEEPGEMM
        # PPU MODIFICATION: end
        return _return_or_raise(
            requested_backend, config, weight_key, None, activation_format
        )

    # Select kernels in order of backend.
    AVAILABLE_BACKENDS = _get_priority_backends()

    for backend in AVAILABLE_BACKENDS:
        reason = _backend_incompatibility_reason(
            backend,
            config,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
        if reason is not None:
            logger.debug_once(_make_log_unsupported(backend, reason), scope="local")
            continue
        activation_key = None  # always BF16 activation for WNA16 MoE
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

    raise NotImplementedError(
        "No WNA16 MoE backend supports the deployment configuration."
    )


select_wna16_moe_backend = patch(_MODULE, "select_wna16_moe_backend", **_META)(
    bind_body(select_wna16_moe_backend, oracle)
)


def make_wna16_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    backend: WNA16MoEBackend = WNA16MoEBackend.MARLIN,
    layer: torch.nn.Module | None = None,
    is_k_full: bool = False,
    w13_g_idx: torch.Tensor | None = None,
    w2_g_idx: torch.Tensor | None = None,
    w13_g_idx_sort_indices: torch.Tensor | None = None,
    w2_g_idx_sort_indices: torch.Tensor | None = None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
        CPUExpertsInt4,
    )
    from vllm.model_executor.layers.fused_moe.experts.int4_emulation_moe import (
        Int4EmulationTritonExperts,
    )
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import (
        XPUExpertsWNA16,
    )

    # PPU MODIFICATION: begin
    from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        PPUBatchedDeepGemmExperts,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
        PPUDeepGemmExperts,
    )
    # PPU MODIFICATION: end

    # Currently, we only support TrtLlmMxint4ExpertsMonolithic, MarlinExperts,
    # BatchedMarlinExperts, XPUExpertsWNA16, CPUExpertsInt4, the Humming
    # grouped/indexed experts, and Int4EmulationTritonExperts
    allowed_experts: tuple[type[mk.FusedMoEExperts], ...] = (
        MarlinExperts,
        BatchedMarlinExperts,
        TritonWNA16Experts,
        # PPU MODIFICATION: begin
        PPUDeepGemmExperts,
        PPUBatchedDeepGemmExperts,
        # PPU MODIFICATION: end
        TrtLlmMxint4ExpertsMonolithic,
        XPUExpertsWNA16,
        CPUExpertsInt4,
        Int4EmulationTritonExperts,
    )
    if backend == WNA16MoEBackend.HUMMING:
        allowed_experts += tuple(backend_to_kernel_cls(WNA16MoEBackend.HUMMING))
    assert experts_cls in allowed_experts

    is_monolithic = experts_cls.is_monolithic()

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")
    logger.info_once("Using %s", experts_cls.__name__, scope="local")

    extra_args: dict[str, Any] = {}
    if backend == WNA16MoEBackend.HUMMING:
        assert layer is not None
        extra_args = {"layer": layer}
    elif issubclass(experts_cls, MarlinExpertsBase):
        extra_args = {
            "w13_g_idx": w13_g_idx,
            "w2_g_idx": w2_g_idx,
            "w13_g_idx_sort_indices": w13_g_idx_sort_indices,
            "w2_g_idx_sort_indices": w2_g_idx_sort_indices,
            "is_k_full": is_k_full,
        }

    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        extra_args["max_num_tokens"] = max_num_tokens
        extra_args["num_dispatchers"] = prepare_finalize.num_dispatchers()

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        **extra_args,
    )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )


make_wna16_moe_kernel = patch(_MODULE, "make_wna16_moe_kernel", **_META)(
    bind_body(make_wna16_moe_kernel, oracle)
)

_upstream_reason = oracle._backend_incompatibility_reason
_upstream_convert = oracle.convert_to_wna16_moe_kernel_format


@patch(_MODULE, "_backend_incompatibility_reason", **_META)
def _backend_incompatibility_reason(
    backend, moe_config, quant_config, may_have_zp, may_have_bias, allow_tile_padding
):
    if backend not in _PPU:
        return _upstream_reason(
            backend,
            moe_config,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
    from compressed_tensors.quantization import QuantizationArgs
    from vllm import envs
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    from vllm_sail.patch.enhancement.models.moe_marlin_gate import supports_marlin_layout

    if may_have_zp or may_have_bias or moe_config.has_bias:
        return "PPU W4A16 does not consume zero points or expert bias"
    if not isinstance(quant_config, (AutoGPTQConfig, QuantizationArgs)):
        return "PPU W4A16 requires GPTQ or compressed-tensors symmetric INT4 weights"
    if getattr(quant_config, "desc_act", False) or getattr(
        quant_config, "actorder", None
    ) in ("group", "dynamic"):
        return "PPU W4A16 does not consume GPTQ activation ordering"
    if (envs.VLLM_MARLIN_INPUT_DTYPE or "").lower() in ("int8", "fp8"):
        return "PPU W4A16 requires 16-bit activation packing"
    if quant_config.group_size != 32 or not supports_marlin_layout(
        moe_config, 32, allow_tile_padding
    ):
        return "PPU W4A16 requires group32 and Marlin-compatible tile dimensions"
    return None


@patch(_MODULE, "convert_to_wna16_moe_kernel_format", **_META)
def convert_to_wna16_moe_kernel_format(*args, **kwargs):
    bound = inspect.signature(_upstream_convert).bind(*args, **kwargs)
    bound.apply_defaults()
    if bound.arguments["backend"] in _PPU:
        # Shared load-time Marlin packing, followed by PPU DeepGEMM inference.
        bound.arguments["backend"] = WNA16MoEBackend.MARLIN
    return _upstream_convert(*bound.args, **bound.kwargs)
