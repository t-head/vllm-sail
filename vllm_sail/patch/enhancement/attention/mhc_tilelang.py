# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.model_executor.kernels.mhc.tilelang``.

The fork's PPU changes to the three tilelang MHC launchers:

* DeepGEMM import switch: ``tf32_hc_prenorm_gemm`` / ``is_deep_gemm_supported``
  come from the PPU DeepGEMM wrapper under ``is_ppu()``.
* ``n_splits`` is forced to 1 on PPU instead of ``compute_num_split(...)``.
* ``gemm_out_mul`` / ``gemm_out_sqrsum`` use ``torch.zeros`` instead of
  ``torch.empty`` — in ``mhc_pre_tilelang`` this is UNCONDITIONAL in the fork
  ("NOTE(PPU): Need to use torch.zeros instead of torch.empty here to fix
  precision problem on PPU SM80 with DeepSeek V4 Pro W8A8"), so it also changes
  stock CUDA behaviour; reproduced as-in-fork and flagged in the Phase-4 report.
  In ``mhc_pre_broadcast_tilelang`` the same zeros change is PPU-gated.

All three are mid-function changes, so delegation cannot express them; each is
a verbatim upstream body copy (fork HEAD) with only the fork's lines marked.
The bodies are rebased onto the target module's ``__dict__`` so every free name
(``torch``, ``_tilelang_hc_prenorm_gemm``, ...) resolves exactly as upstream.

Deviation from the fork: the fork imports ``vllm.utils.ppu_deep_gemm``, a
fork-added module; the plugin imports ``vllm_sail.utils.deep_gemm`` instead
(same API). The fork's function-local ``from vllm.platforms import
current_platform`` imports are kept unchanged.
"""

from __future__ import annotations

from vllm.model_executor.kernels.mhc import tilelang as _tilelang

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.30.0,<0.31.0"
_MODULE = "vllm.model_executor.kernels.mhc.tilelang"


def _with_target_globals(fn):
    from vllm_sail.patch.bodies import bind_body

    return bind_body(fn, _tilelang)


def _mhc_pre_tilelang_body(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;
        norm_weight: optional RMSNorm weight, shape (hidden_size,), dtype
            torch.bfloat16. When provided, RMSNorm is fused into the
            layer_input write path of the big_fuse kernel.
        norm_eps: epsilon for the fused RMSNorm; only consulted when
            norm_weight is given.

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.platforms import current_platform

    # PPU MODIFICATION: begin
    # Fork's import switch; the plugin wrapper module replaces the fork's
    # vllm.utils.ppu_deep_gemm.
    if current_platform.is_ppu():
        from vllm_sail.utils.deep_gemm import tf32_hc_prenorm_gemm
    else:
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    # PPU MODIFICATION: end
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        from vllm_sail.utils.deep_gemm import is_deep_gemm_supported
    else:
        from vllm.utils.deep_gemm import is_deep_gemm_supported
    # PPU MODIFICATION: end

    use_deep_gemm = is_deep_gemm_supported()
    if use_deep_gemm:
        # these numbers are from deepgemm kernel impl
        block_k = 64
        block_m = 64
        # PPU MODIFICATION: begin
        n_splits = 1 if current_platform.is_ppu() else compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))
        # PPU MODIFICATION: end
    else:
        n_splits = 1

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    # PPU MODIFICATION: begin
    # NOTE(PPU): Need to use torch.zeros instead of torch.empty here
    # to fix precision problem on PPU SM80 with DeepSeek V4 Pro W8A8.
    # UNCONDITIONAL in the fork (also changes stock CUDA).
    gemm_out_mul = torch.zeros(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.zeros(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )
    # PPU MODIFICATION: end

    residual_2d = residual_flat.view(num_tokens, hc_mult * hidden_size)
    if use_deep_gemm:
        tf32_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    else:
        _tilelang_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            hc_mult,
        )

    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
        )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


patch(
    _MODULE,
    "mhc_pre_tilelang",
    reason=(
        "PPU MHC pre kernel needs the PPU DeepGEMM wrapper's "
        "tf32_hc_prenorm_gemm, n_splits forced to 1, and torch.zeros instead "
        "of torch.empty for the split-k accumulators (precision fix for PPU "
        "SM80 with DeepSeek V4 Pro W8A8; the zeros change is unconditional in "
        "the fork). Mid-function changes, hence the verbatim body."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vllm.utils.deep_gemm dispatches on current_platform internally and "
        "upstream adopts the PPU n_splits/zeros changes."
    ),
)(_with_target_globals(_mhc_pre_tilelang_body))


def _mhc_pre_broadcast_tilelang_body(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fn_broadcast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """First-layer mHC pre for a residual broadcast from ``(T, H)``."""
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_broadcast_with_norm_tilelang,
    )
    from vllm.platforms import current_platform
    from vllm.utils.math_utils import cdiv

    assert norm_weight is not None, "broadcast mHC pre currently requires fused RMSNorm"
    assert residual.dtype == torch.bfloat16
    assert residual.dim() == 2
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hidden_size = residual.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)
    assert fn_broadcast is not None
    assert fn_broadcast.dtype == torch.float32
    assert fn_broadcast.shape == (hc_mult3, hidden_size)

    if norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)
    if not norm_weight.is_contiguous():
        norm_weight = norm_weight.contiguous()

    residual_flat = residual
    num_tokens = residual.shape[0]

    # PPU MODIFICATION: begin
    n_splits = (
        1
        if current_platform.is_ppu()
        else compute_num_split(64, hidden_size, cdiv(num_tokens, 64))
    )
    # PPU MODIFICATION: end

    residual_out = torch.empty(
        num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    # PPU MODIFICATION: begin
    # Fork's import switch plus the PPU-gated zeros accumulators; the plugin
    # wrapper module replaces the fork's vllm.utils.ppu_deep_gemm.
    if current_platform.is_ppu():
        # NOTE(PPU): Need to use torch.zeros instead of torch.empty here
        # to fix precision problem on PPU SM80 with DeepSeek V4 Pro W8A8
        gemm_out_mul = torch.zeros(
            n_splits, num_tokens, hc_mult3, dtype=torch.float32,
            device=residual.device
        )
        gemm_out_sqrsum = torch.zeros(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )
        from vllm_sail.utils.deep_gemm import tf32_hc_prenorm_gemm
    else:
        gemm_out_mul = torch.empty(
            n_splits, num_tokens, hc_mult3, dtype=torch.float32,
            device=residual.device
        )
        gemm_out_sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    # PPU MODIFICATION: end

    tf32_hc_prenorm_gemm(
        residual_flat,
        fn_broadcast,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )
    mhc_pre_big_fuse_broadcast_with_norm_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        residual_out,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        n_splits,
        hc_mult,
    )
    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        layer_input,
    )


patch(
    _MODULE,
    "mhc_pre_broadcast_tilelang",
    reason=(
        "PPU first-layer (broadcast) MHC pre kernel: n_splits forced to 1 and "
        "torch.zeros split-k accumulators plus the PPU DeepGEMM wrapper's "
        "tf32_hc_prenorm_gemm. Mid-function changes, hence the verbatim body."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vllm.utils.deep_gemm dispatches on current_platform internally and "
        "upstream adopts the PPU n_splits/zeros changes."
    ),
)(_with_target_globals(_mhc_pre_broadcast_tilelang_body))


def _mhc_fused_post_pre_tilelang_body(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one MHC post block followed by the next MHC pre block.

    When ``norm_weight`` is provided, the layer_input_cur output is the
    RMSNorm'd activation (fused into the kernel); otherwise it is the
    raw pre-norm activation as before.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_fused_tilelang,
        mhc_post_tilelang,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.math_utils import cdiv

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    assert n_splits in (1, 2, 4, 8)
    assert hidden_size % n_splits == 0

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    from vllm.platforms import current_platform

    # PPU MODIFICATION: begin
    # Fork's import switch; the plugin wrapper module replaces the fork's
    # vllm.utils.ppu_deep_gemm.
    if current_platform.is_ppu():
        from vllm_sail.utils.deep_gemm import is_deep_gemm_supported
    else:
        from vllm.utils.deep_gemm import is_deep_gemm_supported
    # PPU MODIFICATION: end

    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = num_tokens <= 16
    if use_small_fma:
        # TODO(gnovack): investigate autotuning these heuristics
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
    else:
        if use_deep_gemm:
            # these number are from deepgemm kernel impl
            block_k = 64
            block_m = 64
            # PPU MODIFICATION: begin
            n_splits = 1 if current_platform.is_ppu() else compute_num_split(
                block_k, hc_hidden_size, cdiv(num_tokens, block_m)
            )
            # PPU MODIFICATION: end
        else:
            n_splits = 1

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    if use_small_fma:
        mhc_fused_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_n=tile_n,
            n_splits=n_splits,
        )
    else:
        mhc_post_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            residual.shape[-2],
            residual.shape[-1],
        )

        residual_cur_2d = residual_cur.view(num_tokens, hc_mult * hidden_size)
        if use_deep_gemm:
            # PPU MODIFICATION: begin
            if current_platform.is_ppu():
                from vllm_sail.utils.deep_gemm import tf32_hc_prenorm_gemm
            else:
                from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
            # PPU MODIFICATION: end

            tf32_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                n_splits,
            )
        else:
            _tilelang_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                hidden_size,
                hc_mult,
            )

    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


patch(
    _MODULE,
    "mhc_fused_post_pre_tilelang",
    reason=(
        "PPU MHC fused post+pre kernel needs the PPU DeepGEMM wrapper's "
        "is_deep_gemm_supported/tf32_hc_prenorm_gemm and n_splits forced to 1. "
        "Mid-function changes, hence the verbatim body."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vllm.utils.deep_gemm dispatches on current_platform internally and "
        "upstream adopts the PPU n_splits change."
    ),
)(_with_target_globals(_mhc_fused_post_pre_tilelang_body))


# Upstream registered the original callable before these module patches ran.
# Own the PPU torch.ops boundary so tracing reaches the new implementation too.
from vllm.utils.torch_utils import direct_register_custom_op

for _name in ("mhc_pre_tilelang", "mhc_fused_post_pre_tilelang"):
    direct_register_custom_op(
        op_name=f"ppu_{_name}",
        op_func=getattr(_tilelang, _name),
        mutates_args=[],
        fake_impl=getattr(_tilelang, f"_{_name}_fake"),
    )

# The package star-export and V4 direct imports may already hold old functions.
import sys

from vllm_sail.patch.utils import PATCH_MARKER

for _name in (
    "mhc_pre_tilelang",
    "mhc_pre_broadcast_tilelang",
    "mhc_fused_post_pre_tilelang",
):
    _replacement = getattr(_tilelang, _name)
    _original = getattr(_replacement, PATCH_MARKER)[f"{_MODULE}.{_name}"]
    for _consumer in (
        "vllm.model_executor.kernels.mhc",
        "vllm.models.deepseek_v4.nvidia.model",
    ):
        _module = sys.modules.get(_consumer)
        if _module is not None and getattr(_module, _name, None) is _original:
            patch(
                _consumer,
                _name,
                reason="Preloaded mHC exports must use the PPU DeepGEMM launcher.",
                affected_versions=_AFFECTED,
                remove_when="mHC callers resolve launchers through the provider module.",
            )(_replacement)
