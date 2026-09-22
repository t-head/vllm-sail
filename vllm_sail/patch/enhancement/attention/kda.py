# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Use PLA KDA kernels on PPU, retaining Triton when PLA is disabled."""

from __future__ import annotations

from vllm.models.kimi_k3.nvidia import kda as _kda

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch

_MODULE = "vllm.models.kimi_k3.nvidia.kda"
_META = dict(
    reason="Kimi KDA must select PPU PLA kernels instead of NVIDIA FlashKDA/native decode.",
    affected_versions=">=0.30.0,<0.31.0",
    remove_when="Kimi KDA supports platform kernel registration for prefill and decode.",
)


def is_fused_kda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
) -> bool:
    # PPU MODIFICATION: begin
    from vllm_sail import envs as ppu_envs
    from vllm_sail.attention.pla_kda import get_pla_kda_kernel
    # PPU MODIFICATION: end
    if (
        num_heads not in (12, 24, 48, 96)
        or head_dim != 128
        or conv_width != 4
        or num_spec != 0
        or input_dtype != torch.bfloat16
        or conv_state_dtype != torch.bfloat16
        or is_conv_state_dim_first()
        # PPU MODIFICATION: begin
        or (not current_platform.is_ppu() and not hasattr(torch.ops._C, "fused_kda_decode"))
        # PPU MODIFICATION: end
    ):
        return False
    # SM90 is architecture-specific; SM10x and SM12x use family binaries.
    # PPU MODIFICATION: begin
    # PPU also supports fused_kda_decode.
    # PPU MODIFICATION: end
    return (
        # PPU MODIFICATION: begin
        (current_platform.is_ppu() and get_pla_kda_kernel("decode") is not None)
        or current_platform.is_device_capability(90)
        # PPU MODIFICATION: end
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    )


is_fused_kda_decode_supported = patch(
    _MODULE, "is_fused_kda_decode_supported", **_META
)(bind_body(is_fused_kda_decode_supported, _kda))


def is_flashkda_supported(
    head_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
) -> bool:
    # PPU MODIFICATION: begin
    from vllm_sail import envs as ppu_envs
    from vllm_sail.attention.pla_kda import get_pla_kda_kernel
    # PPU MODIFICATION: end
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return (
        # PPU MODIFICATION: begin
        (
            (current_platform.is_ppu() and get_pla_kda_kernel("prefill") is not None)
            or (capability is not None and capability.major in (9, 10, 12))
        )
        # PPU MODIFICATION: end
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )


is_flashkda_supported = patch(_MODULE, "is_flashkda_supported", **_META)(
    bind_body(is_flashkda_supported, _kda)
)


def _flashkda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # PPU MODIFICATION: begin
    # PPU MODIFICATION: begin
    from vllm_sail import envs as ppu_envs
    from vllm_sail.attention.pla_kda import get_pla_kda_kernel
    # PPU MODIFICATION: end
    if current_platform.is_ppu() and ppu_envs.VLLM_SAIL_USE_PLA:
        from pla.prefill.flashkdapro import flashkda_fwd

        logger.info_once("Using PLA KDA kernel: flashkda_fwd")
        out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
        final_state = torch.empty_like(initial_state)
        B, T, H, D = q.shape
        flashkda_fwd(
            q=q.view(B * T, H, D),
            k=k.view(B * T, H, D),
            v=v.view(B * T, H, D),
            g=g.view(B * T, H, D),
            beta=beta.view(B * T, H),
            A_log=A_log,
            dt_bias=dt_bias,
            out=out.view(B * T, H, D),
            scale=q.shape[-1] ** -0.5,
            lower_bound=lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
        )
        return out, final_state

    # PPU MODIFICATION: end
    import vllm._flashkda_C  # noqa: F401

    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    final_state = torch.empty_like(initial_state)
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(
            q.shape[0] * q.shape[1],
            q.shape[2],
            cu_seqlens.numel() - 1,
        ),
        dtype=torch.uint8,
        device=q.device,
    )
    # FlashKDA hardcodes dense Q/K/V/G strides. Beta may be row-strided because
    # FlashKDA materializes its transposed [H, T] layout internally.
    # TODO: Teach FlashKDA to consume beta in [T, H] layout directly instead
    # of transposing it to contiguous [H, T] storage internally.
    torch.ops._flashkda_C.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta,
        q.shape[-1] ** -0.5,
        out,
        workspace,
        A_log.contiguous(),
        dt_bias.view(-1, q.shape[-1]).contiguous(),
        lower_bound,
        initial_state.contiguous(),
        final_state,
        cu_seqlens.contiguous(),
    )
    return out, final_state


_flashkda_prefill = patch(_MODULE, "_flashkda_prefill", **_META)(
    bind_body(_flashkda_prefill, _kda)
)


def _forward(
    self,
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
) -> None:
    # PPU MODIFICATION: begin
    from vllm_sail import envs as ppu_envs
    from vllm_sail.attention.pla_kda import get_pla_kda_kernel
    # PPU MODIFICATION: end
    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    if attn_metadata_raw is None:
        return

    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
        fused_recurrent_kda_packed_decode,
    )

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata_narrowed = attn_metadata_raw[self.prefix]
    assert isinstance(attn_metadata_narrowed, KimiK3KDAMetadata)
    m = attn_metadata_narrowed
    has_initial_state = m.has_initial_state
    non_spec_query_start_loc = m.non_spec_query_start_loc
    non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
    spec_token_indx = m.spec_token_indx
    non_spec_token_indx = m.non_spec_token_indx
    spec_state_indices_tensor = m.spec_state_indices_tensor
    spec_query_start_loc = m.spec_query_start_loc
    num_accepted_tokens = m.num_accepted_tokens
    num_actual_tokens = m.num_actual_tokens
    has_spec_decode = m.num_spec_decodes > 0
    mixed_qkv = mixed_qkv[:num_actual_tokens]
    g1 = g1[:, :num_actual_tokens]
    beta = beta[:, :num_actual_tokens]

    conv_state, recurrent_state = self.kv_cache
    # The convolution kernels consume (..., dim, width - 1).
    if not is_conv_state_dim_first():
        conv_state = conv_state.transpose(-1, -2)

    if (
        self.decode_conv1d_weight is not None
        and self.decode_norm_weight is not None
        and not has_spec_decode
        and m.num_prefills == 0
        and m.num_decodes > 0
    ):
        assert non_spec_state_indices_tensor is not None
        # PPU MODIFICATION: begin

        if current_platform.is_ppu() and ppu_envs.VLLM_SAIL_USE_PLA:
            from pla.decode.kda import fused_kda_decode_mega_forward

            logger.info_once(
                "Using PLA KDA kernel: fused_kda_decode_mega_forward"
            )
            fused_kda_decode_mega_forward(
                x=mixed_qkv,
                weight=self.decode_conv1d_weight,
                bias=self.conv1d.bias,
                conv_state=conv_state,
                raw_g=g1,
                raw_beta=beta,
                a_log=self.A_log,
                dt_bias=self.dt_bias,
                state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
                state=recurrent_state,
                out=core_attn_out[:, :num_actual_tokens],
                lower_bound=self.gate_lower_bound,
                output_gate=g2[:num_actual_tokens],
                norm_weight=self.decode_norm_weight,
                norm_eps=self.o_norm.eps,
            )
            return

        # PPU MODIFICATION: end
        ops.fused_kda_decode(
            x=mixed_qkv,
            weight=self.decode_conv1d_weight,
            bias=self.conv1d.bias,
            conv_state=conv_state,
            raw_g=g1,
            raw_beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            state=recurrent_state,
            out=core_attn_out[:, :num_actual_tokens],
            lower_bound=self.gate_lower_bound,
            output_gate=g2[:num_actual_tokens],
            norm_weight=self.decode_norm_weight,
            norm_eps=self.o_norm.eps,
        )
        return

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )
    q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
        self.local_projection_size, dim=0
    )
    q_conv_state, k_conv_state, v_conv_state = conv_state.split(
        self.local_projection_size, dim=-2
    )

    # Separate multi-query speculative tokens from prefill/plain decode.
    if has_spec_decode:
        if m.num_prefills == 0 and m.num_decodes == 0:
            mixed_qkv_spec = mixed_qkv
            g1_spec, beta_spec = g1, beta
            mixed_qkv_ns = g1_ns = beta_ns = None
        else:
            assert spec_token_indx is not None
            assert non_spec_token_indx is not None
            mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
            g1_spec = g1.index_select(1, spec_token_indx)
            beta_spec = beta.index_select(1, spec_token_indx)
            mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
            g1_ns = g1.index_select(1, non_spec_token_indx)
            beta_ns = beta.index_select(1, non_spec_token_indx)
    else:
        mixed_qkv_spec = g1_spec = beta_spec = None
        mixed_qkv_ns, g1_ns, beta_ns = mixed_qkv, g1, beta

    # Spec-decode multi-query path.
    core_attn_out_spec = None
    if has_spec_decode:
        assert spec_state_indices_tensor is not None
        assert spec_query_start_loc is not None
        spec_conv_indices = spec_state_indices_tensor[:, 0][: m.num_spec_decodes]
        spec_max_query_len = spec_state_indices_tensor.size(-1)
        spec_conv_out = torch.empty_like(mixed_qkv_spec)
        mixed_qkv_spec = causal_conv1d_update(
            mixed_qkv_spec,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            activation="silu",
            conv_state_indices=spec_conv_indices,
            num_accepted_tokens=num_accepted_tokens,
            query_start_loc=spec_query_start_loc,
            max_query_len=spec_max_query_len,
            validate_data=False,
            out=spec_conv_out,
        )
        q_spec, k_spec, v_spec = (
            rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
            for x in mixed_qkv_spec.split(self.local_projection_size, dim=-1)
        )
        spec_cu_seqlens = spec_query_start_loc[: m.num_spec_decodes + 1]
        spec_out = (
            core_attn_out[:, : q_spec.shape[1]]
            if m.num_prefills == 0 and m.num_decodes == 0
            else None
        )
        core_attn_out_spec, _ = fused_recurrent_kda(
            q=q_spec,
            k=k_spec,
            v=v_spec,
            raw_g=g1_spec,
            raw_beta=beta_spec,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            initial_state=recurrent_state,
            cu_seqlens=spec_cu_seqlens,
            ssm_state_indices=spec_state_indices_tensor,
            num_accepted_tokens=num_accepted_tokens,
            out=spec_out,
        )

    # Prefill or plain-decode path.
    core_attn_out_non_spec = None
    if mixed_qkv_ns is not None:
        assert g1_ns is not None and beta_ns is not None
        if m.num_prefills > 0:
            q_ns, k_ns, v_ns = mixed_qkv_ns.split(
                self.local_projection_size, dim=-1
            )

            # Separate convolution calls accept row-strided packed inputs
            # and produce dense Q/K/V without an additional V copy.
            def _prefill_conv(
                x: torch.Tensor,
                state: torch.Tensor,
                weight: torch.Tensor,
            ) -> torch.Tensor:
                return causal_conv1d_fn(
                    x.transpose(0, 1),
                    weight,
                    None,
                    activation="silu",
                    conv_states=state,
                    has_initial_state=has_initial_state,
                    cache_indices=non_spec_state_indices_tensor,
                    query_start_loc=non_spec_query_start_loc,
                    metadata=m,
                ).transpose(0, 1)

            q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
            k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
            v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
            q_ns, k_ns, v_ns = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in (q_ns, k_ns, v_ns)
            )

            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            initial_state = gather_initial_states(
                recurrent_state,
                non_spec_state_indices_tensor,
                has_initial_state,
            )
            if self.kda_prefill_backend == "flashkda":
                assert self.gate_lower_bound is not None
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = _flashkda_prefill(
                    q=q_ns,
                    k=k_ns,
                    v=v_ns,
                    g=g1_ns,
                    beta=beta_ns,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=initial_state,
                    cu_seqlens=non_spec_query_start_loc,
                )
            else:
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_kda_with_fused_gate(
                    q=q_ns,
                    k=k_ns,
                    v=v_ns,
                    raw_g=g1_ns,
                    raw_beta=beta_ns,
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc,
                )
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        else:
            # Pure non-speculative decode.
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : mixed_qkv_ns.size(0)
            ]
            packed_conv_out = torch.empty_like(mixed_qkv_ns)
            mixed_qkv_ns = causal_conv1d_update(
                mixed_qkv_ns,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
                out=packed_conv_out,
            )
            (
                core_attn_out_non_spec,
                _,
            ) = fused_recurrent_kda_packed_decode(
                mixed_qkv=mixed_qkv_ns,
                raw_g=g1_ns,
                raw_beta=beta_ns,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                lower_bound=self.gate_lower_bound,
                initial_state=recurrent_state,
                state_indices=decode_conv_indices,
            )

    # Restore the scheduler's original token order for mixed batches.
    if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
        core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
    elif core_attn_out_non_spec is not None:
        # TODO: prefill and decode kernels write directly to core_attn_out
        core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
            0, :num_actual_tokens
        ]
    else:
        assert core_attn_out_spec is not None
    # Triton normalizes in place, so this is a self-copy with no device
    # work. Keep it for the out-of-place native implementation.
    core_attn_out.copy_(self.o_norm(core_attn_out, g2))


_forward = patch(_MODULE, "KimiK3DeltaAttention._forward", **_META)(
    _kda.eager_break_during_capture(bind_body(_forward, _kda))
)
