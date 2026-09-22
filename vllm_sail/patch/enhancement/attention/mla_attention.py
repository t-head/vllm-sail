# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.model_executor.layers.attention.mla_attention``.

Two of the fork's three changes are patched here; both are verbatim method
copies (upstream 4bdc8a788) with only the fork's changed lines marked:

* ``MLAAttention.forward_impl`` skips the profile-run scratch allocation on PPU
  ("PPU FIXME: WA for deepseek dp + ep OOM on PPU, need investigate later.").
  Delegation cannot reach an allocation inside the method, hence the body copy.
* ``MLACommonBaseImpl._compute_prefill_context`` skips the ``kv_c_normed.to()``
  cast when the kv_b_proj weights are int8 ("PPU FIXME: WA for assert fail when
  run dpsk-r1 int8 with acext"). NOTE: this change has no is_ppu() guard in the
  fork — it is an UNCONDITIONAL upstream behaviour change (stock CUDA with int8
  weights no longer casts), reproduced as-in-fork and flagged in the report.

The fork's third change (``self.aot_schedule = current_platform.is_cuda() or
current_platform.is_ppu()`` in ``MLACommonMetadataBuilder.__init__``) is NOT
patched: PPU declares ``PlatformEnum.CUDA``, so ``is_cuda()`` is already True
and the fork's addition is a no-op in the plugin.

The copied bodies are rebased onto the target module's ``__dict__`` before
installation so every free name resolves exactly as it does upstream.
"""

from __future__ import annotations

import types

from vllm.model_executor.layers.attention import mla_attention as _mla_attention

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.30.0,<0.31.0"
_MODULE = "vllm.model_executor.layers.attention.mla_attention"


def _with_target_globals(fn):
    """Return ``fn`` with its globals rebased onto the patched target module.

    A verbatim upstream body references the target module's globals (``torch``,
    ``current_platform``, private helpers, ...). A function defined here would
    instead resolve them against this patch module; rebasing keeps the body
    byte-faithful and immune to upstream import-list changes.
    """
    return types.FunctionType(
        fn.__code__,
        _mla_attention.__dict__,
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


def _forward_impl_body(
    self,
    q: torch.Tensor,
    k_c_normed: torch.Tensor,  # key in unified attn
    k_pe: torch.Tensor,  # value in unified attn
    kv_cache: torch.Tensor,
    attn_metadata: MLACommonMetadata,
    output: torch.Tensor,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    quant_group_size: int | None = None,
    quant_scale_ue8m0: bool | None = None,
    quant_col_major: bool | None = None,
    quant_tma_aligned: bool | None = None,
    q_dcp_replicated: torch.Tensor | None = None,
) -> torch.Tensor:
    assert output is not None, "Output tensor must be provided."

    quant_key = _detect_output_quant_key(
        output, output_scale, output_block_scale, self.num_heads * self.v_head_dim
    )
    if quant_key is not None:
        # The fusion pass has allocated output with quantized dtype
        # (FP8 or uint8 for FP4). We can't write into it directly,
        # so we swap in a temp buffer for computation, then quantize
        # into the real output at the end.
        # NOTE(carlyou): this is temporary until kernels support fp8 output
        quant_output = output
        output = torch.empty(
            output.shape[0],
            self.num_heads * self.v_head_dim,
            dtype=q.dtype,
            device=output.device,
        )

    if attn_metadata is None:
        # During the profile run try to simulate to worse case output size
        # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
        # since this can be large
        # PPU MODIFICATION: begin
        # PPU FIXME: WA for deepseek dp + ep OOM on PPU, need investigate later.
        if not current_platform.is_ppu():
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )
        # PPU MODIFICATION: end

        # The zero fill is required when used with DP + EP
        # to ensure all ranks within a DP group compute the
        # same expert outputs.
        if quant_key is not None:
            return quant_output.fill_(0)
        return output.fill_(0)

    if self.impl.dcp_world_size == -1:
        self.impl.dcp_world_size = get_dcp_group().world_size

    fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)

    num_actual_toks = attn_metadata.num_actual_tokens
    if self.use_pcp and self.impl.dcp_world_size > 1 and quant_key is not None:
        raise NotImplementedError(
            "MRV2 MLA PCP+DCP does not support fused output quantization yet."
        )

    # Inputs and outputs may be padded for CUDA graphs
    output_padded = output
    output = output[:num_actual_toks, ...]
    q = q[:num_actual_toks, ...]
    if q_dcp_replicated is not None:
        q_dcp_replicated = q_dcp_replicated[:num_actual_toks, ...]
    k_c_normed = k_c_normed[:num_actual_toks, ...]
    k_pe = k_pe[:num_actual_toks, ...]

    if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":
        kv_cache = kv_cache.view(current_platform.fp8_dtype())

    assert (
        attn_metadata.num_decodes is not None
        and attn_metadata.num_prefills is not None
        and attn_metadata.num_decode_tokens is not None
    )
    num_mqa_tokens = attn_metadata.num_decode_tokens
    num_mha_tokens = q.size(0) - num_mqa_tokens

    if self.impl.is_sparse and num_mha_tokens > 0:
        prefill = getattr(attn_metadata, "prefill", None)
        use_dense_mha = getattr(prefill, "use_dense_mha", False)
        prefill_max_seq_len = attn_metadata.prefill_max_seq_len  # type: ignore[attr-defined]
        use_masked_mha = (
            self.prefill_backend is not None
            and self.impl.masked_mha_available  # type: ignore[attr-defined]
            and self.impl.dcp_world_size <= 1
            and prefill is not None
            and _use_masked_mha(
                backend_name=self.attn_backend.get_name(),
                tensor_parallel_size=self._vllm_config.parallel_config.tensor_parallel_size,
                query_len=prefill.max_query_len,
                seq_len=prefill_max_seq_len,
            )
        )
        use_mha = (use_dense_mha or use_masked_mha) and not (
            self._vllm_config.attention_config.sparse_mla_force_mqa
        )
        if not use_mha:
            num_mqa_tokens = q.size(0)
            num_mha_tokens = 0

    mha_use_quant_output = (
        quant_key is not None
        and self.prefill_backend is not None
        and self.prefill_backend.supports_quant_output(quant_key)
        and (
            not self.impl.is_sparse
            or attn_metadata.prefill_max_seq_len  # type: ignore[attr-defined]
            <= attn_metadata.topk_tokens  # type: ignore[attr-defined]
        )
        and attn_metadata is not None
        and attn_metadata.prefill is not None
        and attn_metadata.prefill.chunked_context is None
        and self.impl.dcp_world_size <= 1
    )

    if num_mha_tokens > 0:
        if mha_use_quant_output:
            mha_output = quant_output
            mha_output_scale = output_scale
        else:
            mha_output = output
            mha_output_scale = None

        self.impl.forward_mha(  # type: ignore[attr-defined]
            q[num_mqa_tokens:],
            k_c_normed[num_mqa_tokens:],
            k_pe[num_mqa_tokens:],
            kv_cache,
            attn_metadata,
            self._k_scale,
            output=mha_output[num_mqa_tokens:num_actual_toks],
            output_scale=mha_output_scale,
        )

    if num_mqa_tokens > 0:
        if q_dcp_replicated is not None:
            mqa_q = q_dcp_replicated[:num_mqa_tokens]
            qrep_decode = True
        else:
            mqa_q = q[:num_mqa_tokens]
            qrep_decode = False
        mqa_output_slice = output[:num_mqa_tokens]

        mqa_q_nope, mqa_q_pe = mqa_q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # Convert from (B, N, P) to (N, B, P)
        mqa_q_nope = mqa_q_nope.transpose(0, 1)

        if self.q_pad_num_heads is not None:
            B, N, L = mqa_q_pe.shape
            mqa_pe_padded = mqa_q_pe.new_empty((B, self.q_pad_num_heads, L))
            mqa_pe_padded.resize_((B, N, L))
            mqa_pe_padded.copy_(mqa_q_pe)
            mqa_q_pe = mqa_pe_padded

        if self.is_aiter_triton_fp4_bmm_enabled:
            from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

            mqa_ql_nope = batched_gemm_a16wfp4(
                mqa_q_nope,
                self.W_K,
                self.W_K_scale,
                transpose_bm=True,
                prequant=True,
                y_scale=self._q_scale if fp8_attention else None,
            )
        elif self.is_aiter_triton_fp8_bmm_enabled:
            # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
            mqa_ql_nope = rocm_aiter_ops.triton_fp8_bmm(
                mqa_q_nope,
                self.W_K,
                self.W_K_scale,
                group_size=128,
                transpose_bm=True,
            )
        else:
            # Pads the head_dim if necessary (for the underlying kernel)
            N, B, P = mqa_q_nope.shape
            W_UK_T = self.W_UK_T_dcp_qrep if qrep_decode else self.W_UK_T
            assert W_UK_T is not None
            _, _, L = W_UK_T.shape

            if self.q_pad_num_heads is not None:
                mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))
                mqa_ql_nope.resize_((N, B, L))
            else:
                mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))

            # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
            torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)

            # Convert from (N, B, L) to (B, N, L)
            mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

        if fp8_attention and self.impl.supports_quant_query_input:
            assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]
            assert mqa_ql_nope.shape[1] == mqa_q_pe.shape[1]
            mqa_q = self._decode_concat_quant_fp8_op(
                mqa_ql_nope, mqa_q_pe, self._q_scale
            )
        else:
            mqa_q = (mqa_ql_nope, mqa_q_pe)
        # concatenate nope + pe -> (B, N, L + P) (fp8 op above may have fused)
        if self.impl.dcp_world_size > 1:
            if self.use_pcp:
                if self.impl.dcp_world_size > self.impl.pcp_world_size:
                    if isinstance(mqa_q, tuple):
                        mqa_q = torch.cat(mqa_q, dim=-1)
                    mqa_q = get_tp_group().all_gather(mqa_q, dim=1)
            else:
                if isinstance(mqa_q, tuple):
                    # concatenate mqa_ql_nope and mqa_q_pe -> (B, N, L + P)
                    mqa_q = torch.cat(mqa_q, dim=-1)
                if not qrep_decode:
                    # mqa_q do allgather in head dim.
                    mqa_q = get_dcp_group().all_gather(mqa_q, dim=1)

        # call decode attn
        if not self.impl.is_sparse:
            assert attn_metadata.decode is not None
        attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)  # type: ignore[attr-defined]

        # correct dcp attn_out with lse.
        if self.impl.dcp_world_size > 1:
            assert lse is not None
            if self.dcp_a2a:
                attn_out = dcp_a2a_lse_reduce(
                    attn_out,
                    lse,
                    get_dcp_group(),
                    is_lse_base_on_e=self.impl.lse_base_on_e,
                )
            elif self.use_pcp:
                attn_out = cp_lse_ag_out_ar(
                    attn_out,
                    lse,
                    get_dcp_group(),
                    is_lse_base_on_e=self.impl.lse_base_on_e,
                )
            else:
                attn_out = cp_lse_ag_out_rs(
                    attn_out,
                    lse,
                    get_dcp_group(),
                    is_lse_base_on_e=self.impl.lse_base_on_e,
                )
            if self.use_pcp:
                attn_out = finalize_mla_pcp_decode(attn_out, self.num_heads)

        # v_up projection
        self._v_up_proj(attn_out, out=mqa_output_slice)

    if quant_key is not None:
        quant_idx = num_mqa_tokens if mha_use_quant_output else num_actual_toks
        if quant_idx == 0:
            return quant_output
        actual = output[:quant_idx]
        if quant_key == kNvfp4Dynamic:
            # NVFP4: two FP4 values packed into one uint8
            assert output_block_scale is not None
            fp4_data, fp4_scales = ops.scaled_fp4_quant(actual, output_scale)
            quant_output[:quant_idx].copy_(fp4_data)
            output_block_scale[: fp4_scales.shape[0]].copy_(fp4_scales)
        elif quant_key in (kFp8Dynamic128Sym, kFp8Dynamic64Sym):
            # Per-group FP8
            assert output_block_scale is not None
            assert quant_group_size is not None, (
                "Group FP8 output quant requested but "
                "quant_group_size not passed through custom op"
            )
            finfo = torch.finfo(_FP8_DTYPE)
            torch.ops._C.per_token_group_fp8_quant(
                actual,
                quant_output[:quant_idx],
                output_block_scale[:quant_idx],
                quant_group_size,
                1e-10,  # eps
                finfo.min,
                finfo.max,
                quant_scale_ue8m0,
                quant_col_major,
                quant_tma_aligned,
            )
        elif quant_key == kFp8StaticTensorSym:
            # Static FP8 quantization
            fp8_data, _ = self._quant_fp8_op(actual, output_scale)
            quant_output[:quant_idx].copy_(fp8_data)
        else:
            raise ValueError(f"Unsupported quant_key: {quant_key}")
        return quant_output

    if self.use_pcp and output_padded.shape[0] > num_actual_toks:
        output_padded[num_actual_toks:].zero_()
    return output_padded


patch(
    _MODULE,
    "MLAAttention.forward_impl",
    reason=(
        "PPU FIXME: WA for deepseek dp + ep OOM on PPU — the fork skips the "
        "profile-run torch.empty scratch allocation in forward_impl. The "
        "allocation is mid-method, so this is a verbatim upstream body with "
        "only the fork's change marked."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the PPU OOM workaround is root-caused upstream-side, or upstream stops "
        "allocating the profile-run scratch buffer."
    ),
)(_with_target_globals(_forward_impl_body))


def _compute_prefill_context_body(
    self,
    q: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    attn_metadata: MLACommonMetadata,
    k_scale: torch.Tensor,
):
    assert attn_metadata.prefill is not None
    prefill_metadata = attn_metadata.prefill
    assert prefill_metadata.prefill_backend is not None
    assert prefill_metadata.chunked_context is not None

    use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()

    output = None
    merge_output = None
    iters = len(prefill_metadata.chunked_context.seq_tot)
    workspace = prefill_metadata.chunked_context.workspace

    if use_fp8_prefill:
        q = q.to(prefill_metadata.q_data_type)

    for i in range(iters):
        toks = prefill_metadata.chunked_context.seq_tot[i]
        if self.kv_cache_dtype == "fp8_ds_mla":
            ops.cp_gather_and_upconvert_fp8_kv_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace[:toks],
                block_table=prefill_metadata.block_table,
                workspace_starts=prefill_metadata.chunked_context.cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )
        elif not use_fp8_prefill:
            ops.gather_and_maybe_dequant_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                token_to_seq=prefill_metadata.chunked_context.token_to_seq[i],
                num_tokens=prefill_metadata.chunked_context.chunk_total_token[i],
                kv_cache_dtype=self.kv_cache_dtype,
                scale=k_scale,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )
        else:
            # FP8 path: gather cache without dequantization
            ops.cp_gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )

        # Extract kv_c_normed from workspace
        kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
        # When FP8 weights are used without FP8 prefill, kv_b_proj expects
        # model dtype input and will quantize internally.
        # For quantized layers (AWQ/GPTQ) that lack a .weight attribute,
        # use params_dtype which is the expected input dtype.
        _kv_b_proj_w_dtype = (
            self.kv_b_proj.weight.dtype
            if hasattr(self.kv_b_proj, "weight")
            else self.kv_b_proj.params_dtype
        )
        # For NVFP4, weights are packed uint8 — keep input in model dtype
        # since the NVFP4 linear layer quantizes internally.
        if (
            use_fp8_prefill or _kv_b_proj_w_dtype != current_platform.fp8_dtype()
        ) and _kv_b_proj_w_dtype != torch.uint8:
            # PPU MODIFICATION: begin
            # PPU FIXME: WA for assert fail when run dpsk-r1 int8 with acext
            if _kv_b_proj_w_dtype != torch.int8:
                kv_c_normed = kv_c_normed.to(self.kv_b_proj.weight.dtype)
            # PPU MODIFICATION: end

        k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )

        # To Do: Use epilogue of kv_b_proj to generate fp8 kv_nope.
        if use_fp8_prefill:
            kv_nope = kv_nope.to(prefill_metadata.q_data_type)
            k_pe = k_pe.to(prefill_metadata.q_data_type)
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        attn_output, attn_softmax_lse = (
            prefill_metadata.prefill_backend.run_prefill_context_chunk(
                chunk_idx=i,
                q=q,
                k=k,
                v=v,
            )
        )
        if prefill_metadata.chunked_context.has_empty_context[i]:
            mask_empty_context(
                attn_softmax_lse,
                attn_output,
                prefill_metadata.query_start_loc,
                prefill_metadata.chunked_context.cu_seq_lens[i],
            )

        if output is None:
            output = attn_output
            output_lse = attn_softmax_lse
        else:
            if merge_output is None:
                merge_output = torch.empty_like(output)
                merge_output_lse = torch.empty_like(output_lse)
            merge_attn_states(
                output=merge_output,
                output_lse=merge_output_lse,
                prefix_output=output,
                prefix_lse=output_lse,
                suffix_output=attn_output,
                suffix_lse=attn_softmax_lse,
            )
            output, merge_output = merge_output, output
            output_lse, merge_output_lse = merge_output_lse, output_lse

    return output, output_lse


patch(
    _MODULE,
    "MLACommonBaseImpl._compute_prefill_context",
    reason=(
        "PPU FIXME: WA for assert fail when run dpsk-r1 int8 with acext — the "
        "fork skips the kv_c_normed dtype cast for int8 kv_b_proj weights. "
        "UNCONDITIONAL in the fork (no is_ppu guard), so it also changes stock "
        "CUDA int8 behaviour; reproduced as-in-fork and flagged in the Phase-4 "
        "report. Verbatim upstream body with only the fork's change marked."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "the PPU int8/acext assert is root-caused, or upstream changes the "
        "kv_c_normed casting logic this patch copies."
    ),
)(_with_target_globals(_compute_prefill_context_body))
