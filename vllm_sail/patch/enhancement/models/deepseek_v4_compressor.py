# ruff: noqa: E731, W291, W293, UP037
# ruff: noqa: F821
# Copied bodies resolve globals in their upstream module through bind_body.
# SPDX-License-Identifier: Apache-2.0
"""Keep DeepSeek V4 compression on Triton on PPU (both architectures)."""

from __future__ import annotations

from vllm.models.deepseek_v4 import compressor as _compressor

from vllm_sail.patch.bodies import bind_body
from vllm_sail.patch.utils import patch


def forward(
    self,
    # [num_tokens, 2 * self.coff * self.head_dim]
    kv_score: torch.Tensor,
    # [num_tokens]
    positions: torch.Tensor,
    rotary_emb,
) -> None:
    # Each of shape [num_tokens, coff * self.head_dim]
    # input bf16, output are fp32
    kv, score = kv_score.split(
        [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
    )

    # Get the metadata and handle dummy profiling run.
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if not isinstance(attn_metadata, dict):
        return

    state_metadata = cast(CompressorMetadata, attn_metadata[self.state_cache.prefix])
    token_to_req_indices = state_metadata.token_to_req_indices
    slot_mapping = state_metadata.slot_mapping
    num_actual = slot_mapping.shape[0]
    block_table = state_metadata.block_table
    block_size = state_metadata.block_size

    # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
    state_cache = self.state_cache.kv_cache
    # kv_state stored in first half, score_state stored in second half
    state_width = state_cache.shape[-1] // 2
    pdl_kwargs = (
        {}
        if current_platform.is_rocm() or current_platform.is_xpu()
        else {"launch_pdl": False}
    )

    # Store the KV and score (with fused APE addition) in the state.
    # NOTE: PDL is disabled — both this kernel and the compress kernels
    # below depend on preceding kernel outputs (kv/score from the cublas
    # GEMM; state_cache from this kernel) but neither emits/waits on PDL
    # grid dependency primitives, so launch_pdl=True caused a
    # read-after-write race and non-deterministic output.
    _SAVE_PARTIAL_STATES_KERNEL(
        kv=kv,
        score=score,
        ape=self.ape,
        positions=positions,
        state_cache=state_cache,
        slot_mapping=slot_mapping,
        block_size=block_size,
        state_width=state_width,
        compress_ratio=self.compress_ratio,
        pdl_kwargs=pdl_kwargs,
    )

    # full graph cannot branch on per-step CPU metadata after capture
    if (
        current_platform.is_cuda()
        and self.head_dim == 512
        and self.compress_ratio == 128
        and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
        and state_metadata.c128_boundary is False
    ):
        return

    # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
    # RoPE requirements (kernel applies forward GPT-J style rotation):
    # - is_neox_style=False (interleaved pairs, NOT split-half)
    # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
    #   second half sin (per-pair, length rope_head_dim // 2 each)
    # - applied to LAST rope_head_dim elements of head_dim
    # - position used: (positions // compress_ratio) * compress_ratio
    cos_sin_cache = rotary_emb.cos_sin_cache
    k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
    k_cache_layer = self._static_forward_context[self.k_cache_prefix]
    kv_cache = k_cache_layer.kv_cache

    # Plain-row V4 reads a contiguous bf16 / per-tensor fp8 cache row; the
    # fp8_ds_mla path uses the UE8M0 paged uint8 layout.
    store_full_kv = self.head_dim == 512 and kv_cache.dtype != torch.uint8
    store_full_fp8 = kv_cache.dtype == torch.float8_e4m3fn
    fp8_scale = (
        getattr(k_cache_layer, "_flashinfer_fp8_kv_scale", None)
        if store_full_fp8
        else None
    )

    # cutedsl (head=512) accepts the full-cache flags; triton (indexer/AMD)
    # does not, so the two callables have different signatures.
    compress_norm_rope_store_fn: Any
    # PPU MODIFICATION: begin
    if (
        current_platform.is_cuda()
        and not current_platform.is_ppu()
        and self.head_dim == 512
    ):
        # PPU MODIFICATION: end
        from .nvidia.ops.sparse_attn_compress_cutedsl import (
            _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL,
        )

        # head=512 on CUDA always uses cutedsl, for both the fp8_ds_mla
        # layout and the plain full-cache layout. The full-cache flags
        # are consumed only here.
        compress_norm_rope_store_fn = _SPARSE_ATTN_COMPRESSOR_CUTEDSL_KERNEL
        extra_kwargs: dict[str, Any] = dict(
            store_full_kv=store_full_kv,
            store_full_fp8=store_full_fp8,
            fp8_scale=fp8_scale,
        )
    elif self._use_two_stage_fused_compressor:
        # head=512 cr>=128 (no overlap): two-pass split compressor on the
        # prefill suffix, single-pass on the decode prefix.
        assert state_metadata.num_decode_tokens is not None
        compress_norm_rope_store_fn = compress_norm_rope_store_two_stage_triton
        extra_kwargs = {
            "num_decode_tokens": state_metadata.num_decode_tokens,
            "compress_scratch": self._compress_scratch,
        }
    else:
        # Indexer path (head_dim == 128) or non-CUDA GPUs (AMD, XPU, etc.).
        compress_norm_rope_store_fn = compress_norm_rope_store_triton
        extra_kwargs = {}

    compress_norm_rope_store_fn(
        state_cache=state_cache,
        num_actual=num_actual,
        token_to_req_indices=token_to_req_indices,
        positions=positions,
        slot_mapping=slot_mapping,
        block_table=block_table,
        block_size=block_size,
        state_width=state_width,
        cos_sin_cache=cos_sin_cache,
        kv_cache=kv_cache,
        k_cache_metadata=k_cache_metadata,
        pdl_kwargs=pdl_kwargs,
        head_dim=self.head_dim,
        rope_head_dim=self.rope_head_dim,
        compress_ratio=self.compress_ratio,
        overlap=self.overlap,
        use_fp4_cache=self.use_fp4_cache,
        rms_norm_weight=self.norm.weight,
        rms_norm_eps=self.rms_norm_eps,
        quant_block=self._quant_block,
        token_stride=self._token_stride,
        scale_dim=self._scale_dim,
        **extra_kwargs,
    )


patch(
    "vllm.models.deepseek_v4.compressor",
    "DeepseekCompressor.forward",
    reason=(
        "The upstream head-512 CUDA branch imports CuTeDSL directly; PPU "
        "shares CUDA identity but requires the Triton compressor. No "
        "compressor backend registration hook is available."
    ),
    affected_versions=">=0.30.0,<0.31.0",
    remove_when="DeepseekCompressor selects compression through a backend capability hook.",
)(bind_body(forward, _compressor))
