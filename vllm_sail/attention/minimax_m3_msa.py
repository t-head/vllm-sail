# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 BF16 eager backends using the optional SAIL MSA library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerBackend,
    MiniMaxM3IndexerImpl,
    MiniMaxM3IndexerMetadata,
    MiniMaxM3IndexerTritonMetadataBuilder,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_sail import envs
from vllm_sail.attention.msa import (
    MSAChunkMetadata,
    main_kv_views,
    make_chunks,
    prepare_chunks,
    run_indexer,
    run_sparse_attention,
)

logger = init_logger(__name__)


@dataclass
class SAILIndexerMetadata(MiniMaxM3IndexerMetadata):
    msa_chunks: list[MSAChunkMetadata] = field(default_factory=list)


@dataclass
class SAILSparseMetadata(MiniMaxM3SparseMetadata):
    msa_chunks: list[MSAChunkMetadata] = field(default_factory=list)


def _build_msa_chunks(
    common: CommonAttentionMetadata,
    num_index_heads: int,
    budget_bytes: int,
) -> list[MSAChunkMetadata]:
    """Snapshot real eager lengths once per cache-group metadata build."""
    num_reqs = common.num_reqs
    starts = common.query_start_loc_cpu[: num_reqs + 1].tolist()
    lengths = common.seq_lens[:num_reqs].cpu().tolist()
    if starts[-1] != common.num_actual_tokens:
        raise ValueError("MSA query offsets disagree with num_actual_tokens")
    if common.block_table_tensor.shape[1] * 128 < max(lengths, default=0):
        raise ValueError("MSA block table is too short for the sequence lengths")
    segments = make_chunks(starts, lengths, num_index_heads, budget_bytes)
    return prepare_chunks(segments, common.block_table_tensor)


class SAILIndexerMetadataBuilder(MiniMaxM3IndexerTritonMetadataBuilder):
    """Extend common MiniMax metadata with PPU MSA inputs shared across layers."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.msa_budget_bytes = (
            envs.VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_MEM_BUDGET_MB * 1024**2
        )
        if kv_cache_spec.block_size != 128:
            raise ValueError("SAIL MiniMax MSA requires 128-token cache pages")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SAILIndexerMetadata:
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        return SAILIndexerMetadata(
            **vars(md),
            msa_chunks=_build_msa_chunks(
                common_attn_metadata, self.num_index_heads, self.msa_budget_bytes
            ),
        )


class SAILSparseMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # The supported configuration has one index head per KV head. Use the
        # rank-local cache spec rather than re-deriving the TP head partition.
        self.num_index_heads = kv_cache_spec.num_kv_heads
        self.msa_budget_bytes = (
            envs.VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_MEM_BUDGET_MB * 1024**2
        )
        if kv_cache_spec.block_size != 128:
            raise ValueError("SAIL MiniMax MSA requires 128-token cache pages")

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SAILSparseMetadata:
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        return SAILSparseMetadata(
            **vars(md),
            msa_chunks=_build_msa_chunks(
                common_attn_metadata, self.num_index_heads, self.msa_budget_bytes
            ),
        )


class SAILIndexerBackend(MiniMaxM3IndexerBackend):
    @staticmethod
    def get_builder_cls() -> type[SAILIndexerMetadataBuilder]:
        return SAILIndexerMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[SAILIndexerImpl]:
        return SAILIndexerImpl


class SAILSparseBackend(MiniMaxM3SparseBackend):
    @staticmethod
    def get_builder_cls() -> type[SAILSparseMetadataBuilder]:
        return SAILSparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[SAILSparseImpl]:
        return SAILSparseImpl


class SAILIndexerImpl(MiniMaxM3IndexerImpl):
    indexer_backend_cls: ClassVar[type[SAILIndexerBackend]] = SAILIndexerBackend

    def __init__(self, **kwargs):
        if kwargs["index_head_dim"] != 128:
            raise ValueError("SAIL MiniMax MSA requires index_head_dim=128")
        if kwargs["num_index_heads"] != kwargs["num_kv_heads"]:
            raise ValueError(
                "SAIL MiniMax MSA requires index heads == KV heads per rank"
            )
        if kwargs.get("topk_indices_buffer") is None:
            raise ValueError(
                "SAIL MiniMax MSA needs vLLM's shared token-major top-k buffer"
            )
        super().__init__(**kwargs)
        logger.info_once("MiniMax M3 indexer selected SAIL MSA (BF16, eager)")

    def forward(
        self,
        index_query: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None, None
        md = metadata[self.index_cache.prefix]
        assert isinstance(md, SAILIndexerMetadata)
        n = md.num_actual_tokens
        if not n:
            return None, None
        query = index_query[:n].view(n, self.num_index_heads, 128)
        cache = self.index_cache.kv_cache
        if cache.ndim != 3 or cache.shape[1:] != (128, 128):
            raise ValueError("MSA index cache must have shape [pages, 128, 128]")
        key = cache.unsqueeze(2)  # NHD [pages,128,1,128]; OnlyScore never reads V.
        topk = self.topk_indices_buffer
        if (
            topk.shape[0] < n
            or topk.shape[1:] != (self.num_index_heads, 16)
            or topk.dtype != torch.int32
            or not topk.is_contiguous()
        ):
            raise ValueError("MSA top-k buffer must be token-major int32 [T,H,16]")
        run_indexer(
            query=query,
            key=key,
            chunks=md.msa_chunks,
            scale=self.scale,
            topk=topk,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
        )
        return None, None


class SAILSparseImpl(MiniMaxM3SparseImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.head_size != 128 or self.num_kv_heads == 128:
            raise ValueError("unsupported SAIL MSA head size/count")
        logger.info_once("MiniMax M3 attention selected SAIL MSA (BF16, eager)")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return output
        md = metadata[layer.layer_name]
        assert isinstance(md, SAILSparseMetadata)
        n = md.num_actual_tokens
        if not n:
            return output
        k, v = main_kv_views(kv_cache)
        run_sparse_attention(
            query=query[:n].view(n, self.num_heads, 128),
            key=k,
            value=v,
            chunks=md.msa_chunks,
            scale=self.scale,
            topk=layer.topk_indices_buffer,
            output=output[:n].view(n, self.num_heads, 128),
        )
        return output
