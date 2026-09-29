# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 BF16 eager backends using the optional SAIL MSA library."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
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
from vllm.v1.attention.backend import AttentionCGSupport

from vllm_sail import envs
from vllm_sail.attention.msa import Segment, main_kv_views, make_chunks, run_chunks

logger = init_logger(__name__)


@dataclass
class SAILIndexerMetadata(MiniMaxM3IndexerMetadata):
    sail_msa_chunks: list[list[Segment]] = field(default_factory=list)
    sail_msa_block_table: torch.Tensor | None = None


@dataclass
class SAILSparseMetadata(MiniMaxM3SparseMetadata):
    sail_msa_chunks: list[list[Segment]] = field(default_factory=list)
    sail_msa_block_table: torch.Tensor | None = None


def _schedule(builder, common):
    """Snapshot real eager lengths once per cache-group metadata build."""
    num_reqs = common.num_reqs
    starts = common.query_start_loc_cpu[: num_reqs + 1].tolist()
    lengths = common.seq_lens[:num_reqs].cpu().tolist()
    if starts[-1] != common.num_actual_tokens:
        raise ValueError("MSA query offsets disagree with num_actual_tokens")
    if common.block_table_tensor.shape[1] * 128 < max(lengths, default=0):
        raise ValueError("MSA block table is too short for the sequence lengths")
    return make_chunks(starts, lengths, builder.sail_index_heads, builder.sail_budget)


def _init_schedule(builder, config):
    sparse = config.model_config.hf_text_config.sparse_attention_config
    tp = config.parallel_config.tensor_parallel_size
    builder.sail_index_heads = max(1, sparse["sparse_num_index_heads"] // tp)
    builder.sail_budget = envs.VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_MEM_BUDGET_MB * 1024**2
    if builder.kv_cache_spec.block_size != 128:
        raise ValueError("SAIL MiniMax MSA requires 128-token cache pages")


class SAILIndexerMetadataBuilder(MiniMaxM3IndexerTritonMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        _init_schedule(self, vllm_config)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        return SAILIndexerMetadata(
            **vars(md),
            sail_msa_chunks=_schedule(self, common_attn_metadata),
            sail_msa_block_table=common_attn_metadata.block_table_tensor,
        )


class SAILSparseMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        _init_schedule(self, vllm_config)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        return SAILSparseMetadata(
            **vars(md),
            sail_msa_chunks=_schedule(self, common_attn_metadata),
            sail_msa_block_table=common_attn_metadata.block_table_tensor,
        )


class SAILIndexerBackend(MiniMaxM3IndexerBackend):
    @staticmethod
    def get_builder_cls():
        return SAILIndexerMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return SAILIndexerImpl


class SAILSparseBackend(MiniMaxM3SparseBackend):
    @staticmethod
    def get_builder_cls():
        return SAILSparseMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return SAILSparseImpl


class SAILIndexerImpl(MiniMaxM3IndexerImpl):
    indexer_backend_cls = SAILIndexerBackend

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

    def forward(self, index_query):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None, None
        md = metadata[self.index_cache.prefix]
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
        run_chunks(
            query=query,
            key=key,
            value=key,
            block_table=md.sail_msa_block_table,
            chunks=md.sail_msa_chunks,
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

    def forward(self, layer, query, kv_cache, output, *, query_fp8=None):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return output
        md = metadata[layer.layer_name]
        n = md.num_actual_tokens
        if not n:
            return output
        k, v = main_kv_views(kv_cache)
        run_chunks(
            query=query[:n].view(n, self.num_heads, 128),
            key=k,
            value=v,
            block_table=md.sail_msa_block_table,
            chunks=md.sail_msa_chunks,
            scale=self.scale,
            topk=layer.topk_indices_buffer,
            output=output[:n].view(n, self.num_heads, 128),
        )
        return output
