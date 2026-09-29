# SPDX-License-Identifier: Apache-2.0
"""Route MiniMax's direct selectors to the opt-in SAIL MSA backends."""

import sys

from vllm.config import get_current_vllm_config
from vllm.models.minimax_m3.common import indexer, sparse_attention
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.attention.msa import load_msa, validate_config
from vllm_sail.patch.utils import patch

_INDEXER = "vllm.models.minimax_m3.common.indexer"
_ATTEND = "vllm.models.minimax_m3.common.sparse_attention"
_MODEL = "vllm.models.minimax_m3.nvidia.model"
_AFFECTED = ">=0.30.0,<0.31.0"
_REASON = (
    "MiniMax M3 selects indexer and attend implementations directly behind an "
    "SM100 gate, bypassing platform attention backend registration. PPU needs "
    "an explicit opt-in route to the SAIL MSA package."
)
_REMOVE = (
    "MiniMax M3 exposes platform registration hooks for both its indexer and "
    "main sparse-attention implementation selectors."
)
_original_indexer = indexer.select_indexer_impl_cls
_original_attend = sparse_attention.select_main_backend_and_impl_cls


def _enabled() -> bool:
    return current_platform.is_ppu() and envs.VLLM_SAIL_MINIMAX_M3_MSA


def _validate(topk_blocks, indexer_kv_dtype):
    validate_config(
        get_current_vllm_config(),
        topk_blocks=topk_blocks,
        indexer_kv_dtype=indexer_kv_dtype,
    )
    load_msa()


@patch(
    _INDEXER,
    "select_indexer_impl_cls",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE,
)
def select_indexer_impl_cls(*, topk_blocks, indexer_kv_dtype="bf16"):
    if not _enabled():
        return _original_indexer(
            topk_blocks=topk_blocks, indexer_kv_dtype=indexer_kv_dtype
        )
    _validate(topk_blocks, indexer_kv_dtype)
    from vllm_sail.attention.minimax_m3_msa import SAILIndexerImpl

    return SAILIndexerImpl


@patch(
    _ATTEND,
    "select_main_backend_and_impl_cls",
    reason=_REASON,
    affected_versions=_AFFECTED,
    remove_when=_REMOVE,
)
def select_main_backend_and_impl_cls(
    *,
    topk_blocks,
    kv_cache_dtype,
    num_kv_heads,
    emits_sparse_block_table=False,
):
    if _enabled():
        config = get_current_vllm_config()
        _validate(topk_blocks, config.attention_config.resolve_indexer_kv_dtype("bf16"))
        if envs.VLLM_SAIL_MINIMAX_M3_MSA_ATTEND:
            from vllm_sail.attention.minimax_m3_msa import (
                SAILSparseBackend,
                SAILSparseImpl,
            )

            return SAILSparseBackend, SAILSparseImpl
    return _original_attend(
        topk_blocks=topk_blocks,
        kv_cache_dtype=kv_cache_dtype,
        num_kv_heads=num_kv_heads,
        emits_sparse_block_table=emits_sparse_block_table,
    )


# Future imports see the patched provider. Repair a model imported before the
# general-plugin hook as well; do not import the heavy model just to patch it.
_model = sys.modules.get(_MODEL)
if (
    _model is not None
    and _model.select_main_backend_and_impl_cls is not select_main_backend_and_impl_cls
):
    if _model.select_main_backend_and_impl_cls is not _original_attend:
        raise RuntimeError("MiniMax M3 selector alias was replaced by another provider")
    patch(
        _MODEL,
        "select_main_backend_and_impl_cls",
        reason=_REASON,
        affected_versions=_AFFECTED,
        remove_when=_REMOVE,
    )(select_main_backend_and_impl_cls)
