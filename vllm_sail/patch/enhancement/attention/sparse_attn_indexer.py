# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.model_executor.layers.sparse_attn_indexer``.

The fork makes three changes to ``SparseAttnIndexer``:

* ``__init__`` warns instead of raising when DeepGEMM is missing on PPU (the
  op falls back to a less efficient PyTorch path); upstream raises on CUDA,
  which PPU inherits via ``PlatformEnum.CUDA``. Verbatim body with markers —
  delegation cannot intercept the upstream raise.
* ``forward_native`` redirects to ``forward_ppu`` on PPU before the CUDA
  branch. Verbatim body with markers (the fork's exact insertion point).
* Adds ``forward_ppu``, which calls the PPU ``ppu_sparse_attn_indexer`` custom
  op (registered by ``vllm_sail.ops``). Additive, installed with
  ``allow_missing=True``; the enabled-custom-op path reaches it through the
  ``custom_op_dispatch`` patch.

The fork's module-level ``from vllm._ppu_ops import ppu_ops`` is what
``vllm_sail.ops`` replaces; importing it here ensures the custom op is
registered before any forward call.
"""

from __future__ import annotations

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import has_deep_gemm
from vllm.utils.torch_utils import _encode_layer_name

import vllm_sail.ops  # noqa: F401  (registers torch.ops.vllm.ppu_sparse_attn_indexer)
from vllm_sail.patch.utils import patch
from vllm_sail.utils.deep_gemm import is_deep_gemm_supported

_AFFECTED = ">=0.30.0,<0.31.0"
_MODULE = "vllm.model_executor.layers.sparse_attn_indexer"

logger = init_logger(__name__)


@patch(
    _MODULE,
    "SparseAttnIndexer.__init__",
    reason=(
        "PPU tolerates a missing DeepGEMM in SparseAttnIndexer with a warning "
        "(falling back to a less efficient PyTorch path), while upstream raises "
        "on CUDA. PPU inherits the CUDA raise via PlatformEnum.CUDA, so the "
        "fork's PPU branch must run first. Delegation cannot intercept the "
        "upstream raise, hence the verbatim body."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream downgrades the missing-DeepGEMM error to a warning, or PPU always ships DeepGEMM.",
)
def sparse_attn_indexer_init(
    self,
    k_cache,
    quant_block_size: int,
    scale_fmt: str,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    max_total_seq_len: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool = False,
    use_fp4_cache: bool = False,
):
    CustomOp.__init__(self)
    self.k_cache = k_cache
    self.quant_block_size = quant_block_size
    self.scale_fmt = scale_fmt
    self.topk_tokens = topk_tokens
    self.head_dim = head_dim
    self.max_model_len = max_model_len
    self.max_total_seq_len = max_total_seq_len
    self.topk_indices_buffer = topk_indices_buffer
    self.skip_k_cache_insert = skip_k_cache_insert
    self.use_fp4_cache = use_fp4_cache
    self.dense_mha_metadata_layer_name = ""
    # DCP scalars are constant for the run; resolve them here (config is set
    # during model construction) and pass them into the custom op, rather
    # than threading them through per-step metadata.
    parallel_config = get_current_vllm_config().parallel_config
    self.dcp_world_size = parallel_config.decode_context_parallel_size
    self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
    self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
    self.use_pcp = parallel_config.prefill_context_parallel_size > 1
    # PPU MODIFICATION: begin
    # Fork's PPU branch: warn (PyTorch fallback) instead of raising when
    # DeepGEMM is missing; the upstream CUDA raise moves to an elif.
    if current_platform.is_ppu() and not is_deep_gemm_supported():
        logger.warning_once(
            "DeepGEMM is not supported or available. SparseAttnIndexer will use a "
            "less efficient PyTorch implementation. "
            "Please make sure you have the required hardware and software setup "
            "for DeepGEMM to achieve optimal performance."
        )
    elif current_platform.is_cuda() and not has_deep_gemm():
        # PPU MODIFICATION: end
        raise RuntimeError(
            "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
            "the current vLLM environment."
        )


@patch(
    _MODULE,
    "SparseAttnIndexer.forward_native",
    reason=(
        "Fork redirects SparseAttnIndexer.forward_native to forward_ppu on PPU "
        "before the CUDA branch, so the disabled-custom-op path also uses the "
        "PPU kernel. Verbatim upstream body with the fork's insertion marked."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream's forward_native grows a platform hook, or PPU stops overriding this op.",
)
def sparse_attn_indexer_forward_native(
    self,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    k: torch.Tensor,
    weights: torch.Tensor,
):
    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        return self.forward_ppu(hidden_states, q_quant, k, weights)
    # PPU MODIFICATION: end
    if current_platform.is_cuda() or current_platform.is_xpu():
        return self.forward_cuda(hidden_states, q_quant, k, weights)
    elif current_platform.is_rocm():
        return self.forward_hip(hidden_states, q_quant, k, weights)
    else:
        raise NotImplementedError(
            "SparseAttnIndexer native forward is only implemented for "
            "CUDA, ROCm and XPU platforms."
        )


@patch(
    _MODULE,
    "SparseAttnIndexer.forward_ppu",
    allow_missing=True,
    reason=(
        "PPU implementation of the sparse attention indexer, calling the PPU "
        "ppu_sparse_attn_indexer custom op registered by vllm_sail.ops. Upstream "
        "has no forward_ppu; installed additively and reached through the "
        "custom_op_dispatch patch (enabled path) or the forward_native redirect "
        "(disabled path)."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream adds a forward_ppu hook and the PPU op moves into the dispatch ladder.",
)
def sparse_attn_indexer_forward_ppu(
    self,
    hidden_states: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    k: torch.Tensor,
    weights: torch.Tensor,
):
    if isinstance(q_quant, tuple):
        q_values, q_scale = q_quant
    else:
        q_values, q_scale = q_quant, None
    return torch.ops.vllm.ppu_sparse_attn_indexer(
        hidden_states,
        _encode_layer_name(self.k_cache.prefix),
        self.k_cache.kv_cache,
        q_values,
        q_scale,
        k,
        weights,
        self.quant_block_size,
        self.scale_fmt,
        self.topk_tokens,
        self.head_dim,
        self.max_model_len,
        self.max_total_seq_len,
        self.topk_indices_buffer,
        self.skip_k_cache_insert,
        self.use_fp4_cache,
    )
