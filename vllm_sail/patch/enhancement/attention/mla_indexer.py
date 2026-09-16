# SPDX-License-Identifier: Apache-2.0
"""PPU patches for ``vllm.v1.attention.backends.mla.indexer`` (DSA indexer).

The fork's changes, translated to plugin patches:

* Two module-level constants added by the fork (``PPU_DEEP_GEMM_TB_PER_CU``,
  ``PPU_FP4_INDEXER_ELEMENT_SIZE``) — installed with ``patch_value``.
* The fork's import switch (``vllm.utils.ppu_deep_gemm`` instead of
  ``vllm.utils.deep_gemm`` under ``is_ppu()``) affects two names.
  ``has_deep_gemm`` needs no patch: ``vllm_sail.utils.deep_gemm`` re-exports the
  very same object upstream already binds, so that half of the switch is an
  identity. ``get_paged_mqa_logits_metadata`` IS rebound on the target module
  at the plugin's PPU implementation (which accepts the PPU ``metadata_extra``
  tuple), the same shape as ``enhancement/deep_gemm_redirect.py``. On PPU the
  patched ``build`` below always takes the PPU branch, so the rebound name
  cannot reach a CUDA-only call signature.
* ``split_indexer_prefill_chunks`` gains a ``logits_dtype`` parameter. NOTE:
  this is UNCONDITIONAL in the fork (it also changes the stock default by
  clamping ``max_logits_elems`` to >= 1); reproduced as-in-fork and flagged in
  the report. Verbatim body with the fork's lines marked.
* ``DeepseekV32IndexerMetadataBuilder.__init__`` gains PPU branches for the fp4
  capability assert (sm_89), ``indexer_n_head``/``indexer_q_head_dim`` from the
  KV-cache spec, the SM count (``get_num_sms``), and the enlarged fp4
  scheduler-metadata buffer. Delegation cannot intercept the upstream assert,
  hence the verbatim body.
* ``DeepseekV32IndexerMetadataBuilder.build`` passes ``logits_dtype`` to the
  chunk splitter and builds PPU paged-MQA-logits scheduler metadata with the
  fork's ``metadata_extra`` tuple. Verbatim body with the fork's lines marked.

The copied bodies are rebased onto the target module's ``__dict__`` before
installation so every free name resolves exactly as it does upstream.
"""

from __future__ import annotations

import types

import torch
from vllm.v1.attention.backends.mla import indexer as _indexer_module
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder

import vllm_sail.utils.deep_gemm as _ppu_deep_gemm
from vllm_sail.patch.utils import patch, patch_value

_AFFECTED = ">=0.27.0,<0.28.0"
_MODULE = "vllm.v1.attention.backends.mla.indexer"


def _with_target_globals(fn):
    """Return ``fn`` with its globals rebased onto the patched target module.

    A verbatim upstream body references the target module's globals (``torch``,
    ``envs``, ``current_platform``, private helpers, ...). A function defined
    here would instead resolve them against this patch module; rebasing keeps
    the body byte-faithful and makes it see the names this file rebinds on the
    target module (``get_paged_mqa_logits_metadata``,
    ``split_indexer_prefill_chunks``, ...).
    """
    return types.FunctionType(
        fn.__code__,
        _indexer_module.__dict__,
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )


patch_value(
    _MODULE,
    "PPU_DEEP_GEMM_TB_PER_CU",
    8,
    allow_missing=True,
    reason=(
        "Fork constant: thread blocks per compute unit used by PPU DeepGEMM "
        "when sizing the fp4 indexer scheduler-metadata buffer."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream defines the constant itself (patch_value becomes a no-op).",
)

patch_value(
    _MODULE,
    "PPU_FP4_INDEXER_ELEMENT_SIZE",
    1,
    allow_missing=True,
    reason=(
        "Fork constant: element size (bytes) of the PPU fp4 indexer cache, "
        "passed to the PPU paged MQA logits metadata builder via metadata_extra."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream defines the constant itself (patch_value becomes a no-op).",
)

# Fork's import switch: on PPU the indexer takes its DeepGEMM helpers from the
# PPU DeepGEMM wrapper. Only get_paged_mqa_logits_metadata needs rebinding —
# the plugin's wrapper re-exports the very same has_deep_gemm object upstream
# already binds, so that half of the switch is an identity. The patched build()
# below resolves the rebound name through module globals at call time.
patch(
    _MODULE,
    "get_paged_mqa_logits_metadata",
    reason=(
        "Fork's is_ppu() import switch selects the PPU DeepGEMM wrapper's "
        "get_paged_mqa_logits_metadata, which accepts the PPU metadata_extra "
        "tuple. The CUDA helper has no such parameter."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "vllm.utils.deep_gemm dispatches on current_platform internally, or "
        "upstream accepts a DeepGEMM backend registry."
    ),
)(_ppu_deep_gemm.get_paged_mqa_logits_metadata)


def _split_indexer_prefill_chunks_body(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
    # PPU MODIFICATION: begin
    logits_dtype: torch.dtype = torch.float32,
    # PPU MODIFICATION: end
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    # PPU MODIFICATION: begin
    bytes_per_elem = torch.empty((), dtype=logits_dtype).element_size()
    max_logits_elems = max(1, max_logits_bytes // bytes_per_elem)
    # PPU MODIFICATION: end
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else max(1, chunk_m)
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


patch(
    _MODULE,
    "split_indexer_prefill_chunks",
    reason=(
        "Fork adds a logits_dtype parameter so the fp4 indexer cache (bf16 "
        "logits) chunks against the correct logits budget. UNCONDITIONAL in the "
        "fork: the default path also changes (max_logits_elems clamped to >= 1); "
        "reproduced as-in-fork and flagged in the Phase-4 report."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream adds the logits_dtype parameter itself.",
)(_with_target_globals(_split_indexer_prefill_chunks_body))


def _builder_init_body(self, *args, block_table_width: int, **kwargs) -> None:
    # PPU MODIFICATION: begin
    # This copied body is defined outside a class, so it has no __class__
    # closure for zero-argument super(). Resolve the owner in target globals.
    super(DeepseekV32IndexerMetadataBuilder, self).__init__(*args, **kwargs)
    # PPU MODIFICATION: end
    scheduler_config = self.vllm_config.scheduler_config
    parallel_config = self.vllm_config.parallel_config
    self.dcp_world_size = parallel_config.decode_context_parallel_size
    self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
    self.pcp_world_size = parallel_config.prefill_context_parallel_size
    self.use_pcp = self.pcp_world_size > 1
    self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
    # The DCP sparse-indexer code is parameterized by interleave size, but
    # interleave > 1 is not yet validated end-to-end (gsm8k parity fails),
    # so fail closed here rather than silently produce wrong output.
    if self.dcp_world_size > 1 and self.cp_kv_cache_interleave_size > 1:
        raise NotImplementedError(
            "DCP sparse indexer currently supports only "
            f"cp_kv_cache_interleave_size=1 (got "
            f"{self.cp_kv_cache_interleave_size})."
        )
    # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
    self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
    self.num_speculative_tokens = (
        self.vllm_config.speculative_config.num_speculative_tokens
        if self.vllm_config.speculative_config
        else 0
    )
    self.use_fp4_indexer_cache = (
        self.vllm_config.attention_config.use_fp4_indexer_cache
    )

    # PPU MODIFICATION: begin
    if self.use_fp4_indexer_cache and current_platform.is_ppu():
        self.indexer_n_head = self.kv_cache_spec.indexer_n_head
        self.indexer_q_head_dim = self.kv_cache_spec.indexer_q_head_dim
    else:
        self.indexer_n_head = 0
        self.indexer_q_head_dim = 0

    # NOTE(kai): ppu fp4 indexer cache requires sm_89
    if current_platform.is_ppu():
        assert (
            current_platform.is_device_capability(89)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires PPUs "
            "sm_89 and "
            "earlier architectures are not supported."
        )
    else:
        assert (
            current_platform.is_device_capability_family(100)
            or not self.use_fp4_indexer_cache
        ), (
            "use_fp4_indexer_cache requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )
    # PPU MODIFICATION: end

    next_n = self.num_speculative_tokens + 1
    self.decode_threshold = next_n
    self.reorder_batch_threshold = None
    # NOTE: SM100 datacenter GPUs support any next_n natively via the
    # multi-atom paged MQA logits kernels (FP8 and FP4 indexer
    # caches). Outside the SM100 family the FP8
    # paged MQA logits kernel only supports next_n in (1, 2)
    # (deepgemm smxx_fp8_fp4_paged_mqa_logits.hpp:233), so flatten there.
    self.use_flattening = not current_platform.is_device_capability_family(
        100
    ) and next_n not in (1, 2)
    logger.info_once(
        "DSA indexer decode path: use_flattening=%s "
        "(next_n=%d, use_fp4_indexer_cache=%s)",
        self.use_flattening,
        next_n,
        self.use_fp4_indexer_cache,
    )

    # PPU MODIFICATION: begin
    if current_platform.is_ppu():
        from vllm_sail.utils.deep_gemm import get_num_sms

        sm_count = get_num_sms()
    else:
        sm_count = num_compute_units(self.device.index)
    # PPU MODIFICATION: end
    self.num_sms = sm_count

    self.offsets_buffer = torch.arange(
        next_n, device=self.device, dtype=torch.int32
    )
    self.decode_lens_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens,),
        dtype=torch.int32,
        device=self.device,
    )
    # Shared workspace for decode seq_lens. Native MTP views this as
    # (B, max_decode_len) at runtime, keeping context_lens contiguous even
    # when max_decode_len is smaller than next_n.
    self.decode_seq_lens_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens,),
        dtype=torch.int32,
        device=self.device,
    )
    self.global_decode_seq_lens_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens,),
        dtype=torch.int32,
        device=self.device,
    )
    self.arange_buffer = torch.arange(
        max(
            scheduler_config.max_num_seqs * next_n,
            scheduler_config.max_num_batched_tokens,
        ),
        dtype=torch.int32,
        device=self.device,
    )
    self.expanded_block_table_buffer = torch.zeros(
        (scheduler_config.max_num_batched_tokens, block_table_width),
        dtype=torch.int32,
        device=self.device,
    )

    # See: DeepGEMM/csrc/apis/attention.hpp
    # PPU MODIFICATION: begin
    if current_platform.is_ppu() and self.use_fp4_indexer_cache:
        # NOTE(kai): fp4 paged using extra metadata
        # (num_sms * tb_per_cu + 1, 2)
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms * PPU_DEEP_GEMM_TB_PER_CU + 1, 2), dtype=torch.int32, device=self.device
        )
    else:
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )
    # PPU MODIFICATION: end

    # KV compression. Default to 1 for no compression.
    self.compress_ratio = 1
    # Get compress_ratio for DeepseekV4 support
    if isinstance(self.kv_cache_spec, MLAAttentionSpec):
        self.compress_ratio = self.kv_cache_spec.compress_ratio
    if self.dcp_world_size > 1 and self.compress_ratio > 1:
        raise NotImplementedError(
            "DCP is not supported with sparse indexer KV compression "
            f"(compress_ratio={self.compress_ratio})."
        )

    # Pre-allocate buffers for CUDA graph compatibility when
    if self.compress_ratio > 1:
        # compress_ratio > 1 (DeepseekV4)
        # Compressed slot mapping output buffer
        self.compressed_slot_mapping_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int64,
            device=self.device,
        )
        # Buffer for compressed seq_lens in decode path
        self.expanded_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.__init__",
    reason=(
        "PPU branches from the fork: fp4 indexer cache allowed on sm_89, "
        "indexer_n_head/indexer_q_head_dim read from the KV-cache spec on PPU, "
        "SM count from PPU DeepGEMM's get_num_sms, and the enlarged "
        "num_sms * PPU_DEEP_GEMM_TB_PER_CU scheduler-metadata buffer for fp4 "
        "paged logits. Delegation cannot intercept the upstream capability "
        "assert, hence the verbatim body with only the fork's changes marked."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream grows PPU branches in the DSA indexer metadata builder.",
)(_with_target_globals(_builder_init_body))


def _build_body(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> DeepseekV32IndexerMetadata:
    num_reqs = common_attn_metadata.num_reqs
    num_tokens = common_attn_metadata.num_actual_tokens
    query_start_loc = common_attn_metadata.query_start_loc
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    seq_lens = common_attn_metadata.seq_lens
    slot_mapping = common_attn_metadata.slot_mapping
    block_table = common_attn_metadata.block_table_tensor
    dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

    num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
        split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.decode_threshold,
            require_uniform=not self.use_flattening,
            treat_short_extends_as_decodes=not self.use_pcp,
        )
    )

    assert num_decodes + num_prefills == num_reqs
    assert num_decode_tokens + num_prefill_tokens == num_tokens

    compressed_slot_mapping = slot_mapping
    compressed_seq_lens = seq_lens
    if self.compress_ratio > 1:
        padded_num_tokens = num_tokens
        if self.pcp_world_size > 1:
            padded_num_tokens = slot_mapping.shape[0] // self.pcp_world_size
        compressed_slot_mapping = get_compressed_slot_mapping(
            num_tokens,
            query_start_loc,
            seq_lens,
            block_table,
            self.kv_cache_spec.storage_block_size,
            self.compress_ratio,
            out=self.compressed_slot_mapping_buffer,
        )
        if self.pcp_world_size > 1:
            compressed_slot_mapping = get_pcp_group().all_gather(
                self.compressed_slot_mapping_buffer[:padded_num_tokens],
                dim=0,
            )
        compressed_seq_lens = seq_lens // self.compress_ratio

    prefill_metadata = None
    if num_prefills > 0:
        # This CPU value is an upper bound for async-spec extend rows.  It
        # is safe for chunking/allocation because CUDA metadata below is
        # built from exact device seq_lens and gather ignores the tail.
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        compressed_seq_lens_cpu = (
            seq_lens_cpu // self.compress_ratio
            if self.compress_ratio > 1
            else seq_lens_cpu
        )
        prefill_query_lens_cpu = torch.diff(
            query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
        )
        max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        # Upper bound is exact for prefill rows (the `[num_decodes:]`
        # slice below).
        assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
        chunk_specs = split_indexer_prefill_chunks(
            compressed_seq_lens_cpu[num_decodes:],
            prefill_query_lens_cpu,
            self.max_prefill_buffer_size,
            max_logits_bytes,
            request_offset=num_decodes,
            # PPU MODIFICATION: begin
            logits_dtype=torch.bfloat16 if self.use_fp4_indexer_cache else torch.float32,
            # PPU MODIFICATION: end
        )

        chunks = []
        for req_slice, query_slice in chunk_specs:
            metadata = build_prefill_chunk_metadata(
                req_slice.start,
                req_slice.stop,
                query_start_loc,
                query_start_loc_cpu,
                seq_lens,
                compressed_seq_lens,
                compressed_seq_lens_cpu,
                common_attn_metadata.block_table_tensor,
                self.compress_ratio,
                query_slice=query_slice,
                skip_kv_gather=query_slice.start > 0,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )
            # Skip when total_seq_lens is 0 (i.e., no compressed token).
            if metadata is not None:
                chunks.append(metadata)
        prefill_metadata = DeepseekV32IndexerPrefillMetadata(chunks)

    decode_metadata = None
    if num_decodes > 0:
        torch.diff(
            common_attn_metadata.query_start_loc[: num_decodes + 1],
            out=self.decode_lens_buffer[:num_decodes],
        )
        decode_lens = self.decode_lens_buffer[:num_decodes]
        decode_lens_cpu = torch.diff(
            common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
        )

        # Under DCP the per-token decode bounds must be localized AFTER the
        # per-token expansion below, not before. Expanding from a
        # request-level localized length subtracts decode offsets in local
        # space and yields too-short bounds (e.g. world=2, rank=1, global
        # per-token bounds [8, 9, 10] -> [3, 4, 5] instead of [4, 4, 5]), so
        # the first decode token would run top-k against too short a local KV
        # range and miss valid tokens. Keep the global seq_lens here and
        # localize the expanded bounds further down.
        global_seq_lens_for_decode: torch.Tensor | None = None
        if dcp_local_seq_lens is not None:
            global_seq_lens_for_decode = common_attn_metadata.seq_lens[:num_decodes]
        seq_lens = common_attn_metadata.seq_lens[:num_decodes]
        block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

        max_decode_len = int(decode_lens_cpu.max().item())
        next_n = 1 + self.num_speculative_tokens
        use_native = not self.use_flattening and max_decode_len <= next_n

        global_seq_lens_for_decode = self._prepare_global_decode_seq_lens(
            global_seq_lens=global_seq_lens_for_decode,
            decode_lens=decode_lens,
            decode_lens_cpu=decode_lens_cpu,
            query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
            num_decode_tokens=num_decode_tokens,
            use_native=use_native,
            max_decode_len=max_decode_len,
        )

        seq_lens, block_table, decode_lens, batch_size, requires_padding = (
            self._prepare_decode_tensors(
                seq_lens=seq_lens,
                block_table=block_table,
                decode_lens=decode_lens,
                decode_lens_cpu=decode_lens_cpu,
                query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                use_native=use_native,
                next_n=next_n,
                max_decode_len=max_decode_len,
            )
        )

        seq_lens_is_buffer_view = (use_native and next_n > 1) or (
            not use_native and max_decode_len > 1
        )

        # DCP: localize the now-expanded per-token global bounds to this
        # rank's owned KV. Done here (after expansion) so each token's global
        # causal length is localized individually; see the comment above.
        if dcp_local_seq_lens is not None:
            seq_lens = self._dcp_localize_decode_seq_lens(
                seq_lens, num_decodes, seq_lens_is_buffer_view
            )

        # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
        # compressed tokens. Convert uncompressed seq_lens to compressed.
        if self.compress_ratio > 1:
            if seq_lens_is_buffer_view:
                seq_lens //= self.compress_ratio
            else:
                # Copy to avoid mutating shared state; keeps CG address stable.
                self.expanded_seq_lens_buffer[:num_decodes] = (
                    seq_lens // self.compress_ratio
                )
                self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

        # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
        # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
        # kernels see the same (B, next_n) layout as the MTP path.
        if seq_lens.dim() == 1:
            seq_lens = seq_lens.unsqueeze(-1)

        # PPU MODIFICATION: begin
        metadata_extra = None
        if current_platform.is_ppu() and self.indexer_n_head > 0:
            metadata_extra = (
                next_n,
                self.indexer_n_head,
                self.indexer_q_head_dim,
                PPU_FP4_INDEXER_ELEMENT_SIZE,
            )
        scheduler_metadata = None
        n = self.scheduler_metadata_buffer.shape[0]
        if current_platform.is_ppu() and has_deep_gemm():
            scheduler_metadata = get_paged_mqa_logits_metadata(
                seq_lens,
                self.kv_cache_spec.storage_block_size,
                self.num_sms,
                metadata_extra,
            )
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata_buffer[:n] = scheduler_metadata
        # PPU MODIFICATION: end
        # DeepGEMM is required for the paged MQA logits on CUDA devices
        # PPU MODIFICATION: begin
        # Fork demotes this to an elif behind the PPU branch above.
        elif current_platform.is_cuda() and has_deep_gemm():
            # PPU MODIFICATION: end
            self.scheduler_metadata_buffer[:] = get_paged_mqa_logits_metadata(
                seq_lens,
                self.kv_cache_spec.storage_block_size,
                self.num_sms,
            )

        decode_metadata = DeepSeekV32IndexerDecodeMetadata(
            block_table=block_table,
            seq_lens=seq_lens,
            decode_lens=decode_lens,
            requires_padding=requires_padding,
            # PPU MODIFICATION: begin
            # Fork slices [:n]; identical to the full buffer on the CUDA path.
            schedule_metadata=self.scheduler_metadata_buffer[:n],
            # PPU MODIFICATION: end
            global_seq_lens=global_seq_lens_for_decode,
        )

    attn_metadata = DeepseekV32IndexerMetadata(
        seq_lens=common_attn_metadata.seq_lens,
        max_seq_len=common_attn_metadata.max_seq_len,
        slot_mapping=compressed_slot_mapping,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        prefill=prefill_metadata,
        decode=decode_metadata,
    )

    return attn_metadata


patch(
    _MODULE,
    "DeepseekV32IndexerMetadataBuilder.build",
    reason=(
        "Fork's DSA indexer build path for PPU: pass logits_dtype to the "
        "prefill chunk splitter, and build PPU paged-MQA-logits scheduler "
        "metadata with the metadata_extra tuple (next_n, indexer_n_head, "
        "indexer_q_head_dim, PPU_FP4_INDEXER_ELEMENT_SIZE), slicing the "
        "scheduler buffer to the returned size. Verbatim body with only the "
        "fork's changes marked."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream grows PPU branches in the DSA indexer build path.",
)(_with_target_globals(_build_body))
