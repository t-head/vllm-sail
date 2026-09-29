# SPDX-License-Identifier: Apache-2.0
"""Adapter for the SAIL ``fmha_sm100`` package, loaded only on selection.

The first integration is BF16 and eager-only. Plans are consumed immediately:
some library versions return views into shared planner workspaces, so retaining
several plans without owning their workspaces would corrupt earlier plans.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

PAGE_SIZE = 128
TOPK = 16
MAX_K_TILES = 12288


def aligned_k_tiles(kv_len: int) -> int:
    return ((kv_len + 127) // 128 + 127) // 128 * 128


@dataclass(frozen=True)
class Segment:
    request: int
    start: int
    length: int
    kv_length: int


def make_chunks(
    query_starts: list[int],
    seq_lens: list[int],
    num_heads: int,
    budget_bytes: int,
) -> list[list[Segment]]:
    """Bound live FP32 scores plus the library's equally sized transpose buffer.

    Split even an individual long request. Its effective KV length ends at the
    chunk's final query, preserving bottom-right causal alignment. Tokens and
    logical block ids retain their original request-relative positions.
    """
    if len(query_starts) != len(seq_lens) + 1 or query_starts[0] != 0:
        raise ValueError("query offsets must start at zero and delimit every request")
    if num_heads <= 0 or budget_bytes <= 0:
        raise ValueError("MSA heads and memory budget must be positive")
    max_tiles = aligned_k_tiles(max(seq_lens, default=0))
    if max_tiles >= MAX_K_TILES:
        raise ValueError("SAIL MSA requires aligned KV tile count < 12288")
    capacity = budget_bytes // max(1, 2 * 4 * num_heads * max_tiles)
    if capacity < 1:
        raise ValueError("MSA score memory budget cannot hold one query token")
    chunks: list[list[Segment]] = []
    chunk: list[Segment] = []
    used = 0
    for request, kv_len in enumerate(seq_lens):
        start, stop = query_starts[request : request + 2]
        if not 0 <= stop - start <= kv_len:
            raise ValueError("invalid MSA query/KV lengths")
        prefix = kv_len - (stop - start)
        offset = 0
        while start + offset < stop:
            take = min(stop - start - offset, capacity - used)
            chunk.append(Segment(request, start + offset, take, prefix + offset + take))
            used += take
            offset += take
            if used == capacity:
                chunks.append(chunk)
                chunk, used = [], 0
    if chunk:
        chunks.append(chunk)
    return chunks


def validate_config(config, *, topk_blocks: int, indexer_kv_dtype: str) -> None:
    """Reject unsupported opt-in configurations before allocating side caches."""
    model = config.model_config
    text = model.hf_text_config
    sparse = text.sparse_attention_config
    if not model.enforce_eager:
        raise ValueError("SAIL MiniMax MSA currently requires --enforce-eager")
    if config.speculative_config is not None:
        raise ValueError("SAIL MiniMax MSA does not yet support speculative decoding")
    if str(model.dtype) != "torch.bfloat16":
        raise ValueError("SAIL MiniMax MSA currently requires --dtype bfloat16")
    if config.cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("SAIL MiniMax MSA currently requires BF16 main KV cache")
    if indexer_kv_dtype != "bf16":
        raise ValueError("SAIL MiniMax MSA currently requires indexer_kv_dtype=bf16")
    if topk_blocks != TOPK or sparse["sparse_block_size"] != PAGE_SIZE:
        raise ValueError(
            "SAIL MiniMax MSA requires topk_blocks=16 and sparse_block_size=128"
        )
    block_size = config.cache_config.block_size
    if block_size is not None and block_size != PAGE_SIZE:
        raise ValueError("SAIL MiniMax MSA requires --block-size 128")
    if sparse.get("sparse_score_type", "max") != "max":
        raise ValueError("SAIL MiniMax MSA requires sparse_score_type=max")
    if text.head_dim != 128 or sparse["sparse_index_dim"] != 128:
        raise ValueError("SAIL MiniMax MSA requires main and index head_dim=128")
    if sparse["sparse_num_index_heads"] != text.num_key_value_heads:
        raise ValueError("SAIL MiniMax MSA requires index heads == KV heads")
    if aligned_k_tiles(model.max_model_len) >= MAX_K_TILES:
        raise ValueError("SAIL MiniMax MSA requires aligned KV tile count < 12288")


class _JITLock:
    """One lock domain across every JIT entry, including nested calls."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a")
        self.thread_lock = threading.RLock()
        self.depth = 0

    @contextmanager
    def held(self):
        import fcntl

        with self.thread_lock:
            if self.depth == 0:
                fcntl.flock(self.file, fcntl.LOCK_EX)
            self.depth += 1
            try:
                yield
            finally:
                self.depth -= 1
                if self.depth == 0:
                    fcntl.flock(self.file, fcntl.LOCK_UN)

    def wrap(self, fn):
        memo = {}

        @functools.wraps(fn)
        def call(*args, **kwargs):
            try:
                key = (args, tuple(sorted(kwargs.items())))
                hash(key)
            except TypeError:
                key = None
            if key is not None and key in memo:
                return memo[key]
            with self.held():
                if key is not None and key in memo:
                    return memo[key]
                result = fn(*args, **kwargs)
                if key is not None:
                    memo[key] = result
                return result

        return call


def _install_jit_lock(api, jit) -> None:
    if getattr(jit, "_sail_msa_jit_lock", None) is not None:
        return
    entries = (
        "get_plan_fn",
        "get_prepare_metadata_fn",
        "get_sparse_topk_module",
        "get_reduction_module",
    )
    manager = jit._variant_manager
    # Resolve everything before changing either module; no partial installation.
    originals = {name: getattr(jit, name) for name in entries}
    variant = manager.get_variant
    lock = _JITLock(Path(jit.CACHE_BASE) / ".jit_cross_process.lock")
    wrapped_variant = lock.wrap(variant)
    manager.get_variant = wrapped_variant
    for mod in (jit, api):
        if hasattr(mod, "get_fmha_variant"):
            mod.get_fmha_variant = wrapped_variant
    for name, original in originals.items():
        wrapped = lock.wrap(original)
        setattr(jit, name, wrapped)
        if hasattr(api, name):
            setattr(api, name, wrapped)
    jit._sail_msa_jit_lock = lock


@functools.lru_cache(maxsize=1)
def load_msa():
    """Require the PPU package's HKT-score and paged-layout API, not SM100 vLLM."""
    try:
        msa = importlib.import_module("fmha_sm100")
        api = importlib.import_module("fmha_sm100.api")
        jit = importlib.import_module("fmha_sm100.jit")
        for name in ("fmha_sm100", "fmha_sm100_plan", "sparse_topk_select"):
            if not callable(getattr(msa, name)):
                raise TypeError(f"{name} is not callable")
        if not callable(api._detect_paged_kv_layout):
            raise TypeError("paged KV layout detection is missing")
        run_params = inspect.signature(api._fmha_sm100).parameters
        if not {"max_score", "kv_indices", "kv_block_indexes"} <= run_params.keys():
            raise TypeError("fmha_sm100 lacks paged sparse/OnlyScore arguments")
        select = inspect.signature(msa.sparse_topk_select).parameters
        if not {"num_valid_pages", "output"} <= select.keys():
            raise TypeError("sparse_topk_select lacks output/num_valid_pages")
        _install_jit_lock(api, jit)
    except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
        raise RuntimeError(
            "SAIL MiniMax MSA needs the PPU fmha_sm100 package (ppu_dev) with "
            "paged NHD/HND, HKT OnlyScore and output-buffer top-k support. "
            "Check the installed SAIL MSA build, or unset VLLM_SAIL_MINIMAX_M3_MSA."
        ) from exc
    return msa


def main_kv_views(kv_cache):
    """vLLM [pages,H,128,256] -> separate strided NHD [pages,128,H,128]."""
    if kv_cache.ndim != 4 or kv_cache.shape[2:] != (128, 256):
        raise ValueError("MSA expects vLLM main cache [pages, heads, 128, 256]")
    if kv_cache.shape[1] == 128 or kv_cache.stride(-1) != 1:
        raise ValueError("unsupported or ambiguous MSA main cache layout")
    k, v = kv_cache.split(128, dim=-1)
    return k.transpose(1, 2), v.transpose(1, 2)


def causal_pages(chunk, device):
    import torch

    return torch.cat(
        [
            (
                torch.arange(s.kv_length - s.length, s.kv_length, device=device) // 128
                + 1
            )
            for s in chunk
        ]
    ).to(torch.int32)


def force_local_scores(scores, pages, init_blocks: int, local_blocks: int) -> None:
    import torch

    tiles = torch.arange(scores.shape[1], device=scores.device)[None, :, None]
    limit = pages[None, None, :]
    scores.masked_fill_(tiles >= limit, -float("inf"))
    scores.masked_fill_((tiles < init_blocks) & (tiles < limit), 1e30)
    # The upstream Triton indexer gives local precedence on overlapping windows.
    scores.masked_fill_((tiles >= limit - local_blocks) & (tiles < limit), 1e29)


def sorted_blocks(topk, pages):
    """Ascending logical block ids; invalid entries (-1) always trail."""
    import torch

    invalid = (topk < 0) | (topk >= pages[:, None, None])
    sentinel = torch.iinfo(torch.int32).max
    values = topk.masked_fill(invalid, sentinel).sort(dim=-1).values
    return values.masked_fill(values == sentinel, -1).contiguous()


def run_chunks(
    *,
    query,
    key,
    value,
    block_table,
    chunks,
    scale,
    topk,
    output=None,
    init_blocks=0,
    local_blocks=0,
):
    """Run OnlyScore/top-k or sparse attend, consuming each eager plan immediately.

    ``topk`` is always the model's token-major shared output buffer. No plans or
    score tensors accumulate across chunks or layers. Main and index callers
    supply their own physical block tables.
    """
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("SAIL MiniMax MSA is eager-only; use --enforce-eager")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise ValueError("SAIL MiniMax MSA currently accepts BF16 Q/K/V only")
    if topk.dtype != torch.int32 or topk.shape[0] < query.shape[0]:
        raise ValueError("MSA requires a token-major int32 top-k buffer")
    if key.shape != value.shape or key.shape[1] != 128 or key.shape[-1] != 128:
        raise ValueError("MSA requires NHD K/V views [pages, 128, heads, 128]")
    msa = load_msa()
    only_score = output is None
    for chunk in chunks:
        begin, end = chunk[0].start, chunk[-1].start + chunk[-1].length
        qo_lens = torch.tensor([s.length for s in chunk], dtype=torch.int32)
        kv_lens = torch.tensor([s.kv_length for s in chunk], dtype=torch.int32)
        indices = (
            torch.cat(
                [block_table[s.request, : (s.kv_length + 127) // 128] for s in chunk]
            )
            .to(torch.int32)
            .contiguous()
        )
        # No later plan is built until this one has been consumed. The library's
        # shared eager planner workspace can therefore be safely reused.
        plan = msa.fmha_sm100_plan(
            qo_lens,
            kv_lens,
            query.shape[1],
            num_kv_heads=key.shape[2],
            page_size=128,
            kv_block_num=-1 if only_score else TOPK,
            num_kv_splits=-1,
            output_maxscore=only_score,
            causal=True,
            device=query.device,
        )
        pages = causal_pages(chunk, query.device)
        if only_score:
            scores = torch.full(
                (
                    query.shape[1],
                    aligned_k_tiles(max(s.kv_length for s in chunk)),
                    end - begin,
                ),
                -float("inf"),
                dtype=torch.float32,
                device=query.device,
            )
            msa.fmha_sm100(
                query[begin:end],
                key,
                value,
                plan,
                kv_indices=indices,
                output_o=False,
                output_maxscore=True,
                max_score=scores,
                sm_scale=scale,
            )
            force_local_scores(scores, pages, init_blocks, local_blocks)
            selected = topk[begin:end]
            msa.sparse_topk_select(
                scores, TOPK, num_valid_pages=scores.shape[1], output=selected
            )
            selected.masked_fill_(selected >= pages[:, None, None], -1)
            del scores
        else:
            blocks = sorted_blocks(topk[begin:end], pages)
            msa.fmha_sm100(
                query[begin:end],
                key,
                value,
                plan,
                kv_indices=indices,
                kv_block_indexes=blocks,
                out=output[begin:end],
                sm_scale=scale,
                output_maxscore=False,
            )
