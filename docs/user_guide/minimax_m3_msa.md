# MiniMax M3 SAIL MSA verification

The optional SAIL MSA path replaces MiniMax M3's indexer and main sparse
attention with the PPU `fmha_sm100` package. It is disabled by default and
requires device qualification before use in production.

The initial path supports BF16 Q/K/V, BF16 index K, FP32 index scores,
128-token pages, 128-dimensional main/index heads, top-16 blocks, and eager
execution. Index heads must equal KV heads. Speculative decoding, FP8 caches,
and graph capture are rejected when this path is selected.

## Prepare the environment

Use vLLM 0.30 and install vLLM SAIL following the
[installation guide](../getting_started/installation.md). Install the SAIL
build of `fmha_sm100` (`ppu_dev`) in the same environment. The NVIDIA package
or vLLM's separate `vllm.third_party.fmha_sm100` interface is not a substitute.
The adapter requires paged NHD/HND detection, HKT score output, and top-k
selection with `num_valid_pages` and an `output` buffer. Missing or incompatible
APIs fail explicitly on selection.

Check where the process imports both packages:

```bash
python -c 'import vllm_sail, fmha_sm100; print(vllm_sail.__file__); print(fmha_sm100.__file__)'
```

## Run correctness checks

From the plugin checkout in that prepared environment:

```bash
python -m pytest tests/ut/test_minimax_m3_msa.py tests/ut/test_minimax_m3_msa_tensors.py -q
python -m pytest tests/e2e/tier_b/test_minimax_m3_msa.py -q -rs
python -m pytest tests/e2e/tier_b/test_minimax_m3.py -q -rs
```

The first command checks configuration, selector installation, scheduling,
and tensor plumbing. Its numerical adapter tests use a library double, not
SAIL kernels. The second runs real MSA score, top-k, and attention kernels
against a PyTorch reference, including mixed query lengths, partial pages,
long-request chunking, separate main/index physical page tables, and strided
views of both cache layouts. The third checks the existing native fused
norm/RoPE/cache writer. A skip is not device evidence.

The main K/V adapter splits vLLM's `[pages, heads, 128, 256]` cache into strided
NHD K/V views. No whole-cache copy is performed. The device test must establish
that the installed SAIL library honors those strides; its Python API probe
alone cannot establish this.

For runtime patch maintenance, use the same real vLLM environment:

```bash
python tools/patch_manifest.py
python tools/patch_manifest.py --check
python tools/check_patch_drift.py
```

New selector targets require a drift-baseline refresh after reviewing the
reported source changes; do not accept unrelated upstream drift blindly.

## Compare model execution

Keep the model, weights, TP size, prompts and sampling parameters identical
across three runs. Add these arguments to the existing model launch:

```bash
--enforce-eager --dtype bfloat16 --kv-cache-dtype auto --block-size 128 \
--attention-config '{"indexer_kv_dtype":"bf16"}'
```

Select each run using environment variables:

| Run | `VLLM_SAIL_MINIMAX_M3_MSA` | `VLLM_SAIL_MINIMAX_M3_MSA_ATTEND` |
| --- | --- | --- |
| Existing upstream selection | `0` | `0` |
| MSA indexer, existing attend | `1` | `0` |
| MSA indexer and attend | `1` | `1` |

Check the worker logs for `MiniMax M3 indexer selected SAIL MSA` and, in the
third run, `MiniMax M3 attention selected SAIL MSA`. Test single-request decode,
concurrent requests, chunked prefill, prefix-cache reuse, and TP worker startup.
Report accuracy and model execution separately from operator test results.

`VLLM_SAIL_MINIMAX_M3_MSA_INDEXER_MEM_BUDGET_MB` defaults to `256` and bounds the
live score tensor plus the library's equally sized top-k transpose workspace.
It does not bound all MSA planning, output, allocator, or KV-cache memory.
Requests longer than a chunk are split while retaining their causal offsets.
All three variables accept corresponding `VLLM_PPU_*` aliases; SAIL names win.

This first path consumes eager plans immediately, rather than retaining plans
that may alias the library's shared workspace. It also serializes JIT entries
across worker processes. Performance tuning, persistent isolated plans, FP8,
and captured decode require subsequent verification. See the
[verification guide](verification_guide.md) for the overall evidence boundaries.
