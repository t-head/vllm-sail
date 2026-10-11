# SPDX-License-Identifier: Apache-2.0
"""MSA routing, lifecycle and scheduling tests requiring neither torch nor vLLM."""

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm_sail.attention import msa
from vllm_sail.patch.utils import PATCH_MARKER, PATCH_REGISTRY, patch


def config():
    return NS(
        model_config=NS(
            enforce_eager=True,
            dtype="torch.bfloat16",
            max_model_len=8192,
            hf_text_config=NS(
                head_dim=128,
                num_key_value_heads=4,
                sparse_attention_config={
                    "sparse_block_size": 128,
                    "sparse_index_dim": 128,
                    "sparse_num_index_heads": 4,
                },
            ),
        ),
        cache_config=NS(block_size=128, cache_dtype="auto"),
        speculative_config=None,
        attention_config=NS(resolve_indexer_kv_dtype=lambda default: default),
    )


def test_long_request_is_split_with_correct_causal_offsets():
    # 2 heads * 128 tiles * 4 bytes * score+transpose = 2048 bytes/token.
    chunks = msa.make_chunks([0, 1, 1, 11], [129, 0, 267], 2, 3 * 2048)
    flat = [s for c in chunks for s in c]
    assert [(s.request, s.start, s.length, s.kv_length) for s in flat] == [
        (0, 0, 1, 129),
        (2, 1, 2, 259),
        (2, 3, 3, 262),
        (2, 6, 3, 265),
        (2, 9, 2, 267),
    ]
    assert all(sum(s.length for s in c) <= 3 for c in chunks)
    assert [t for s in flat for t in range(s.start, s.start + s.length)] == list(
        range(11)
    )


@pytest.mark.parametrize(
    "args",
    [
        ([1, 2], [2], 1, 4096),
        ([0, 3], [2], 1, 4096),
        ([0, 1], [128], 0, 4096),
        ([0, 1], [128], 2, 1),
        ([0, 1], [12288 * 128], 1, 1 << 30),
    ],
)
def test_invalid_schedules_fail(args):
    with pytest.raises(ValueError):
        msa.make_chunks(*args)


def test_zero_tokens_and_tile_boundaries():
    assert msa.make_chunks([0, 0], [0], 1, 1024) == []
    assert [msa.aligned_k_tiles(n) for n in (1, 128, 16384, 16385)] == [
        128,
        128,
        128,
        256,
    ]


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("enforce_eager", False, "enforce-eager"),
        ("dtype", "torch.float16", "bfloat16"),
        ("max_model_len", 12288 * 128, "tile count"),
    ],
)
def test_configuration_rejects_unsupported_model(field, value, message):
    cfg = config()
    setattr(cfg.model_config, field, value)
    with pytest.raises(ValueError, match=message):
        msa.validate_config(cfg, topk_blocks=16, indexer_kv_dtype="bf16")


def test_configuration_rejects_fp8_spec_and_wrong_pages():
    cfg = config()
    msa.validate_config(cfg, topk_blocks=16, indexer_kv_dtype="bf16")
    for target, attr, value in (
        (cfg, "speculative_config", object()),
        (cfg.cache_config, "cache_dtype", "fp8_e4m3"),
        (cfg.cache_config, "block_size", 16),
    ):
        old = getattr(target, attr)
        setattr(target, attr, value)
        with pytest.raises(ValueError):
            msa.validate_config(cfg, topk_blocks=16, indexer_kv_dtype="bf16")
        setattr(target, attr, old)
    with pytest.raises(ValueError, match="indexer_kv_dtype"):
        msa.validate_config(cfg, topk_blocks=16, indexer_kv_dtype="fp8")
    with pytest.raises(ValueError, match="topk_blocks"):
        msa.validate_config(cfg, topk_blocks=8, indexer_kv_dtype="bf16")


def test_jit_lock_rebinds_aliases_and_is_idempotent(tmp_path):
    calls = []

    def compile_variant(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    names = (
        "get_plan_fn",
        "get_prepare_metadata_fn",
        "get_sparse_topk_module",
        "get_reduction_module",
    )
    api = NS(get_fmha_variant=compile_variant, **dict.fromkeys(names, compile_variant))
    jit = NS(
        CACHE_BASE=tmp_path,
        _variant_manager=NS(get_variant=compile_variant),
        get_fmha_variant=compile_variant,
        **dict.fromkeys(names, compile_variant),
    )
    msa._install_jit_lock(api, jit)
    try:
        assert api.get_fmha_variant is jit._variant_manager.get_variant
        assert api.get_plan_fn is jit.get_plan_fn
        a = api.get_fmha_variant(a=1, b=2)
        assert api.get_fmha_variant(b=2, a=1) is a
        assert len(calls) == 1
        msa._install_jit_lock(api, jit)
        assert api.get_fmha_variant(a=1, b=2) is a
    finally:
        jit._sail_msa_jit_lock.file.close()


def test_jit_failure_propagates_without_deleting_cache(tmp_path):
    cache_file = tmp_path / "keep.so"
    cache_file.write_text("existing cache")
    lock = msa._JITLock(tmp_path / "lock")

    def fail():
        raise RuntimeError("compiler failure")

    try:
        with pytest.raises(RuntimeError, match="compiler failure"):
            lock.wrap(fail)()
        assert cache_file.read_text() == "existing cache"
        assert lock.depth == 0
    finally:
        lock.file.close()


def test_missing_library_error_preserves_cause(monkeypatch):
    msa.load_msa.cache_clear()

    def missing(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(msa.importlib, "import_module", missing)
    try:
        with pytest.raises(RuntimeError, match="PPU fmha_sm100") as caught:
            msa.load_msa()
        assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    finally:
        msa.load_msa.cache_clear()


@pytest.fixture
def selectors(monkeypatch):
    def module(name, **attrs):
        result = types.ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    def original_i(**kw):
        return "upstream-indexer", kw

    def original_a(**kw):
        return "upstream-attend", kw

    idx = module(
        "vllm.models.minimax_m3.common.indexer", select_indexer_impl_cls=original_i
    )
    attn = module(
        "vllm.models.minimax_m3.common.sparse_attention",
        select_main_backend_and_impl_cls=original_a,
    )
    model = module(
        "vllm.models.minimax_m3.nvidia.model",
        select_main_backend_and_impl_cls=original_a,
    )
    module("vllm.models.minimax_m3.common", indexer=idx, sparse_attention=attn)
    platform = NS(is_ppu=lambda: True)
    module("vllm.platforms", current_platform=platform)
    cfg = config()
    module("vllm.config", get_current_vllm_config=lambda: cfg)
    backends = module(
        "vllm_sail.attention.minimax_m3_msa",
        SAILIndexerImpl=object(),
        SAILSparseBackend=object(),
        SAILSparseImpl=object(),
    )
    probes = []
    monkeypatch.setattr(msa, "load_msa", lambda: probes.append(True))
    for name in (
        "VLLM_SAIL_MINIMAX_M3_MSA",
        "VLLM_PPU_MINIMAX_M3_MSA",
        "VLLM_SAIL_MINIMAX_M3_MSA_ATTEND",
        "VLLM_PPU_MINIMAX_M3_MSA_ATTEND",
    ):
        monkeypatch.delenv(name, raising=False)
    path = (
        Path(__file__).parents[2]
        / "vllm_sail/patch/enhancement/attention/minimax_m3_msa.py"
    )
    spec = importlib.util.spec_from_file_location("_msa_selector_test", path)
    mod = importlib.util.module_from_spec(spec)
    before = list(PATCH_REGISTRY)
    try:
        spec.loader.exec_module(mod)
        yield NS(
            module=mod,
            idx=idx,
            attn=attn,
            model=model,
            platform=platform,
            cfg=cfg,
            backends=backends,
            probes=probes,
            original_a=original_a,
        )
    finally:
        PATCH_REGISTRY[:] = before


def test_selectors_preserve_defaults_and_non_ppu(selectors, monkeypatch):
    s = selectors
    assert s.idx.select_indexer_impl_cls(topk_blocks=16)[0] == "upstream-indexer"
    monkeypatch.setenv("VLLM_SAIL_MINIMAX_M3_MSA", "1")
    s.platform.is_ppu = lambda: False
    assert s.idx.select_indexer_impl_cls(topk_blocks=16)[0] == "upstream-indexer"
    assert not s.probes


def test_selectors_route_both_stages_and_repair_model_alias(selectors, monkeypatch):
    s = selectors
    monkeypatch.setenv("VLLM_SAIL_MINIMAX_M3_MSA", "1")
    assert s.idx.select_indexer_impl_cls(topk_blocks=16) is s.backends.SAILIndexerImpl
    assert (
        s.model.select_main_backend_and_impl_cls
        is s.attn.select_main_backend_and_impl_cls
    )
    kwargs = dict(topk_blocks=16, kv_cache_dtype="auto", num_kv_heads=4)
    assert s.model.select_main_backend_and_impl_cls(**kwargs) == (
        s.backends.SAILSparseBackend,
        s.backends.SAILSparseImpl,
    )
    monkeypatch.setenv("VLLM_SAIL_MINIMAX_M3_MSA_ATTEND", "0")
    assert s.model.select_main_backend_and_impl_cls(**kwargs)[0] == "upstream-attend"
    marker = getattr(s.model.select_main_backend_and_impl_cls, PATCH_MARKER)
    target = "vllm.models.minimax_m3.nvidia.model.select_main_backend_and_impl_cls"
    assert marker[target] is s.original_a
    with pytest.raises(RuntimeError):
        patch(
            "vllm.models.minimax_m3.common.indexer",
            "select_indexer_impl_cls",
            reason="test",
            affected_versions="test",
            remove_when="test",
        )(s.idx.select_indexer_impl_cls)


def test_opt_in_fails_before_loading_library_for_unsupported_dtype(
    selectors, monkeypatch
):
    monkeypatch.setenv("VLLM_SAIL_MINIMAX_M3_MSA", "1")
    with pytest.raises(ValueError, match="indexer_kv_dtype"):
        selectors.idx.select_indexer_impl_cls(topk_blocks=16, indexer_kv_dtype="fp8")
    assert not selectors.probes


@pytest.mark.upstream_source
def test_pinned_selector_interfaces_and_shared_buffer(upstream_source_root):
    root = upstream_source_root / "vllm/models/minimax_m3"
    required = {
        "common/indexer.py": (
            "select_indexer_impl_cls",
            {"topk_blocks", "indexer_kv_dtype"},
        ),
        "common/sparse_attention.py": (
            "select_main_backend_and_impl_cls",
            {
                "topk_blocks",
                "kv_cache_dtype",
                "num_kv_heads",
                "emits_sparse_block_table",
            },
        ),
    }
    for name, (function, keywords) in required.items():
        tree = ast.parse((root / name).read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == function
        )
        assert {a.arg for a in node.args.kwonlyargs} == keywords
    model = ast.parse((root / "nvidia/model.py").read_text())
    imports = [
        n
        for n in model.body
        if isinstance(n, ast.ImportFrom)
        and n.module == "vllm.models.minimax_m3.common.sparse_attention"
    ]
    assert any(
        a.name == "select_main_backend_and_impl_cls" for n in imports for a in n.names
    )
    sparse_class = next(
        n
        for n in model.body
        if isinstance(n, ast.ClassDef) and n.name == "MiniMaxM3SparseAttention"
    )
    calls = [
        n
        for n in ast.walk(sparse_class)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "forward"
    ]
    assert any(any(k.arg == "query_fp8" for k in n.keywords) for n in calls)
