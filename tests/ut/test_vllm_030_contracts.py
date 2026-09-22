# SPDX-License-Identifier: Apache-2.0
"""Upgrade contracts exercised without importing torch, vLLM or SAIL SDK."""

from __future__ import annotations

import ast
import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


def _definition(path: Path, name: str):
    node = ast.parse(path.read_text())
    for part in name.split("."):
        node = next(
            n
            for n in node.body
            if isinstance(n, ast.ClassDef | ast.FunctionDef) and n.name == part
        )
    return copy.deepcopy(node)


def _function(relative, name, namespace):
    node = _definition(ROOT / relative, name)
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    return namespace[node.name]


def _source_root():
    source = os.environ.get("VLLM_SOURCE_ROOT")
    if source is None:
        pytest.skip("VLLM_SOURCE_ROOT enables pinned upstream source contracts")
    return Path(source)


@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "attention/fa_utils",
            "get_flash_attn_version",
            "v1/attention/backends/fa_utils",
            "get_flash_attn_version",
        ),
        (
            "attention/fa_utils",
            "flash_attn_supports_kv_cache_dtype",
            "v1/attention/backends/fa_utils",
            "flash_attn_supports_kv_cache_dtype",
        ),
        (
            "attention/flash_attn",
            "supports_combination",
            "v1/attention/backends/flash_attn",
            "FlashAttentionBackend.supports_combination",
        ),
        (
            "attention/kda",
            "_init",
            "models/kimi_k3/nvidia/kda",
            "KimiK3DeltaAttention.__init__",
        ),
        (
            "attention/kda",
            "_flashkda_prefill",
            "models/kimi_k3/nvidia/kda",
            "_flashkda_prefill",
        ),
        (
            "attention/kda",
            "is_flashkda_supported",
            "models/kimi_k3/nvidia/kda",
            "is_flashkda_supported",
        ),
        (
            "attention/kda",
            "is_fused_kda_decode_supported",
            "models/kimi_k3/nvidia/kda",
            "is_fused_kda_decode_supported",
        ),
        (
            "attention/sparse_attn_indexer",
            "sparse_attn_indexer_init",
            "model_executor/layers/sparse_attn_indexer",
            "SparseAttnIndexer.__init__",
        ),
        (
            "attention/mla_indexer",
            "_builder_init_body",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder.__init__",
        ),
        (
            "attention/mla_indexer",
            "_split_indexer_prefill_chunks_body",
            "v1/attention/backends/mla/indexer",
            "DeepseekV32IndexerMetadataBuilder._split_indexer_prefill_chunks",
        ),
        (
            "deepep",
            "_maybe_make_prepare_finalize_body",
            "model_executor/layers/fused_moe/all2all_utils",
            "maybe_make_prepare_finalize",
        ),
        (
            "models/deepseek_v4_cache",
            "dequantize_and_gather_k_cache",
            "models/deepseek_v4/common/ops/cache_utils",
            "dequantize_and_gather_k_cache",
        ),
        (
            "mxfp4",
            "install._convert_weight",
            "model_executor/layers/fused_moe/oracle/mxfp4",
            "convert_weight_to_mxfp4_moe_kernel_format",
        ),
    ],
)
def test_replacements_accept_current_upstream_keywords(
    local, local_name, upstream, upstream_name
):
    replacement = _definition(
        ROOT / f"vllm_sail/patch/enhancement/{local}.py", local_name
    )
    original = _definition(_source_root() / f"vllm/{upstream}.py", upstream_name)
    # Compare call shape without importing annotations or executing decorators.
    accepted = {a.arg for a in replacement.args.args + replacement.args.kwonlyargs}
    offered = {a.arg for a in original.args.args + original.args.kwonlyargs}
    assert replacement.args.kwarg is not None or offered <= accepted
    if original.args.kwarg is not None:
        assert replacement.args.kwarg is not None
    if original.args.vararg is not None:
        assert replacement.args.vararg is not None


def test_fa4_probe_does_not_load_sdk(monkeypatch):
    from vllm_sail.attention.flash_attn import flash_attn_interface as interface

    def forbidden():
        pytest.fail("a capability probe for FA4 must not import PPU binaries")

    monkeypatch.setattr(interface, "_kernels", forbidden)
    assert interface.is_fa_version_supported(4) is False
    assert "FA4" in interface.fa_version_unsupported_reason(4)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("mxfp4", [False, True])
def test_ppu_experts_reuse_base_activation_config(batched, mxfp4):
    """Custom quantizers must use the same resolved values as activation()."""
    basename = "batched_deep_gemm_moe" if batched else "deep_gemm_moe"
    name = "PPUBatchedDeepGemmExperts" if batched else "PPUDeepGemmExperts"
    if mxfp4:
        name += "MXFP4"
    path = ROOT / f"vllm_sail/model_executor/layers/fused_moe/experts/{basename}.py"
    init = _definition(path, f"{name}.__init__")
    resolved = SimpleNamespace(clamp_limit=6.5, alpha=1.8, beta=0.25)

    class Base:
        def __init__(self, **kwargs):
            self.activation_config = resolved

    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[init],
        decorator_list=[],
    )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    namespace = {"Base": Base}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    moe = SimpleNamespace(activation_situ_beta=None, activation_situ_linear_beta=None)
    quant = SimpleNamespace(
        block_shape=None, gemm1_clamp_limit=None, gemm1_alpha=None, gemm1_beta=None
    )
    kwargs = dict(moe_config=moe, quant_config=quant)
    if batched:
        kwargs.update(max_num_tokens=32, num_dispatchers=2)
    obj = namespace[name](**kwargs)
    assert (obj.gemm1_clamp_limit, obj.gemm1_alpha, obj.gemm1_beta) == (6.5, 1.8, 0.25)


def test_upstream_activation_config_precedence():
    """Execute upstream's resolver to pin quant > model > neutral precedence."""
    path = _source_root() / "vllm/model_executor/layers/fused_moe/activation.py"
    cls = _definition(path, "ApplyMoEActivationConfig")
    module = ast.Module(body=[cls], type_ignores=[])
    namespace = {"dataclass": dataclass}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    config = namespace["ApplyMoEActivationConfig"]
    moe = SimpleNamespace(
        swiglu_limit=7.0,
        swiglu_alpha=1.7,
        swiglu_beta=0.3,
        activation_situ_beta=1.2,
        activation_situ_linear_beta=-0.8,
    )
    quant = SimpleNamespace(gemm1_clamp_limit=None, gemm1_alpha=None, gemm1_beta=None)
    model = config.from_configs(moe, quant)
    assert (model.clamp_limit, model.alpha, model.beta) == (7.0, 1.7, 0.3)
    quant.gemm1_clamp_limit, quant.gemm1_alpha, quant.gemm1_beta = 4.0, 0.0, 0.0
    override = config.from_configs(moe, quant)
    assert (override.clamp_limit, override.alpha, override.beta) == (4.0, 0.0, 0.0)
    assert (override.activation_situ_beta, override.activation_situ_linear_beta) == (
        1.2,
        -0.8,
    )
    moe.swiglu_limit = moe.swiglu_alpha = moe.swiglu_beta = None
    quant.gemm1_clamp_limit = quant.gemm1_alpha = quant.gemm1_beta = None
    neutral = config.from_configs(moe, quant)
    assert (neutral.clamp_limit, neutral.alpha, neutral.beta) == (None, 1.0, 0.0)


def test_pla_prefill_preserves_caller_buffers(monkeypatch):
    calls = []
    pla = ModuleType("pla.prefill.flashkdapro")
    pla.flashkda_fwd = lambda **kw: calls.append(kw)
    monkeypatch.setitem(sys.modules, pla.__name__, pla)
    monkeypatch.setenv("VLLM_SAIL_USE_PLA", "1")

    class Tensor:
        shape = (1, 5, 12, 128)

        def view(self, *shape):
            return self

    namespace = {
        "current_platform": SimpleNamespace(is_ppu=lambda: True),
        "logger": SimpleNamespace(info_once=lambda *args: None),
    }
    fn = _function(
        "vllm_sail/patch/enhancement/attention/kda.py", "_flashkda_prefill", namespace
    )
    tensors = [Tensor() for _ in range(13)]
    q, k, v, g, beta, a, bias, initial, seq, out, final, workspace, checkpoint = tensors
    result = fn(q, k, v, g, beta, a, bias, -5.0, initial, seq, out, final, workspace)
    assert result == (out, final)
    assert calls[0]["out"] is out and calls[0]["final_state"] is final
    with pytest.raises(NotImplementedError, match="checkpoint"):
        fn(
            q,
            k,
            v,
            g,
            beta,
            a,
            bias,
            -5.0,
            initial,
            seq,
            out,
            final,
            workspace,
            checkpoint_state=checkpoint,
        )
    assert len(calls) == 1


def test_mhc_prenorm_preserves_ppu_accumulator_contract(monkeypatch):
    calls = []
    gemm = ModuleType("vllm_sail.utils.deep_gemm")
    gemm.is_deep_gemm_supported = lambda: True
    gemm.tf32_hc_prenorm_gemm = lambda *args: calls.append(args)
    monkeypatch.setitem(sys.modules, gemm.__name__, gemm)
    allocations = []

    def zeros(*shape, **kwargs):
        value = SimpleNamespace(shape=shape, kwargs=kwargs)
        allocations.append(value)
        return value

    namespace = {
        "current_platform": SimpleNamespace(is_ppu=lambda: True),
        "torch": SimpleNamespace(zeros=zeros, float32="fp32"),
    }
    fn = _function(
        "vllm_sail/patch/enhancement/attention/mhc_tilelang.py",
        "_hc_prenorm_gemm_outputs",
        namespace,
    )
    x, w = (
        SimpleNamespace(shape=(7, 4096), device="ppu"),
        SimpleNamespace(shape=(24, 4096)),
    )
    assert fn(x, w, hidden_size=1024, hc_mult=4) == tuple(allocations)
    assert [t.shape for t in allocations] == [(1, 7, 24), (1, 7)]
    assert calls[0][-1] == 1


@pytest.mark.parametrize(
    "ppu,sm80,backend,expected",
    [
        (True, False, "auto", "mxfp4"),
        (True, False, "ppu_deep_gemm", "mxfp4"),
        (True, True, "auto", None),
        (True, False, "marlin", None),
        (True, False, "ppu_deep_gemm_w4a16", None),
        (False, False, "auto", "upstream"),
    ],
)
def test_deepseek_v4_mxfp4_selector_preserves_activation_mode(
    monkeypatch, ppu, sm80, backend, expected
):
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        is_ppu=lambda: ppu,
        is_device_capability=lambda cap: sm80 and cap == (8, 0),
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms)
    namespace = {
        "oracle": SimpleNamespace(
            kMxfp4Dynamic="mxfp4",
            select_mxfp4_moe_backend=lambda config, activation_key: activation_key,
        ),
        "_upstream_select_deepseek_v4": lambda config: "upstream",
    }
    fn = _function(
        "vllm_sail/registry/moe_backends/mxfp4.py",
        "select_deepseek_v4_mxfp4_moe_backend",
        namespace,
    )
    assert fn(SimpleNamespace(moe_backend=backend)) == expected


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_ppu_kda_cannot_select_nvidia_kernel_when_pla_disabled(monkeypatch, mode):
    sdk = ModuleType("vllm_sail.attention.pla_kda")
    sdk.get_pla_kda_kernel = lambda mode: None
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    namespace = {
        "current_platform": SimpleNamespace(
            is_ppu=lambda: True,
            is_cuda=lambda: True,
            get_device_capability=lambda: SimpleNamespace(major=10),
            is_device_capability=lambda cap: True,
            is_device_capability_family=lambda cap: True,
        ),
        "torch": SimpleNamespace(bfloat16="bf16", float32="fp32"),
        "is_conv_state_dim_first": lambda: False,
    }
    fn = _function(
        "vllm_sail/patch/enhancement/attention/kda.py",
        "is_flashkda_supported"
        if mode == "prefill"
        else "is_fused_kda_decode_supported",
        namespace,
    )
    args = (
        (128, "bf16", "fp32", -5.0)
        if mode == "prefill"
        else (12, 128, 4, 0, "bf16", "bf16", "fp32")
    )
    assert fn(*args) is False


def test_flashmla_consumer_inventory_against_source_without_vllm_imports():
    """Keep preloaded consumer coverage checkable on the dependency-free runner."""
    path = ROOT / "vllm_sail/patch/enhancement/attention/flashmla_ops.py"
    assignment = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_FLASHMLA_ALIAS_CONSUMERS"
    )
    expected = {
        name: set(consumers)
        for name, consumers in ast.literal_eval(assignment.value).items()
    }
    discovered = {name: set() for name in expected}
    source = _source_root() / "vllm"
    for path in source.rglob("*.py"):
        consumer = ".".join(("vllm", *path.relative_to(source).with_suffix("").parts))
        for node in ast.parse(path.read_text()).body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "vllm.v1.attention.ops.flashmla"
            ):
                for alias in node.names:
                    if alias.name in discovered:
                        discovered[alias.name].add(consumer)
    assert discovered == expected
