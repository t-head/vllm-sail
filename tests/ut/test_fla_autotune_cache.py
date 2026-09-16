# SPDX-License-Identifier: Apache-2.0
"""FLA decorator-chain regression tests without Triton, torch or vLLM."""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest


class Autotuner:
    def __init__(self, fn, configs, key, cache_results=False):
        self.fn = fn
        self.configs = configs
        self.keys = key
        self.cache_results = cache_results

    def run(self, **kwargs):
        return self.fn(**kwargs)


class Heuristics:
    def __init__(self, fn, values):
        self.fn = fn
        self.values = values

    def run(self, **kwargs):
        for name, value in self.values.items():
            kwargs[name] = value(kwargs)
        return self.fn.run(**kwargs)


@pytest.fixture
def cache_patch(monkeypatch, patch_utils_module):
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = logging.getLogger
    importing = types.ModuleType("vllm.triton_utils.importing")
    importing.HAS_TRITON = False
    triton_utils = types.ModuleType("vllm.triton_utils")

    def autotune(configs, key, cache_results=False):
        return lambda fn: Autotuner(fn, configs, key, cache_results)

    triton_utils.triton = types.SimpleNamespace(autotune=autotune)
    for name, module in {
        "vllm.logger": logger,
        "vllm.triton_utils": triton_utils,
        "vllm.triton_utils.importing": importing,
        "vllm_sail.patch.utils": patch_utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).parents[2] / "vllm_sail/patch/performance/fla/autotune_cache.py"
    )
    spec = importlib.util.spec_from_file_location("_test_fla_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HAS_TRITON = True
    target = types.ModuleType("_test_fla_ops")
    monkeypatch.setitem(sys.modules, target.__name__, target)
    module._TARGETS = {target.__name__: ("kernel",)}
    return module, target, triton_utils.triton


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_install_preserves_heuristics_and_enables_inner_cache(
    cache_patch, depth, caplog
):
    module, target, _ = cache_patch
    tuner = Autotuner(lambda **kw: kw, [object()], ["N"])
    original = tuner
    for _ in range(depth):
        original = Heuristics(original, {"EVEN": lambda args: args["N"] % 2 == 0})
    target.kernel = original

    module._install()

    rebuilt = target.kernel
    for _ in range(depth):
        assert isinstance(rebuilt, Heuristics)
        rebuilt = rebuilt.fn
    assert rebuilt.cache_results is True
    assert rebuilt.configs is tuner.configs
    assert rebuilt.keys is tuner.keys
    assert target.kernel.run(N=4) == ({"N": 4, "EVEN": True} if depth else {"N": 4})
    assert "skipping the cache_results patch" not in caplog.text
    assert target.kernel.__vllm_sail_patch__["_test_fla_ops.kernel"] is original
    # Captured upstream references and patch-drift metadata retain the old chain.
    assert tuner.cache_results is False


def test_rebuild_preserves_pruning_benchmark_and_user_hooks(cache_patch):
    module, target, triton = cache_patch
    tuner = Autotuner(lambda **kw: kw, [object()], ["N"])
    tuner.perf_model = object()
    tuner.configs_top_k = 0.5
    tuner.early_config_prune = object()
    tuner._do_bench = object()
    tuner.pre_hook = object()
    tuner.post_hook = object()
    tuner.user_defined_pre_hook = True
    tuner.user_defined_post_hook = True
    tuner.reset_to_zero = ["out"]
    tuner.restore_value = ["input"]
    tuner.use_cuda_graph = False
    captured = {}

    # Some Triton builds make configs optional, so it must be forwarded even
    # when it is not a required positional parameter in the live signature.
    def autotune(configs=None, key=None, **kwargs):
        captured.update(kwargs)
        return lambda fn: Autotuner(fn, configs, key, kwargs["cache_results"])

    triton.autotune = autotune
    target.kernel = Heuristics(tuner, {})
    module._install()
    assert target.kernel.fn.configs is tuner.configs
    assert captured["prune_configs_by"] == {
        "perf_model": tuner.perf_model,
        "top_k": 0.5,
        "early_config_prune": tuner.early_config_prune,
    }
    assert captured["do_bench"] is tuner._do_bench
    assert captured["pre_hook"] is tuner.pre_hook
    assert captured["post_hook"] is tuner.post_hook
    assert captured["reset_to_zero"] == ["out"]
    assert captured["restore_value"] == ["input"]
    assert captured["use_cuda_graph"] is False


def test_generated_hooks_are_recreated_without_initializing_driver(cache_patch):
    module, target, triton = cache_patch

    class LazyDriverAutotuner(Autotuner):
        @property
        def do_bench(self):
            raise AssertionError("driver initialized during patch installation")

    tuner = LazyDriverAutotuner(lambda **kw: kw, [object()], ["N"])
    tuner._do_bench = None
    tuner.pre_hook = object()
    tuner.post_hook = object()
    tuner.user_defined_pre_hook = False
    tuner.user_defined_post_hook = False
    tuner.reset_to_zero = ["out"]
    captured = {}

    def autotune(configs, key, **kwargs):
        captured.update(kwargs)
        return lambda fn: Autotuner(fn, configs, key, kwargs["cache_results"])

    triton.autotune = autotune
    target.kernel = tuner
    module._install()
    assert "do_bench" not in captured
    assert "pre_hook" not in captured
    assert "post_hook" not in captured
    assert captured["reset_to_zero"] == ["out"]


def test_ppu_positional_only_signature(cache_patch):
    module, target, triton = cache_patch

    def autotune(configs, key, /, cache_results=False):
        return lambda fn: Autotuner(fn, configs, key, cache_results)

    triton.autotune = autotune
    target.kernel = Autotuner(lambda **kw: kw, [object()], ["N"])
    module._install()
    assert target.kernel.cache_results is True


def test_triton_without_disk_cache_keeps_original_kernel(cache_patch, caplog):
    module, target, triton = cache_patch

    def autotune(configs, key):
        raise AssertionError("a rebuild without cache support has no effect")

    triton.autotune = autotune
    target.kernel = original = Heuristics(
        Autotuner(lambda **kw: kw, [object()], ["N"]), {}
    )
    module._install()
    assert target.kernel is original
    assert "no cache_results parameter" in caplog.text


@pytest.mark.parametrize("cycle", [False, True])
def test_missing_autotuner_is_diagnosed_without_replacing_kernel(
    cache_patch, caplog, cycle
):
    module, target, _ = cache_patch
    original = Heuristics(lambda: None, {})
    if cycle:
        original.fn = original
    target.kernel = original
    module._install()
    assert target.kernel is original
    assert "no triton Autotuner" in caplog.text


def test_no_triton_installs_nothing(cache_patch):
    module, target, _ = cache_patch
    module.HAS_TRITON = False
    # No target kernel exists. Looking it up would fail.
    module._install()
    assert not hasattr(target, "kernel")


def test_reapplication_is_rejected_by_patch_framework(cache_patch):
    module, target, _ = cache_patch
    target.kernel = Heuristics(Autotuner(lambda **kw: kw, [object()], ["N"]), {})
    module._install()
    with pytest.raises(RuntimeError, match="already patched"):
        module._install()
