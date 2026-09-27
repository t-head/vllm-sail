# SPDX-License-Identifier: Apache-2.0
"""MHC prenorm accumulator layout at the PPU GEMM boundary."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from tests.support.source import function


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
    fn = function(
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
