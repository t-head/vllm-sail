# SPDX-License-Identifier: Apache-2.0
"""Bare-Python regression for a copied constructor using super()."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

PATCH_PATH = (
    Path(__file__).parents[2] / "vllm_sail/patch/enhancement/attention/mla_indexer.py"
)


@pytest.mark.parametrize("subclass", [False, True])
def test_rebased_indexer_init_calls_parent_once(
    monkeypatch, patch_utils_module, subclass
):
    """Execute the real copied body, globals rebasing and patch installation.

    Stop in the parent constructor: reproducing the missing class cell needs
    neither vLLM configuration nor device allocations later in the body.
    The subclass case also catches the recursive super(type(self), self) fix.
    """
    calls = []

    class ParentReached(Exception):
        pass

    class Parent:
        def __init__(self, *args, **kwargs):
            calls.append((self, args, kwargs))
            raise ParentReached

    class DeepseekV32IndexerMetadataBuilder(Parent):
        pass

    class DerivedBuilder(DeepseekV32IndexerMetadataBuilder):
        pass

    target = types.ModuleType("_test_mla_indexer_init_target")
    target.DeepseekV32IndexerMetadataBuilder = DeepseekV32IndexerMetadataBuilder
    monkeypatch.setitem(sys.modules, target.__name__, target)
    tree = ast.parse(PATCH_PATH.read_text())
    selected = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_with_target_globals", "_builder_init_body")
        ],
        type_ignores=[],
    )
    namespace = {"types": types, "_indexer_module": target}
    exec(compile(selected, str(PATCH_PATH), "exec"), namespace)
    replacement = namespace["_with_target_globals"](namespace["_builder_init_body"])
    patch_utils_module.patch(
        target.__name__,
        "DeepseekV32IndexerMetadataBuilder.__init__",
        reason="exercise the copied indexer constructor's parent dispatch",
        affected_versions=">=0.27.0,<0.28.0",
        remove_when="the indexer constructor no longer needs a copied body",
    )(replacement)

    builder_class = DerivedBuilder if subclass else DeepseekV32IndexerMetadataBuilder
    config = object()
    with pytest.raises(ParentReached):
        builder_class(config, block_table_width=29, device="test-device")
    assert len(calls) == 1
    assert type(calls[0][0]) is builder_class
    assert calls[0][1:] == ((config,), {"device": "test-device"})
    assert replacement.__globals__ is target.__dict__
