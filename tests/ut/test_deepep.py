# SPDX-License-Identifier: Apache-2.0
"""DeepEP prepare/finalize integration accepts upstream keyword arguments."""

from __future__ import annotations

import ast

import pytest

from tests.support.source import ROOT, assert_accepts_upstream_keywords


@pytest.mark.upstream_source
def test_prepare_finalize_consumer_inventory_matches_upstream(upstream_source_root):
    tree = ast.parse((ROOT / "vllm_sail/patch/enhancement/deepep.py").read_text())
    inventory = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_CONSUMERS"
            for target in node.targets
        )
    )
    discovered = {name: set() for name in inventory}
    source = upstream_source_root / "vllm"
    for path in (source / "model_executor").rglob("*.py"):
        module = ".".join(("vllm", *path.relative_to(source).with_suffix("").parts))
        pending = list(ast.parse(path.read_text()).body)
        while pending:
            node = pending.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "vllm.model_executor.layers.fused_moe.all2all_utils"
            ):
                for alias in node.names:
                    if alias.name in discovered:
                        assert alias.asname is None, "renamed alias needs rebinding"
                        discovered[alias.name].add(module)
            pending.extend(ast.iter_child_nodes(node))
    assert discovered == {name: set(consumers) for name, consumers in inventory.items()}


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "deepep",
            "_maybe_make_prepare_finalize_body",
            "model_executor/layers/fused_moe/all2all_utils",
            "maybe_make_prepare_finalize",
        )
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
