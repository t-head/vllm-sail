# SPDX-License-Identifier: Apache-2.0
"""FlashMLA patches cover upstream consumers that import operations by value."""

from __future__ import annotations

import ast

import pytest

from tests.support.source import ROOT


@pytest.mark.upstream_source
def test_flashmla_consumer_inventory_against_source_without_vllm_imports(
    upstream_source_root,
):
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
    source = upstream_source_root / "vllm"
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
