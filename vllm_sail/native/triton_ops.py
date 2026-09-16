# SPDX-License-Identifier: Apache-2.0
"""Triton implementations of upstream ops whose native TUs stay excluded.

Register at the torch.ops boundary so already-imported Python callers also use
the implementation. Schemas come from the generated corpus, just like stubs.
Importing this module requires neither torch, vLLM nor Triton.
"""

from __future__ import annotations

from vllm_sail.native import stubs


def merge_attn_states(
    output,
    output_lse,
    prefix_output,
    prefix_lse,
    suffix_output,
    suffix_lse,
    prefill_tokens_with_context,
    output_scale=None,
):
    from vllm_sail.attention.ops.merge_attn_states import merge_attn_states as merge

    return merge(
        output,
        output_lse,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        prefill_tokens_with_context,
        output_scale,
    )


IMPLEMENTATIONS = {("_C", "merge_attn_states"): merge_attn_states}
_libraries: list = []


def install() -> tuple[str, ...]:
    """Install before stubs, preserving kernels provided by a native vLLM."""
    import torch

    schemas = {(op.namespace, op.name): op for op in stubs.load_schemas()}
    installed = []
    for key, implementation in IMPLEMENTATIONS.items():
        namespace, name = key
        if stubs._op_exists(torch.ops, namespace, name):
            continue
        op = schemas.get(key)
        if op is None:
            raise RuntimeError(f"Missing generated schema for {namespace}::{name}")
        library = torch.library.Library(namespace, "FRAGMENT")
        try:
            library.define(op.schema)
            library.impl(name, implementation, "CUDA")
            library.impl(name, lambda *args, **kwargs: None, "Meta")
        except BaseException:
            # A schema without its implementation would make a retry skip this op.
            library._destroy()
            raise
        _libraries.append(library)
        installed.append(f"{namespace}::{name}")
    return tuple(installed)
