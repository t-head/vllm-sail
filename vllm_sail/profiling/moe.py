# SPDX-License-Identifier: Apache-2.0
"""Exception-safe MoE ranges around actual dispatch."""

from contextlib import contextmanager

from vllm_sail.profiling.nvtx import range_pop, range_push


@contextmanager
def moe_range(hidden_states, w1, topk_ids):
    import torch

    mode = "D" if torch.cuda.is_current_stream_capturing() else "P"
    range_push(
        f"{mode}_MoE,M_{hidden_states.shape[0]}_E_{w1.shape[0]}_H_{w1.shape[2]}_In_{w1.shape[1]}_topk_{topk_ids.shape[1]}"
    )
    try:
        yield
    finally:
        range_pop()
