# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 with the ``SupportsQuant`` interface.

The fork's *only* change to MiniMax-M3
(``vllm/models/minimax_m3/nvidia/model.py``, +3/-2) is adding
``SupportsQuant`` to the two public classes so the model works with
quantization configs such as the plugin's ``mixed_precision_w4``. It is not
handled by quantization utils: it is a model-class interface change.

Rather than patch the class bases, the plugin registers subclasses that add
the interface through the documented ``ModelRegistry.register_model`` API.
``SupportsQuant`` sits after the upstream classes in the MRO, so its
``__new__`` hook runs and the upstream behaviour is otherwise untouched.
"""

from __future__ import annotations

from vllm.model_executor.models.interfaces import SupportsQuant
from vllm.models.minimax_m3.nvidia.model import (
    MiniMaxM3SparseForCausalLM as _UpstreamMiniMaxM3SparseForCausalLM,
)
from vllm.models.minimax_m3.nvidia.model import (
    MiniMaxM3SparseForConditionalGeneration as _UpstreamMiniMaxM3SparseForConditionalGeneration,
)


class MiniMaxM3SparseForCausalLM(_UpstreamMiniMaxM3SparseForCausalLM, SupportsQuant):
    """Upstream MiniMax-M3 sparse backbone + the fork's SupportsQuant."""


class MiniMaxM3SparseForConditionalGeneration(
    _UpstreamMiniMaxM3SparseForConditionalGeneration, SupportsQuant
):
    """Upstream MiniMax-M3 VL entry point + the fork's SupportsQuant."""
