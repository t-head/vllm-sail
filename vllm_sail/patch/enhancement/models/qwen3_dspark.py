# SPDX-License-Identifier: Apache-2.0
"""PPU changes to ``vllm.model_executor.models.qwen3_dspark``.

Ports the fork's +4 hunk set: the DSpark Markov head must accept the model
quant config so ``markov_w2`` is built (and loaded) quantized like the rest
of the draft model. Two small body copies — the change is a new constructor
argument threaded through two adjacent constructors, which has no
delegation seam. Zero-argument ``super()`` in the upstream bodies becomes
explicit two-argument ``super`` because the replacements are module-level
functions installed on the class.
"""

from __future__ import annotations

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
    Qwen3DSparkModel,
)
from vllm.model_executor.models.utils import maybe_prefix

from vllm_sail.patch.utils import patch

_AFFECTED = ">=0.27.0,<0.28.0"


@patch(
    "vllm.model_executor.models.qwen3_dspark",
    "DSparkMarkovHead.__init__",
    reason=(
        "PPU quant schemes quantize the Markov head's markov_w2 projection; "
        "upstream builds it unquantized. Adds the quant_config argument and "
        "threads it into the ParallelLMHead. Small verbatim body copy; the "
        "change is a constructor argument with no delegation seam."
    ),
    affected_versions=_AFFECTED,
    remove_when="upstream DSparkMarkovHead takes a quant_config argument itself.",
)
def markov_head_init(
    self,
    vocab_size: int,
    draft_vocab_size: int,
    markov_rank: int,
    prefix: str,
    quant_config: QuantizationConfig | None = None,
) -> None:
    super(DSparkMarkovHead, self).__init__()
    self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
    self.markov_w2 = ParallelLMHead(
        draft_vocab_size,
        markov_rank,
        bias=False,
        # PPU MODIFICATION: begin
        quant_config=quant_config,
        # PPU MODIFICATION: end
        prefix=maybe_prefix(prefix, "markov_w2"),
        disable_tp=True,
    )


@patch(
    "vllm.model_executor.models.qwen3_dspark",
    "Qwen3DSparkModel.__init__",
    reason=(
        "Passes the model quant config into the DSpark Markov head so "
        "markov_w2 is quantized like the rest of the draft model. Small "
        "verbatim body copy; the change is one constructor argument."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "upstream Qwen3DSparkModel passes quant_config into DSparkMarkovHead "
        "itself."
    ),
)
def model_init(
    self,
    *,
    vllm_config: VllmConfig,
    start_layer_id: int = 0,
    prefix: str = "",
) -> None:
    super(Qwen3DSparkModel, self).__init__(
        vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
    )
    config = self.config
    draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
    self.markov_head = DSparkMarkovHead(
        config.vocab_size,
        draft_vocab_size,
        config.markov_rank,
        prefix=maybe_prefix(prefix, "markov_head"),
        # PPU MODIFICATION: begin
        quant_config=self.quant_config,
        # PPU MODIFICATION: end
    )
