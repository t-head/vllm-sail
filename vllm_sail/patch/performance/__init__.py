# SPDX-License-Identifier: Apache-2.0
"""PPU performance patches.

Optimisations that PPU needs on top of upstream's code paths. Unlike the
`enhancement` category, removing these does not break correctness — it costs
performance.

Currently contains only `fla` (flash-linear-attention): the fork's
cache_results autotune additions and the Qwen3-Next fused GDN decode path.
"""

from vllm_sail.patch.performance import fla  # noqa: F401
