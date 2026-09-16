# SPDX-License-Identifier: Apache-2.0
"""Let block-quantized GEMM config lookup find the plugin's tuned configs.

``get_w8a8_block_fp8_configs`` and ``get_w8a8_block_int8_configs`` resolve their
``configs/`` directory relative to their own ``__file__`` and have **no**
user-folder hook (unlike ``fused_moe.get_moe_configs``, which honours
``VLLM_TUNED_CONFIG_FOLDER`` — see ``vllm_sail/registry/tuned_configs.py``). So
the config files this plugin ships are invisible to them.

Both patches delegate: try upstream first, and only if it found nothing, look in
the plugin's directory. That means

* a config bundled with vLLM still wins (upstream behaviour is unchanged for
  every non-PPU device), and
* nothing here has to reproduce upstream's filename-construction logic... except
  that it does, because upstream builds the filename *inside* the function and
  does not expose it. The format strings below are therefore duplicated from
  upstream and are the fragile part of this patch; the accompanying test pins
  them against real config files.

Note an upstream inconsistency worth knowing about: the fp8 variant builds
``block_shape=[{block_n},{block_k}]`` while the int8 variant builds
``block_shape=[{block_n}, {block_k}]`` — with a space. The formats below match
each function as it actually is, not as it ought to be.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import fp8_utils, int8_utils
from vllm.utils.platform_utils import get_device_name_as_file_name

from vllm_sail.patch.utils import patch
from vllm_sail.registry.tuned_configs import block_quant_config_path

logger = init_logger(__name__)

_AFFECTED = ">=0.27.0,<0.28.0"
_REMOVE_WHEN = (
    "upstream gives get_w8a8_block_{fp8,int8}_configs a user-config-folder hook "
    "like fused_moe.get_moe_configs already has (envs.VLLM_TUNED_CONFIG_FOLDER)."
)

_upstream_fp8 = fp8_utils.get_w8a8_block_fp8_configs
_upstream_int8 = int8_utils.get_w8a8_block_int8_configs


def _load_plugin_config(json_file_name: str, kind: str) -> dict[int, Any] | None:
    """Load a plugin-shipped block-quant config, or return None."""
    path = block_quant_config_path(json_file_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        logger.info(
            "Using vllm-sail bundled configuration from %s for W8A8 Block %s kernel.",
            path,
            kind,
        )
        return {int(key): val for key, val in json.load(f).items()}


@patch(
    "vllm.model_executor.layers.quantization.utils.fp8_utils",
    "get_w8a8_block_fp8_configs",
    reason=(
        "Upstream resolves its tuned-config directory relative to its own "
        "__file__ and offers no user-folder hook, so the FP8 block-quant configs "
        "this plugin ships for PPU-ZW810/ZW810E/ZW-M890P are never found and the "
        "kernel silently falls back to an untuned default."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def get_w8a8_block_fp8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None:
    config = _upstream_fp8(N, K, block_n, block_k)
    if config is not None:
        return config

    device_name = get_device_name_as_file_name()
    # Duplicated from upstream fp8_utils.get_w8a8_block_fp8_configs. Note: no
    # space after the comma in block_shape, unlike the int8 variant below.
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,"
        f"block_shape=[{block_n},{block_k}].json"
    )
    return _load_plugin_config(json_file_name, "FP8")


@patch(
    "vllm.model_executor.layers.quantization.utils.int8_utils",
    "get_w8a8_block_int8_configs",
    reason=(
        "Same directory-resolution problem as the FP8 variant above. Patched for "
        "symmetry and future INT8 block-quant tuning; the plugin currently ships "
        "no INT8 block-quant configs, so today this only ever delegates."
    ),
    affected_versions=_AFFECTED,
    remove_when=_REMOVE_WHEN,
)
def get_w8a8_block_int8_configs(
    N: int, K: int, block_n: int, block_k: int
) -> dict[int, Any] | None:
    config = _upstream_int8(N, K, block_n, block_k)
    if config is not None:
        return config

    device_name = get_device_name_as_file_name()
    # Duplicated from upstream int8_utils.get_w8a8_block_int8_configs, which -- in
    # contrast to the fp8 variant -- puts a space after the comma.
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype=int8_w8a8,"
        f"block_shape=[{block_n}, {block_k}].json"
    )
    return _load_plugin_config(json_file_name, "INT8")
