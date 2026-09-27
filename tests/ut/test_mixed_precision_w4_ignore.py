# SPDX-License-Identifier: Apache-2.0
"""Exercise mixed-W4 routing without importing torch or device libraries."""

from types import SimpleNamespace

import pytest

from tests.support.source import function


class LinearBase:
    pass


class RoutedExperts:
    moe_config = object()


def route(ignored, channelwise, matcher):
    prefix = "model.mtp.layers.0.mlp.experts"
    config = SimpleNamespace(
        ignored_layers=ignored,
        int8_channelwise_layers=channelwise,
        packed_modules_mapping={},
        get_int8_channelwise_quant_method=lambda *args: "int8",
    )
    method = function(
        "vllm_sail/model_executor/layers/quantization/mixed_precision_w4.py",
        "MixedPrecisionW4Config.get_quant_method",
        {
            "LinearBase": LinearBase,
            "RoutedExperts": RoutedExperts,
            "is_layer_skipped": matcher,
            "should_ignore_layer": lambda prefix, ignore, fused_mapping: prefix
            in ignore,
            "UnquantizedFusedMoEMethod": lambda moe: ("unquantized", moe),
            "W4AInt8MoEMethod": lambda config, moe: ("int4", moe),
            "logger": SimpleNamespace(info_once=lambda *args: None),
        },
    )
    return method(config, RoutedExperts(), prefix)


@pytest.mark.parametrize("channelwise", [False, True])
def test_ignored_moe_takes_priority(channelwise):
    prefix = "model.mtp.layers.0.mlp.experts"
    result = route(
        [prefix],
        [prefix] if channelwise else [],
        lambda prefix, ignored_layers, fused_mapping: prefix in ignored_layers,
    )
    assert result == ("unquantized", RoutedExperts.moe_config)


@pytest.mark.parametrize("channelwise", [False, True])
def test_nonignored_moe_preserves_quantization(channelwise):
    prefix = "model.mtp.layers.0.mlp.experts"
    result = route(
        ["model.layers.0.mlp.experts"],
        [prefix] if channelwise else [],
        lambda prefix, ignored_layers, fused_mapping: prefix in ignored_layers,
    )
    assert result == ("int8" if channelwise else ("int4", RoutedExperts.moe_config))


@pytest.mark.upstream_source
def test_expert_checkpoint_path_with_upstream_matcher(upstream_source_root):
    matcher = function(
        upstream_source_root
        / "vllm/model_executor/layers/quantization/utils/quant_utils.py",
        "is_layer_skipped",
        {"MappingProxyType": dict},
    )
    result = route(["model.mtp.layers.0.mlp.experts.0.gate_proj"], [], matcher)
    assert result == ("unquantized", RoutedExperts.moe_config)
