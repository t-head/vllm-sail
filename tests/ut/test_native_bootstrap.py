# SPDX-License-Identifier: Apache-2.0
"""Bootstrap contract for an empty-target vLLM installation."""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

from vllm_sail.native import bootstrap


@pytest.fixture(autouse=True)
def restore_upstream_extension_module():
    missing = object()
    original = sys.modules.get(bootstrap.UPSTREAM_EXTENSION, missing)
    yield
    if original is missing:
        sys.modules.pop(bootstrap.UPSTREAM_EXTENSION, None)
    else:
        sys.modules[bootstrap.UPSTREAM_EXTENSION] = original


def test_install_supplies_the_extension_module_missing_from_empty_vllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = types.ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", parent)
    monkeypatch.delitem(sys.modules, bootstrap.UPSTREAM_EXTENSION, raising=False)
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda name: None)

    assert bootstrap.install() == "shimmed"

    installed = sys.modules[bootstrap.UPSTREAM_EXTENSION]
    assert installed.__name__ == bootstrap.UPSTREAM_EXTENSION
    assert installed.__package__ == "vllm"
    assert installed.__spec__.name == bootstrap.UPSTREAM_EXTENSION
    assert installed.__vllm_sail_empty_shim__ is True
    assert parent._C_stable_libtorch is installed


def test_install_is_idempotent_and_never_replaces_a_loaded_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = types.ModuleType(bootstrap.UPSTREAM_EXTENSION)
    monkeypatch.setitem(sys.modules, bootstrap.UPSTREAM_EXTENSION, existing)

    assert bootstrap.install() == "loaded"
    assert sys.modules[bootstrap.UPSTREAM_EXTENSION] is existing


def test_install_never_masks_a_discoverable_real_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, bootstrap.UPSTREAM_EXTENSION, raising=False)
    spec = importlib.machinery.ModuleSpec(bootstrap.UPSTREAM_EXTENSION, loader=None)
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda name: spec)

    assert bootstrap.install() == "available"
    assert bootstrap.UPSTREAM_EXTENSION not in sys.modules
