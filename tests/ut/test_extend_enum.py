# SPDX-License-Identifier: Apache-2.0
"""Canary tests for the deliberate CPython Enum-internals extension."""

from __future__ import annotations

from enum import Enum

from vllm_sail.registry.moe_backends._extend import extend_enum


def test_extend_enum_updates_every_lookup_and_iteration_path() -> None:
    class Backend(Enum):
        CUDA = "cuda"

    member = extend_enum(Backend, "PPU", "ppu")

    assert member.name == "PPU"
    assert member.value == "ppu"
    assert Backend("ppu") is member
    assert Backend["PPU"] is member
    assert Backend.PPU is member
    assert isinstance(member, Backend)
    assert member in list(Backend)


def test_extend_enum_is_idempotent_on_running_interpreter() -> None:
    class Backend(Enum):
        CUDA = "cuda"

    first = extend_enum(Backend, "PPU", "ppu")
    second = extend_enum(Backend, "PPU", "ppu")

    assert second is first
    assert list(Backend) == [Backend.CUDA, first]
