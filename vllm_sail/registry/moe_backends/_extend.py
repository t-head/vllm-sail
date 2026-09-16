# SPDX-License-Identifier: Apache-2.0
"""Carefully extend an existing Enum without replacing its class object."""

from __future__ import annotations

from enum import Enum


def extend_enum(enum_cls: type[Enum], name: str, value: object) -> Enum:
    """Add a member to an existing Enum in place.

    Mutates the private lookup tables CPython's EnumMeta uses, so every
    existing reference to ``enum_cls`` sees the new member. Idempotent: returns
    the existing member if ``name`` is already present.
    """
    if name in enum_cls._member_map_:
        return enum_cls._member_map_[name]
    member = object.__new__(enum_cls)
    member._name_ = name
    member._value_ = value
    enum_cls._member_map_[name] = member
    enum_cls._value2member_map_[value] = member
    enum_cls._member_names_.append(name)
    # Plain setattr() raises AttributeError("Cannot reassign members.") because
    # EnumMeta.__setattr__ guards member assignment. Bypass that metaclass hook.
    type.__setattr__(enum_cls, name, member)
    return member
