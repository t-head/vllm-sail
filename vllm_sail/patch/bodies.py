# SPDX-License-Identifier: Apache-2.0
"""Bind copied bodies to their upstream module without losing call metadata."""

from __future__ import annotations

import types


def bind_body(body: types.FunctionType, module: types.ModuleType) -> types.FunctionType:
    """Keep live provider lookups, defaults, annotations and closure cells.

    Copied class methods must spell ``super(TargetClass, self)`` explicitly:
    a module-level function has no compiler-created ``__class__`` cell.
    """
    bound = types.FunctionType(
        body.__code__,
        module.__dict__,
        body.__name__,
        body.__defaults__,
        body.__closure__,
    )
    bound.__kwdefaults__ = body.__kwdefaults__
    bound.__annotations__ = body.__annotations__
    bound.__dict__.update(body.__dict__)
    bound.__qualname__ = body.__qualname__
    bound.__doc__ = body.__doc__
    return bound
