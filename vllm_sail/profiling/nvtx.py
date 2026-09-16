# SPDX-License-Identifier: Apache-2.0
"""Shared NVTX helpers for the opt-in profiling patches.

The in-tree PPU fork duplicated an identical import/no-op block in
``v1/worker/gpu_model_runner.py``, ``v1/engine/core.py`` and
``v1/core/sched/scheduler.py``. This module implements those helpers once,
with graceful degradation per optional dependency:

* ``nvtx`` (PyPI, ``vllm-sail[profiling]``) -- ``annotate`` and ``mark``
* ``model_prof`` (PPU SDK)                 -- the ``prof_iter`` hook
* ``torch.cuda.nvtx``                      -- ``range_push`` / ``range_pop``

Every helper is always importable and always callable; whatever is missing
becomes a no-op, so profiling can never crash the engine. The backends are
resolved once at import time and held in module-level slots, which tests can
swap to observe calls.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from vllm.logger import init_logger

logger = init_logger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])

try:
    from nvtx import annotate as _nvtx_annotate
    from nvtx import mark as _nvtx_mark
except ImportError:
    _nvtx_annotate = None
    _nvtx_mark = None

try:
    from model_prof import prof_iter as _model_prof_iter
except ImportError:
    _model_prof_iter = None

try:
    from torch.cuda.nvtx import range_pop as _torch_range_pop
    from torch.cuda.nvtx import range_push as _torch_range_push
except ImportError:
    _torch_range_pop = None
    _torch_range_push = None

#: True when the ``nvtx`` PyPI package (annotate/mark) is usable.
has_nvtx = _nvtx_mark is not None
#: True when ``model_prof`` from the PPU SDK provides ``prof_iter``.
has_model_prof = _model_prof_iter is not None
#: True when ``torch.cuda.nvtx`` imported; calls may still fail on a
#: CPU-only torch build, which :func:`range_push` swallows.
has_torch_nvtx = _torch_range_push is not None

_torch_nvtx_warned = False


def mark(message: str) -> None:
    """Emit an instantaneous NVTX mark, or no-op without the ``nvtx`` package."""
    if _nvtx_mark is not None:
        _nvtx_mark(message)


def annotate(name: str) -> Callable[[_F], _F]:
    """Decorator/context manager naming an NVTX range.

    Without the ``nvtx`` package this degrades to an identity decorator that
    preserves the wrapped callable's metadata, so ``@annotate`` sites stay
    unconditional.
    """
    if _nvtx_annotate is not None:
        return _nvtx_annotate(name)

    def decorator(func: _F) -> _F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def range_push(label: str) -> None:
    """Open a ``torch.cuda.nvtx`` range, or no-op when unavailable.

    Call-time errors (e.g. CPU-only torch raising "NVTX functions not
    installed") are swallowed with a one-time warning: an observability
    helper must never take down inference.
    """
    global _torch_nvtx_warned
    if _torch_range_push is None:
        return
    try:
        _torch_range_push(label)
    except Exception as exc:
        if not _torch_nvtx_warned:
            _torch_nvtx_warned = True
            logger.warning(
                "torch.cuda.nvtx is unusable in this build (%s); NVTX ranges "
                "are disabled for this process.",
                exc,
            )


def range_pop() -> None:
    """Close the range opened by the matching :func:`range_push`."""
    if _torch_range_pop is None:
        return
    try:
        _torch_range_pop()
    except Exception:
        # Already reported by range_push; a pop without a working push must
        # stay silent to keep the pair balanced.
        pass


def prof_iter(iteration: int) -> None:
    """Per-iteration hook for the PPU SDK profiler; no-op without model_prof."""
    if _model_prof_iter is not None:
        _model_prof_iter(iteration)


def sche_mark(sche_output: Any) -> None:
    """Emit NVTX marks describing one scheduler output.

    Mirrors the fork's ``sche_mark`` verbatim (labels included) so existing
    PPU profiling workflows read the same events.
    """
    if _nvtx_mark is None:
        return
    if len(sche_output.scheduled_new_reqs) > 0:
        reqs = [req.req_id for req in sche_output.scheduled_new_reqs]
        mark(f"new_reqs: {reqs}")
    mark(
        f"total_tokens={sche_output.total_num_scheduled_tokens},"
        f"req_id:num_tokens={sche_output.num_scheduled_tokens}"
    )
    if len(sche_output.finished_req_ids) > 0:
        mark(f"finish_req: {sche_output.finished_req_ids}")
