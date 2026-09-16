# SPDX-License-Identifier: Apache-2.0
"""vLLM version compatibility for the PPU plugin.

**Policy (learned from vllm-ascend, which got this wrong):** version gates live
*only* in this module and in patch-module *selection*. Never call
:func:`vllm_version_is` from a hot path — vllm-ascend has 28 such call sites
inside its model runner and attention code and calls it out as a mistake.

The plugin targets a vLLM *range*, not a single tag. The PPU fork's upstream base
is the untagged commit ``4bdc8a788``, which sits after the ``v0.27.0`` release, so
pinning to an exact tag would be a lie. The authoritative pins live in
``.github/vllm-release-tag.commit`` and ``.github/vllm-main-verified.commit`` and
are what CI matrixes over.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

#: Inclusive lower bound / exclusive upper bound of the supported vLLM range.
MIN_VLLM_VERSION = "0.27.0"
MAX_VLLM_VERSION_EXCLUSIVE = "0.28.0"

_checked = False


def installed_vllm_version() -> str:
    """The installed vLLM version string.

    ``VLLM_VERSION`` overrides the detected value. This escape hatch exists
    because a vLLM installed from git reports a version like
    ``0.1.dev1+g4bdc8a788``, which no range check can interpret; developers set
    ``VLLM_VERSION`` to the release the checkout is based on. (Same mechanism as
    ``vllm_ascend.utils``.)
    """
    override = os.getenv("VLLM_VERSION")
    if override:
        return override.strip()

    try:
        from vllm.version import __version__

        return __version__
    except Exception:  # pragma: no cover - vllm always has a version in practice
        return "unknown"


def _version_key(version: str) -> tuple[int, ...] | None:
    """Parse the leading ``X.Y[.Z]`` of a version into a comparable tuple.

    Returns ``None`` when no ``X.Y`` prefix can be read, which callers treat as
    "cannot judge" rather than "incompatible".
    """
    head = version.strip().lstrip("v").split("+", 1)[0]
    parts: list[int] = []
    for chunk in head.split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if len(parts) >= 2 else None


def is_dev_build(version: str) -> bool:
    """Whether ``version`` looks like a VCS build rather than a release.

    This matters because setuptools-scm produces versions whose numeric prefix is
    *not* always a reliable ordering key. A shallow clone with no tags fetched
    reports something like ``0.1.dev1+g4bdc8a788``, whose ``(0, 1)`` prefix cannot
    prove which release the checkout is based on. Those ambiguous builds must use
    ``VLLM_VERSION``; a source build whose numeric prefix is inside the supported
    range remains usable without an override.
    """
    lowered = version.strip().lower()
    return ".dev" in lowered or "+g" in lowered or lowered.endswith(".dev")


def vllm_version_is(target: str) -> bool:
    """Whether the installed vLLM matches ``target`` exactly.

    Use for patch *selection* only, e.g. choosing between two variants of a
    patch across an upstream API change. Never in a hot path.
    """
    return installed_vllm_version().lstrip("v").split("+", 1)[0] == target.lstrip("v")


def vllm_version_at_least(target: str) -> bool:
    """Whether the installed vLLM is >= ``target``. ``False`` if unparseable."""
    current = _version_key(installed_vllm_version())
    wanted = _version_key(target)
    if current is None or wanted is None:
        return False
    return current >= wanted


def check_vllm_compatibility(*, force: bool = False) -> None:
    """Validate the installed vLLM against the supported range.

    Called once from :func:`vllm_sail.register_out_of_tree`, before any patch is
    applied, so that a version mismatch produces an actionable error instead of
    an obscure ``AttributeError`` from a patch whose target has moved.

    Raises:
        RuntimeError: The installed vLLM is outside the supported range, or its
            version cannot prove that it is inside the range.
    """
    global _checked
    if _checked and not force:
        return
    # A forced re-check replaces the cached result. Leave the cache invalid if
    # validation below fails and the caller catches the exception.
    _checked = False

    version = installed_vllm_version()
    key = _version_key(version)
    supported = f">={MIN_VLLM_VERSION},<{MAX_VLLM_VERSION_EXCLUSIVE}"
    explicit = os.getenv("VLLM_VERSION") is not None
    low = _version_key(MIN_VLLM_VERSION)
    high = _version_key(MAX_VLLM_VERSION_EXCLUSIVE)
    assert low is not None and high is not None

    if key is None or (is_dev_build(version) and not explicit and key < low):
        raise RuntimeError(
            f"vllm-sail {_plugin_version()} cannot verify compatibility against "
            f"vLLM version {version!r}. This plugin requires vLLM {supported}. "
            "If this is a source checkout based on a supported release, set "
            "VLLM_VERSION to that release. Refusing to apply patches to an "
            "unverifiable vLLM build."
        )

    if not (low <= key < high):
        raise RuntimeError(
            f"vllm-sail {_plugin_version()} requires vLLM {supported}, but vLLM "
            f"{version} is installed. Install a matching vLLM, or set "
            "VLLM_VERSION if you are running a git build of a supported "
            "release. Refusing to patch a vLLM this plugin has not been "
            "validated against."
        )

    if is_dev_build(version) and not explicit:
        logger.warning(
            "vllm-sail is accepting development build %r because its numeric "
            "version is inside the tested vLLM range %s. Set VLLM_VERSION to "
            "the release your checkout is based on to make this check "
            "authoritative.",
            version,
            supported,
        )

    _checked = True
    logger.debug("vllm-sail %s validated against vLLM %s", _plugin_version(), version)


def _plugin_version() -> str:
    from vllm_sail.version import __version__

    return __version__
