# SPDX-License-Identifier: Apache-2.0
"""DeepEP prepare/finalize integration accepts upstream keyword arguments."""

from __future__ import annotations

import pytest

from tests.support.source import assert_accepts_upstream_keywords


@pytest.mark.upstream_source
@pytest.mark.parametrize(
    "local,local_name,upstream,upstream_name",
    [
        (
            "deepep",
            "_maybe_make_prepare_finalize_body",
            "model_executor/layers/fused_moe/all2all_utils",
            "maybe_make_prepare_finalize",
        )
    ],
)
def test_replacements_accept_upstream_keywords(
    upstream_source_root, local, local_name, upstream, upstream_name
):
    assert_accepts_upstream_keywords(
        local, local_name, upstream_source_root, upstream, upstream_name
    )
