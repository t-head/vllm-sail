# SPDX-License-Identifier: Apache-2.0
"""Source checks must not silently skip a missing CI checkout."""

import pytest

from tests.support.source import source_root


def test_optional_source_checks_skip_without_a_checkout():
    with pytest.raises(pytest.skip.Exception, match="VLLM_SOURCE_ROOT"):
        source_root(None)


def test_required_source_checks_fail_without_a_checkout():
    with pytest.raises(pytest.UsageError, match="requires VLLM_SOURCE_ROOT"):
        source_root(None, required=True)


@pytest.mark.parametrize("required", [False, True])
def test_explicit_invalid_source_is_an_error(tmp_path, required):
    with pytest.raises(pytest.UsageError, match="must contain vllm/__init__.py"):
        source_root(str(tmp_path), required=required)


def test_source_validation_does_not_import_upstream(tmp_path):
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('must not import')\n")
    assert source_root(str(tmp_path), required=True) == tmp_path.resolve()
