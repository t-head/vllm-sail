# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_sail.attention.flash_attn.flash_attn_interface import (
    __getattr__ as __getattr__,
)
from vllm_sail.attention.flash_attn.flash_attn_interface import (
    compile_flash_attn_varlen_func_from_specs,
    fa_version_unsupported_reason,
    flash_attn_varlen_func,
    get_scheduler_metadata,
    is_fa_version_supported,
)

__all__ = [
    "compile_flash_attn_varlen_func_from_specs",
    "fa_version_unsupported_reason",
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "is_fa_version_supported",
]
