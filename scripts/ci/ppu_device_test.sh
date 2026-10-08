#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 参数数组、逐文件执行和失败状态由同一 Python 解释器管理。
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "${ROOT_DIR}/scripts/ci/ppu_device_runner.py" run
