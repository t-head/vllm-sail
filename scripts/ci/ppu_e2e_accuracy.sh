#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# PPU GSM8K accuracy E2E runner (server-client mode via evalscope).
# Executed inside the PPU pod dispatched by t-head/ppu-scheduler-action.
#
# Starts ``vllm serve`` + runs evalscope GSM8K accuracy evaluation.
# Results (junit XML, golden JSON, vllm serve logs) land in RESULTS_DIR.
#
# Environment contract (injected via the action's extra_env):
#   MODEL_KEY          model config key (e.g. deepseek_v4_int8)
#   WHEELS_DIR         directory holding the downloaded *.whl artifacts
#   RESULTS_DIR        absolute output dir for xml/logs/golden
#   UPDATE_GOLDEN      "1" to pass --update-golden (default: "0")
#   THRESHOLD          accuracy threshold (default: 0.9)
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Prevent image defaults from masking the installed version or changing frontend.
unset VLLM_VERSION 2>/dev/null || true
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

MODEL_KEY="${MODEL_KEY:?MODEL_KEY is required}"
WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
UPDATE_GOLDEN="${UPDATE_GOLDEN:-0}"
THRESHOLD="${THRESHOLD:-0.9}"

mkdir -p "${RESULTS_DIR}"
exec > >(tee "${RESULTS_DIR}/accuracy.log") 2>&1

# The internal HTTP(S) proxies do not route to NAS or loopback.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

log() { printf '\n=== %s ===\n' "$1"; }

# ------------------------------------------------------------------
log "environment"
echo "MODEL_KEY=${MODEL_KEY}"
echo "RESULTS_DIR=${RESULTS_DIR}"
echo "UPDATE_GOLDEN=${UPDATE_GOLDEN}"
echo "THRESHOLD=${THRESHOLD}"
python -c 'import sys; print("python", sys.version)'

# ------------------------------------------------------------------
log "install wheels"
WHEELS_DIR="$(cd -- "${WHEELS_DIR}" && pwd)"
python "${ROOT_DIR}/scripts/ci/ppu_wheel_manifest.py" verify "${WHEELS_DIR}"

mapfile -t vllm_wheels < <(
    find "${WHEELS_DIR}" -type f -name 'vllm-*.whl' -print | sort
)
mapfile -t sail_wheels < <(
    find "${WHEELS_DIR}" -type f -name 'vllm_sail-*.whl' -print | sort
)
((${#vllm_wheels[@]} == 1)) || { echo "expected one vLLM wheel under ${WHEELS_DIR}, got ${#vllm_wheels[@]}"; exit 1; }
((${#sail_wheels[@]} == 1)) || { echo "expected one vLLM SAIL wheel under ${WHEELS_DIR}, got ${#sail_wheels[@]}"; exit 1; }
python -m pip install --no-deps --force-reinstall "${vllm_wheels[0]}"
python -m pip install --no-deps --force-reinstall "${sail_wheels[0]}"

# Install test dependencies (pytest, etc.)
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
    python -m pip install -r "${ROOT_DIR}/requirements/dev.txt"

# ------------------------------------------------------------------
log "platform self-check"
cd /tmp
unset PYTHONPATH
python - <<'PY'
import torch, vllm, vllm_sail
from vllm.platforms import current_platform
from vllm_sail.compat import check_vllm_compatibility

check_vllm_compatibility(force=True)
assert current_platform.is_ppu(), f"expected PPU platform, got {current_platform}"
assert torch.cuda.is_available(), "accuracy E2E pod requires a PPU device"
print("Device:", current_platform.get_device_name(0))
print("Capability:", current_platform.get_device_capability())
print("vLLM:", vllm.__version__)
print("vLLM SAIL:", vllm_sail.__version__)
PY

# ------------------------------------------------------------------
log "materialise native extensions into the source tree"
WHEEL_PKG_DIR="$(cd /tmp && python -c 'import os, vllm_sail; print(os.path.dirname(vllm_sail.__file__))')"
echo "installed vllm_sail: ${WHEEL_PKG_DIR}"
mapfile -t wheel_so < <(
    find "${WHEEL_PKG_DIR}" -maxdepth 1 -type f -name '*.so' -print | sort
)
if ((${#wheel_so[@]} == 0)); then
    echo "FATAL: no *.so under ${WHEEL_PKG_DIR}; the wheel ships no native extensions"
    exit 78
fi
cp -fv "${wheel_so[@]}" "${ROOT_DIR}/vllm_sail/"

python -c 'import vllm_sail._C, vllm_sail._moe_C, vllm_sail._upstream_C, vllm_sail._upstream_moe_C' \
    || {
        echo "FATAL: source tree shadows the wheel - native extension import failed from ${ROOT_DIR}"
        exit 78
    }
echo "native extension import OK from ${ROOT_DIR}"

# ------------------------------------------------------------------
log "run accuracy eval: ${MODEL_KEY}"
cd "${ROOT_DIR}"
git config --global --add safe.directory "${ROOT_DIR}" 2>/dev/null || true

eval_args=(
    --model-key "${MODEL_KEY}"
    --threshold "${THRESHOLD}"
    --work-dir "${RESULTS_DIR}"
)
if [[ "${UPDATE_GOLDEN}" == "1" ]]; then
    eval_args+=(--update-golden)
fi

set +e
python scripts/eval/ppu_gsm8k_eval.py "${eval_args[@]}"
rc=$?
set -e

# ------------------------------------------------------------------
log "collect evidence"
mkdir -p "${RESULTS_DIR}/golden"
golden_src="${ROOT_DIR}/tests/e2e/models/_golden"
if [[ -d "${golden_src}" ]]; then
    find "${golden_src}" -maxdepth 1 -type f -name '*.json' \
        -exec cp -v {} "${RESULTS_DIR}/golden/" \; 2>/dev/null || true
fi

ppu-smi > "${RESULTS_DIR}/ppu-smi.log" 2>&1 || true

# ------------------------------------------------------------------
if (( rc != 0 )); then
    log "ACCURACY ${MODEL_KEY} FAILED (exit ${rc})"
    exit "${rc}"
fi
log "ACCURACY ${MODEL_KEY} PASS"
