#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Offline PPU E2E runner (kernels numeric + models offline precision).
# Executed inside the PPU pod dispatched by t-head/ppu-scheduler-action.
#
# NO server is started; pytest drives vLLM's offline LLM API directly.
# Results (junit XML, golden JSON, ppu-smi evidence) land in RESULTS_DIR
# (typically a NAS path shared with the outer runner for artifact upload).
#
# Environment contract (injected via the action's extra_env):
#   SUITE              "kernels" or "models"
#   TEST_PATH          pytest target path (e.g. tests/e2e/kernels)
#   PYTEST_MARKER      pytest -m expression (empty = no marker filter)
#   WHEELS_DIR         directory holding the downloaded *.whl artifacts
#   RESULTS_DIR        absolute output dir for xml/logs/golden/ppu-smi
#   MODEL_PATH         (models suite) checkpoint path → VLLM_SAIL_MODEL_ROOT
#   UPDATE_GOLDEN      "1" to pass --update-golden (default: "0")
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

# Prevent image defaults from masking the installed version or changing frontend.
unset VLLM_VERSION
export VLLM_USE_RUST_FRONTEND=0 VLLM_USE_RUST_BENCH=0

SUITE="${SUITE:?SUITE is required (kernels|models)}"
TEST_PATH="${TEST_PATH:?TEST_PATH is required}"
PYTEST_MARKER="${PYTEST_MARKER:-}"
WHEELS_DIR="${WHEELS_DIR:?WHEELS_DIR is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
MODEL_PATH="${MODEL_PATH:-}"
UPDATE_GOLDEN="${UPDATE_GOLDEN:-0}"

mkdir -p "${RESULTS_DIR}"
exec > >(tee "${RESULTS_DIR}/${SUITE}.log") 2>&1

# The internal HTTP(S) proxies do not route to NAS or loopback.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY || true

log() { printf '\n=== %s ===\n' "$1"; }

# ------------------------------------------------------------------
log "environment"
echo "SUITE=${SUITE} TEST_PATH=${TEST_PATH} PYTEST_MARKER=${PYTEST_MARKER:-<none>}"
echo "MODEL_PATH=${MODEL_PATH:-<none>} UPDATE_GOLDEN=${UPDATE_GOLDEN}"
echo "RESULTS_DIR=${RESULTS_DIR}"
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
echo "vllm wheel: ${vllm_wheels[0]}"
echo "sail wheel: ${sail_wheels[0]}"
# --no-deps keeps the SAIL-built torch/native ABI in the image untouched;
# --force-reinstall guarantees the freshly built wheels win over any preinstall.
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
assert torch.cuda.is_available(), "offline E2E pod requires a PPU device"
print("Device:", current_platform.get_device_name(0))
print("Capability:", current_platform.get_device_capability())
print("vLLM:", vllm.__version__)
print("vLLM SAIL:", vllm_sail.__version__)
PY

# ------------------------------------------------------------------
# For the models suite, bridge MODEL_PATH → VLLM_SAIL_MODEL_ROOT so the
# offline runner's resolve_checkpoint() picks up the CI-provided path.
if [[ "${SUITE}" == "models" && -n "${MODEL_PATH}" ]]; then
    export VLLM_SAIL_MODEL_ROOT="${MODEL_PATH}"
    test -d "${MODEL_PATH}" || { echo "checkpoint dir missing: ${MODEL_PATH}"; exit 1; }
    test -f "${MODEL_PATH}/config.json" || { echo "config.json missing in ${MODEL_PATH}"; exit 1; }
    log "model checkpoint verified: ${MODEL_PATH}"
fi

# ------------------------------------------------------------------
log "pytest ${SUITE}"
cd "${ROOT_DIR}"
git config --global --add safe.directory "${ROOT_DIR}" 2>/dev/null || true

# Build pytest args conditionally.
pytest_args=("${TEST_PATH}" -q -rs --junitxml="${RESULTS_DIR}/${SUITE}.xml")
if [[ -n "${PYTEST_MARKER}" ]]; then
    pytest_args+=(-m "${PYTEST_MARKER}")
fi
if [[ "${UPDATE_GOLDEN}" == "1" ]]; then
    pytest_args+=(--update-golden)
fi

set +e
python -m pytest "${pytest_args[@]}"
rc=$?
set -e

# ------------------------------------------------------------------
log "collect evidence"

# Golden JSON (models suite only): copy to RESULTS_DIR for artifact upload.
if [[ "${SUITE}" == "models" ]]; then
    mkdir -p "${RESULTS_DIR}/golden"
    golden_src="${ROOT_DIR}/tests/e2e/models/_golden"
    if [[ -d "${golden_src}" ]]; then
        find "${golden_src}" -maxdepth 1 -type f -name '*.json' \
            -exec cp -v {} "${RESULTS_DIR}/golden/" \; 2>/dev/null || true
    fi
fi

# ppu-smi occupancy evidence (best-effort; the tool may not exist in all images).
ppu-smi > "${RESULTS_DIR}/ppu-smi.log" 2>&1 || true

# ------------------------------------------------------------------
if (( rc != 0 )); then
    log "OFFLINE ${SUITE} FAILED (pytest exit ${rc})"
    exit "${rc}"
fi
log "OFFLINE ${SUITE} PASS"
