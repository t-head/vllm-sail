#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts}"
VLLM_WHEEL_DIR="${ARTIFACT_DIR}/wheels/vllm"
SAIL_WHEEL_DIR="${ARTIFACT_DIR}/wheels/vllm-sail"
LOG_DIR="${ARTIFACT_DIR}/logs"
MANIFEST="${ARTIFACT_DIR}/build-manifest.txt"

SDK_URL="${SDK_URL:-https://art-pub.eng.t-head.cn/artifactory/apackage/daily/ppu.2v2_release/latest/PPU_SDK/ppu_1.0.0_Ubuntu2404_v13_release.run}"
TORCH_URL="${TORCH_URL:-https://art-pub.eng.t-head.cn/artifactory/apackage/daily/ppu.2v2_release/latest/PPU_SDK_PYPI/pytorch/ppu_sdk_hggcrt3-pytorch2.13.0-ubuntu2404-py312.tar.gz}"
VLLM_REPOSITORY="${VLLM_REPOSITORY:-https://github.com/vllm-project/vllm.git}"
VLLM_REF="${VLLM_REF:-releases/v0.27.1}"
PYTORCH_SAIL_ARCH="${PYTORCH_SAIL_ARCH:-ppu_15;ppu_10}"
# Building the native PPU kernels is memory-heavy (~2 GiB per compile job). The
# CPU runner exposes many cores but a smaller memory limit, so an unbounded
# MAX_JOBS makes the OOM killer terminate the pod (exit 137). Derive a safe
# default from the cgroup memory limit (falling back to MemTotal) and apply a
# conservative ceiling. Override explicitly with BUILD_JOBS.
default_jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')"
mem_bytes=0
if [[ -r /sys/fs/cgroup/memory.max ]]; then
    read -r mem_bytes </sys/fs/cgroup/memory.max || mem_bytes=0
    [[ "${mem_bytes}" == "max" ]] && mem_bytes=0
elif [[ -r /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    read -r mem_bytes </sys/fs/cgroup/memory/memory.limit_in_bytes || mem_bytes=0
fi
if [[ ! "${mem_bytes}" =~ ^[0-9]+$ ]] || ((mem_bytes <= 0 || mem_bytes > 1099511627776)); then
    mem_bytes="$(awk '/MemTotal/{print $2*1024}' /proc/meminfo 2>/dev/null || printf '0')"
fi
if [[ "${mem_bytes}" =~ ^[0-9]+$ ]] && ((mem_bytes > 0)); then
    mem_jobs=$((mem_bytes / (2 * 1024 * 1024 * 1024)))
    ((mem_jobs < 1)) && mem_jobs=1
    ((default_jobs > mem_jobs)) && default_jobs="${mem_jobs}"
fi
((default_jobs > 16)) && default_jobs=16
BUILD_JOBS="${BUILD_JOBS:-${default_jobs}}"

mkdir -p "${VLLM_WHEEL_DIR}" "${SAIL_WHEEL_DIR}" "${LOG_DIR}"
: >"${MANIFEST}"
exec > >(tee -a "${LOG_DIR}/build.log") 2>&1

WORK_DIR="$(mktemp -d)"
cleanup() {
    rm -rf "${WORK_DIR}"
}
on_error() {
    local status=$?
    trap - ERR
    printf 'status=failed\nexit_code=%s\n' "${status}" >>"${MANIFEST}"
    exit "${status}"
}
trap cleanup EXIT
trap on_error ERR

fail() {
    printf '%s\n' "$1" >&2
    return 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

download() {
    local url=$1
    local output=$2
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --retry-all-errors \
            --output "${output}" "${url}"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --output-document="${output}" "${url}"
    else
        fail "curl or wget is required"
    fi
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        fail "sha256sum or shasum is required"
    fi
}

url_basename() {
    local path=${1%%\?*}
    basename "${path}"
}

[[ "$(uname -s)" == "Linux" ]] || fail "native wheel builds require Linux"
for command_name in git python tar tee awk find ldconfig; do
    require_command "${command_name}"
done
PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "${PYTHON_VERSION}" == "3.12" ]] || \
    fail "the selected SAIL PyTorch archive requires Python 3.12"
[[ -w /usr/local ]] || fail "/usr/local must be writable inside the build container"

SDK_INSTALLER="${WORK_DIR}/$(url_basename "${SDK_URL}")"
TORCH_ARCHIVE="${WORK_DIR}/$(url_basename "${TORCH_URL}")"
printf 'status=running\n' >>"${MANIFEST}"
printf 'vllm_ref=%s\n' "${VLLM_REF}" >>"${MANIFEST}"
printf 'pytorch_sail_arch=%s\n' "${PYTORCH_SAIL_ARCH}" >>"${MANIFEST}"
printf 'sdk_file=%s\n' "$(basename "${SDK_INSTALLER}")" >>"${MANIFEST}"
printf 'torch_file=%s\n' "$(basename "${TORCH_ARCHIVE}")" >>"${MANIFEST}"

download "${SDK_URL}" "${SDK_INSTALLER}"
download "${TORCH_URL}" "${TORCH_ARCHIVE}"
printf 'sdk_sha256=%s\n' "$(sha256 "${SDK_INSTALLER}")" >>"${MANIFEST}"
printf 'torch_sha256=%s\n' "$(sha256 "${TORCH_ARCHIVE}")" >>"${MANIFEST}"

# job container 是临时环境；移走镜像内软件栈后安装本次指定版本。
for path in /usr/local/PPU_SDK /usr/local/cuda /usr/local/cuda-13.0; do
    if [[ -e "${path}" || -L "${path}" ]]; then
        mv "${path}" "${WORK_DIR}/$(basename "${path}").image"
    fi
done
sh "${SDK_INSTALLER}" --silent --prefix=/usr/local \
    2>&1 | tee "${LOG_DIR}/sdk-install.log"
ldconfig
# shellcheck disable=SC1091
source /usr/local/PPU_SDK/envsetup.sh
export PPU_SDK=/usr/local/PPU_SDK
export HGCC="${PPU_SDK}/bin/hgcc"
export PYTORCH_SAIL_ARCH
unset TORCH_CUDA_ARCH_LIST
unset VLLM_SAIL_HG_ARCH
unset VLLM_SAIL_SKIP_EXT
# envsetup.sh repoints pip at an internal mirror (art.eng.t-head.cn) that is
# unreachable from the CPU runner; force the public Aliyun PyPI mirror after the
# SDK env is sourced. Override with SAIL_PIP_INDEX_URL. All packages installed
# via pip are ordinary PyPI packages; PPU-specific components come from the SDK
# and torch archive (installed from local wheels), not from any pip index.
export PIP_INDEX_URL="${SAIL_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"

TORCH_DIR="${WORK_DIR}/torch"
mkdir -p "${TORCH_DIR}"
tar -xzf "${TORCH_ARCHIVE}" -C "${TORCH_DIR}"
mapfile -t torch_wheels < <(
    find "${TORCH_DIR}" -type f -path '*/package/*.whl' -print
)
((${#torch_wheels[@]} > 0)) || fail "no SAIL PyTorch wheel found in archive"
python -m pip install --force-reinstall --no-deps "${torch_wheels[@]}" \
    2>&1 | tee "${LOG_DIR}/torch-install.log"

command -v hgcc
hgcc --version 2>&1 | tee "${LOG_DIR}/hgcc-version.log"
if command -v nvcc >/dev/null 2>&1; then
    fail "nvcc must not be present in the CUDA-free build environment"
fi
python -c 'import torch; print(torch.__version__); print(torch.__file__)' \
    2>&1 | tee "${LOG_DIR}/torch-env.log"

VLLM_SRC="${WORK_DIR}/vllm"
git init "${VLLM_SRC}"
git -C "${VLLM_SRC}" remote add origin "${VLLM_REPOSITORY}"
git -C "${VLLM_SRC}" fetch --depth 1 origin "${VLLM_REF}"
git -C "${VLLM_SRC}" checkout --detach FETCH_HEAD
(
    cd "${VLLM_SRC}"
    python use_existing_torch.py
    python -m pip install -r requirements/build/cuda.txt
    VLLM_TARGET_DEVICE=empty MAX_JOBS="${BUILD_JOBS}" \
        python setup.py bdist_wheel --dist-dir "${VLLM_WHEEL_DIR}"
) 2>&1 | tee "${LOG_DIR}/vllm-build.log"

mapfile -t vllm_wheels < <(
    find "${VLLM_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print
)
((${#vllm_wheels[@]} == 1)) || \
    fail "expected exactly one vLLM wheel, found ${#vllm_wheels[@]}"
python -m pip install --force-reinstall --no-deps "${vllm_wheels[0]}"

python -m pip install \
    -r "${ROOT_DIR}/requirements/build.txt" \
    -r "${ROOT_DIR}/requirements/ppu.txt"
(
    cd "${ROOT_DIR}"
    python - <<'PY'
from vllm_sail.native.toolchain import decide, probe

decision = decide(probe())
print(decision.reason)
if not decision.build:
    raise SystemExit(1)
PY
) 2>&1 | tee "${LOG_DIR}/native-preflight.log"
(
    cd "${ROOT_DIR}"
    VERBOSE=1 MAX_JOBS="${BUILD_JOBS}" \
        python setup.py bdist_wheel --dist-dir "${SAIL_WHEEL_DIR}"
) 2>&1 | tee "${LOG_DIR}/vllm-sail-build.log"

mapfile -t sail_wheels < <(
    find "${SAIL_WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' -print
)
((${#sail_wheels[@]} == 1)) || \
    fail "expected exactly one vLLM SAIL wheel, found ${#sail_wheels[@]}"
[[ "$(basename "${sail_wheels[0]}")" != *-none-any.whl ]] || \
    fail "native build unexpectedly produced a pure-Python wheel"
[[ -s "${vllm_wheels[0]}" ]] || fail "vLLM wheel is empty"
[[ -s "${sail_wheels[0]}" ]] || fail "vLLM SAIL wheel is empty"

{
    printf 'status=succeeded\n'
    printf 'python=%s\n' "$(python --version 2>&1)"
    printf 'hgcc=%s\n' "$(hgcc --version 2>&1 | head -n 1)"
    printf 'vllm_commit=%s\n' "$(git -C "${VLLM_SRC}" rev-parse HEAD)"
    printf 'vllm_wheel=%s\n' "$(basename "${vllm_wheels[0]}")"
    printf 'vllm_sail_commit=%s\n' "$(git -C "${ROOT_DIR}" rev-parse HEAD)"
    printf 'vllm_sail_wheel=%s\n' "$(basename "${sail_wheels[0]}")"
} >>"${MANIFEST}"
