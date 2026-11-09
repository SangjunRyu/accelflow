#!/usr/bin/env bash
# Read-only inspection. Redirect stdout or pipe to tee to retain a report.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo 'Usage: bash scripts/check_env.sh (no arguments)' >&2
    exit 2
fi
if [[ $(uname -s) != Linux ]]; then
    echo 'ERROR: Run this script inside WSL Ubuntu or another Linux environment.' >&2
    exit 1
fi

section() { printf '\n## %s\n' "$1"; }
probe() {
    local label=$1
    shift
    printf '\n%s: ' "$label"
    if command -v "$1" >/dev/null 2>&1; then
        if "$@" 2>&1; then
            :
        else
            printf 'WARNING: probe failed for %s\n' "$label"
        fi
    else
        printf 'not found\n'
    fi
}

printf '# AccelFlow environment report\n'
printf 'Captured (Asia/Seoul): %s\n' "$(TZ=Asia/Seoul date --iso-8601=seconds)"
printf 'Captured (UTC): %s\n' "$(date -u --iso-8601=seconds)"
section 'OS and WSL'
if [[ -r /etc/os-release ]]; then
    cat /etc/os-release
else
    echo 'WARNING: /etc/os-release unavailable'
fi
printf 'Kernel: %s\n' "$(uname -r)"
if [[ ${WSL_DISTRO_NAME:-} != '' ]] || grep -qi microsoft /proc/sys/kernel/osrelease; then
    printf 'WSL detected: yes; distribution: %s\n' "${WSL_DISTRO_NAME:-unknown}"
else
    echo 'WSL detected: no'
fi
section 'CPU, memory, and storage'
printf 'Architecture: %s\n' "$(uname -m)"
probe 'Available CPU cores' nproc
probe 'CPU capabilities' lscpu
probe 'RAM and swap' free -h
probe 'Project filesystem' df -h .
probe 'Linux home filesystem' df -h "$HOME"
section 'Tool versions'
probe 'Python' python3 --version
probe 'GCC' gcc --version
probe 'G++' g++ --version
probe 'Clang' clang --version
probe 'Clang 20' clang-20 --version
probe 'CMake' cmake --version
probe 'Ninja' ninja --version
probe 'Git' git --version
probe 'glibc' getconf GNU_LIBC_VERSION
section 'Docker'
probe 'Docker client' docker --version
probe 'Docker server access (10 second timeout)' timeout 10s docker info --format '{{.ServerVersion}}'
section 'NVIDIA and CUDA (informational, unrelated to TTSim)'
if command -v nvidia-smi >/dev/null 2>&1; then
    probe 'NVIDIA GPU' timeout 10s nvidia-smi
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
    probe 'NVIDIA GPU via WSL' timeout 10s /usr/lib/wsl/lib/nvidia-smi
else
    echo 'nvidia-smi: not found'
fi
probe 'CUDA compiler' nvcc --version
section 'External checkouts (only explicitly configured paths)'
for variable in TT_METAL_HOME TT_METAL_SIMULATOR; do
    printf '%s: %s\n' "$variable" "${!variable:-not set}"
done
if [[ -n ${TT_METAL_HOME:-} && -d ${TT_METAL_HOME}/.git ]]; then
    probe 'TT-Metal commit' git -C "$TT_METAL_HOME" rev-parse HEAD
fi
section 'Interpretation'
echo 'Missing optional tools are observations, not script failures.'
echo 'No dependencies were installed and no system configuration was changed.'
echo 'This report does not prove TT-Metal/TTSim compatibility or kernel correctness.'
echo 'Simulator wall-clock time is not physical NPU latency.'
