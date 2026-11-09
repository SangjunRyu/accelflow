#!/usr/bin/env bash
# Minimal Ubuntu baseline, not the entire TT-Metal development environment.
set -euo pipefail

mode=${1:---plan}
if [[ $# -gt 1 || ( "$mode" != --plan && "$mode" != --install ) ]]; then
    echo 'Usage: bash scripts/install_deps.sh [--plan|--install]' >&2
    exit 2
fi
source /etc/os-release
if [[ ${ID:-} != ubuntu || ${VERSION_ID:-} != 24.04 || $(uname -m) != x86_64 ]]; then
    echo 'ERROR: This baseline targets Ubuntu 24.04 x86_64.' >&2
    exit 1
fi
packages=(build-essential cmake ninja-build git pkg-config ca-certificates curl wget
          python3-dev python3-venv libssl-dev libhwloc-dev libnuma-dev libtbb-dev
          libcapstone-dev)
printf 'Planned baseline packages:\n'
printf '  %s\n' "${packages[@]}"
echo 'No LLVM/GCC repositories, driver packages, or Python package installation.'
echo 'This baseline does not certify readiness to build a selected TT-Metal tag.'
export LC_ALL=C
if [[ "$mode" == --plan ]]; then
    apt-get --simulate --no-install-recommends install "${packages[@]}"
    exit 0
fi
if [[ $EUID -ne 0 ]]; then
    echo 'ERROR: Run --install as root or with sudo after reviewing --plan.' >&2
    exit 1
fi
if [[ ! -d /mnt/c ]]; then
    echo 'ERROR: Cannot check the previously identified C: backing volume.' >&2
    exit 1
fi
host_free=$(df -B1 --output=avail /mnt/c | tail -n 1 | tr -d ' ')
if (( host_free < 5 * 1024 * 1024 * 1024 )); then
    echo 'ERROR: C: has less than the 5 GiB minimal-install reserve.' >&2
    exit 1
fi
apt-get --no-install-recommends install -y "${packages[@]}"
echo 'Baseline installation completed; TT-Metal build was not started.'
