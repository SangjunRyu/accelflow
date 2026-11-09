#!/usr/bin/env bash
# Official prebuilt single-chip Wormhole, pinned for reproducibility.
set -euo pipefail
if [[ $# -ne 0 ]]; then
    echo 'Usage: bash scripts/setup_ttsim.sh (configuration via environment)' >&2
    exit 2
fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
    echo 'ERROR: This prebuilt asset requires Linux/x86_64.' >&2; exit 1;
}
chip=${TTSIM_CHIP:-wormhole}
[[ $chip == wormhole ]] || { echo 'ERROR: Only Wormhole is validated by this script.' >&2; exit 1; }
tag=${TTSIM_TAG:-v1.11.1}
if [[ $tag == v1.11.1 ]]; then
    expected_sha=${TTSIM_SHA256:-196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a}
else
    expected_sha=${TTSIM_SHA256:?Specify the official release SHA-256 for a different tag}
fi
[[ $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ && $expected_sha =~ ^[a-f0-9]{64}$ ]] || {
    echo 'ERROR: Invalid release tag or SHA-256.' >&2; exit 1;
}
for tool in curl sha256sum file readelf nm ldd python3 timeout; do
    command -v "$tool" >/dev/null || { echo "ERROR: Missing tool: $tool" >&2; exit 1; }
done
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
sim_dir=${TTSIM_DIR:-$HOME/dev/tenstorrent/sim}
mkdir -p -- "$sim_dir" "$project_root/results/setup"
sim_dir=$(cd -- "$sim_dir" && pwd)
library="$sim_dir/libttsim_wh.so"
url="https://github.com/tenstorrent/ttsim/releases/download/$tag/libttsim_wh.so"
log_dir=$(mktemp -d "$project_root/results/setup/ttsim-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
exec > >(tee "$log_dir/validation.log") 2>&1
printf 'Release: %s\nURL: %s\nInstall directory: %s\n' "$tag" "$url" "$sim_dir"
if [[ -e $library ]]; then
    actual_sha=$(sha256sum "$library" | cut -d ' ' -f1)
    [[ $actual_sha == "$expected_sha" ]] || {
        echo 'ERROR: Existing library checksum differs. Refusing to overwrite it.' >&2; exit 1;
    }
    echo 'Existing library checksum matches; reusing without downloading.'
else
    available=$(df -B1 --output=avail "$sim_dir" | tail -n 1 | tr -d ' ')
    (( available >= 1024 * 1024 * 1024 )) || { echo 'ERROR: Insufficient filesystem headroom.' >&2; exit 1; }
    if [[ -d /mnt/c ]]; then
        available=$(df -B1 --output=avail /mnt/c | tail -n 1 | tr -d ' ')
        (( available >= 1024 * 1024 * 1024 )) || { echo 'ERROR: Insufficient C: backing-volume headroom.' >&2; exit 1; }
    fi
    temporary=$(mktemp "$sim_dir/.libttsim-download-XXXXXX")
    trap 'if [[ -n ${temporary:-} ]]; then rm -f -- "$temporary"; fi' EXIT
    curl --fail --location --retry 3 --connect-timeout 15 --max-time 120 \
        --proto '=https' --proto-redir '=https' --max-filesize 2097152 --output "$temporary" "$url"
    printf '%s  %s\n' "$expected_sha" "$temporary" | sha256sum --check --status
    chmod 0644 "$temporary"
    mv -n -- "$temporary" "$library"
fi
# Validate even a concurrently installed/reused file before executing loader checks.
printf '%s  %s\n' "$expected_sha" "$library" | sha256sum --check
file "$library"
readelf -h "$library"
readelf -d "$library"
readelf -l "$library"
readelf --version-info "$library"
nm -D --defined-only "$library"
ldd "$library"
timeout 30s python3 "$project_root/tools/report/validate_ttsim.py" \
    "$library" "$tag" "$url" "$expected_sha" "$log_dir/metadata.json"
cp -- "$log_dir/metadata.json" "$sim_dir/accelflow-install.json"
printf 'Standalone validation passed. Evidence: %s\n' "$log_dir"
echo 'TT-Metal execution and numerical kernel correctness remain unvalidated.'
echo 'Simulator runtime is not real NPU latency.'
