#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TT_METAL_HOME=${TT_METAL_HOME:-$HOME/dev/tenstorrent/tt-metal-v0.79.0}
export TT_METAL_SIMULATOR=${TT_METAL_SIMULATOR:-$HOME/dev/tenstorrent/sim/libttsim_wh.so}
example="$TT_METAL_HOME/build/programming_examples/metal_example_add_2_integers_in_riscv"
descriptor="$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml"
for required in "$TT_METAL_SIMULATOR" "$descriptor" "$example"; do
    [[ -f $required ]] || { echo "ERROR: Missing required file: $required" >&2; exit 1; }
done
[[ -x $example ]] || { echo "ERROR: Example is not executable: $example" >&2; exit 1; }
mkdir -p "$root/results/setup"
log="$root/results/setup/metal-smoke-$(date -u +%Y%m%dT%H%M%SZ).log"
cp "$descriptor" "$(dirname "$TT_METAL_SIMULATOR")/soc_descriptor.yaml"
cd "$TT_METAL_HOME"
{
    date -u --iso-8601=seconds
    git rev-parse HEAD
    git submodule status --recursive
    sha256sum "$TT_METAL_SIMULATOR" "$descriptor"
    printf 'TT_METAL_HOME=%s\nTT_METAL_SIMULATOR=%s\n' "$TT_METAL_HOME" "$TT_METAL_SIMULATOR"
    echo 'TT_METAL_SLOW_DISPATCH_MODE=1 timeout 300s ./build/programming_examples/metal_example_add_2_integers_in_riscv'
} | tee "$log"
set +e
TT_METAL_SLOW_DISPATCH_MODE=1 timeout 300s "$example" 2>&1 | tee -a "$log"
statuses=("${PIPESTATUS[@]}")
set -e
printf 'example_exit_code=%s\nlog_exit_code=%s\n' "${statuses[0]}" "${statuses[1]}" | tee -a "$log"
test "${statuses[0]}" -eq 0
test "${statuses[1]}" -eq 0
grep -F 'Success: Result is 21' "$log"
echo "Smoke test passed: $log"
echo 'TTSim validates functionality; simulator timing is not NPU hardware performance.'
