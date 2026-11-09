#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkout=${TT_METAL_HOME:-$HOME/dev/tenstorrent/tt-metal-v0.79.0}
test "$(git -C "$checkout" rev-parse HEAD)" = de546d3b146758714d900f11b218c8f9c805f410
mkdir -p "$root/results/setup"
cd "$checkout"
jobs=${TT_BUILD_JOBS:-2}
[[ $jobs == 1 || $jobs == 2 ]] || { echo 'TT_BUILD_JOBS must be 1 or 2.' >&2; exit 1; }
export CMAKE_BUILD_PARALLEL_LEVEL=$jobs
python3 "$root/tools/report/guarded_command.py" "$root/results/setup/tt-metal-configure.log" \
    bash ./build_metal.sh --configure-only --build-programming-examples \
    --without-distributed --without-python-bindings --disable-profiler --disable-pch
python3 "$root/tools/report/guarded_command.py" "$root/results/setup/tt-metal-build.log" \
    cmake --build build --target metal_example_add_2_integers_in_riscv --parallel "$jobs"
