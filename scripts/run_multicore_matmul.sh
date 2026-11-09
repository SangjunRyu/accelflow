#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/results/matmul_multicore"
python3 "$root/tools/report/build_matmul.py" 2>&1 | tee "$root/results/matmul_multicore/build.log"
python3 "$root/tools/report/run_multicore_matmul.py"
