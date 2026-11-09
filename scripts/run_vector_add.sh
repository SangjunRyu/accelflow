#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$root/results/vector_add"
set -o pipefail
python3 "$root/tools/report/build_vector_add.py" 2>&1 | tee "$root/results/vector_add/build.log"
python3 "$root/tools/report/run_vector_add.py"
python3 "$root/tools/report/assess_vector_add.py"
