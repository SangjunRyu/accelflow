#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
checkout=${TT_METAL_HOME:-$HOME/dev/tenstorrent/tt-metal-v0.79.0}
revision=de546d3b146758714d900f11b218c8f9c805f410
mkdir -p "$root/results/setup" "$(dirname "$checkout")"
exec > >(tee "$root/results/setup/tt-metal-clone.log") 2>&1
if [[ ! -e "$checkout" ]]; then
    git clone --branch v0.79.0 --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/tenstorrent/tt-metal.git "$checkout"
fi
test "$(git -C "$checkout" rev-parse HEAD)" = "$revision"
git -C "$checkout" submodule update --init --recursive --depth 1
git -C "$checkout" submodule status --recursive
du -sh "$checkout"
echo "TT_METAL_HOME=$checkout"
