# External dependencies

Keep TTSim binaries and TT-Metal source outside this repository, preferably under
`$HOME/dev/tenstorrent/` in the WSL Linux filesystem. Record release, full commit,
submodule revisions, download URL, and binary SHA-256 after installation.
Set `TT_METAL_HOME` to the external TT-Metal checkout. No upstream code is vendored.

Validated checkout: `$HOME/dev/tenstorrent/tt-metal-v0.79.0`, commit
`de546d3b146758714d900f11b218c8f9c805f410` with pinned submodules.
Validated simulator: `$HOME/dev/tenstorrent/sim/libttsim_wh.so`, TTSim v1.11.1.
The official integer-add smoke test passed; see [the build report](../docs/06-tt-metal-build-validation.md).
