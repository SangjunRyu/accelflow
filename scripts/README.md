# Phase 0 scripts

Implemented:

- `check_env.sh` (read-only; run with Bash inside Ubuntu).
- `check_host_resources.ps1` (read-only Windows VHDX/RAM/volume inspection).
- `install_deps.sh` (Ubuntu 24.04 minimal baseline; defaults to `--plan`).
- `setup_ttsim.sh` (pinned official prebuilt Wormhole + standalone validation).

The baseline was installed on 2026-10-06 and logs were saved under `results/setup/`.
To reproduce after reviewing resources, run `sudo apt-get update`,
`bash scripts/install_deps.sh --plan`, then `sudo bash scripts/install_deps.sh --install`.
The script checks the C: backing-volume reserve; inspect apt's space estimate first.

Phase 0-D/0-E scripts are implemented and verified:

- `setup_tt_metal.sh`: external pinned clone and recursive submodules.
- `build_tt_metal.sh`: upstream build wrapper with a resource-conscious job count.
- `verify_ttsim.sh`: library/descriptor/build checks, official integer-add example,
  nonzero failure status, timestamped log under `results/setup/`.

The pinned TT-Metal example build and simulator smoke test passed on 2026-10-06.
`build_tt_metal.sh` uses upstream Unity builds, disables PCH, and defaults to two jobs
(`TT_BUILD_JOBS=1` is available). The command guard stops the build below 10 GiB C: free.
`verify_ttsim.sh` copies the official SOC descriptor and checks exit code 0 and result 21.
See [the actual build report](../docs/06-tt-metal-build-validation.md).

Phase 0-C is complete. Run `bash scripts/setup_ttsim.sh` in WSL to reproduce:
matching existing binaries are reused and mismatching files are not overwritten.
The default destination is `$HOME/dev/tenstorrent/sim`; no sudo is required.
See [installation and validation](../docs/04-ttsim-installation.md).

Phase 1: `bash scripts/run_vector_add.sh` builds only the two AccelFlow host executables
against the pinned existing TT-Metal libraries, then runs 9 uint32 and 12 BF16 cases.
Logs and correctness/structural JSON live under `results/vector_add/<UTC>/`.

Assessment extension: `run_vector_add.sh` now runs 10 uint32 cases and 14 BF16 cases
at CB depths 1 and 2, then produces design-assessment JSON/Markdown.

Phase 2: `bash scripts/run_matmul.sh` builds the AccelFlow MatMul host executable
against the existing pinned libraries and runs baseline/reuse_a at CB depths 1/2.
All 48 cases passed. Full logs, source hashes, numerical errors and structural
comparison reports are under `results/matmul/<UTC>/`.
See [MatMul implementation and assessment](../docs/10-matmul.md).

Blocked extension: the current `run_matmul.sh` runs baseline/reuse_a/blocked at
depths 1 and 2, with 14 fixtures each (84 passing cases). The latest failed suite
cannot fall back to an older successful assessment. See [11](../docs/11-blocked-matmul.md).

K-streaming extension: current `run_matmul.sh` runs 10 configurations / 188 cases.
Streamed chunk1/2 × depth1/2 and baseline/reuse_a include six extra K>128 fixtures;
old blocked retains its 14-case Kt<=4 coverage. All configurations use FP32 destination
accumulation with BF16 input/output. See [12](../docs/12-k-streamed-matmul.md).

`bash scripts/benchmark_matmul.sh [--samples 7]` measures host/simulator dispatch
wall time with 2 warmups and 5..25 samples per configuration. The default covers
36 configurations / 324 submissions and validates every repeated output.
Logs, raw samples, versions and comparison statistics live in `results/matmul_perf/<UTC>/`.
See [13](../docs/13-matmul-simulator-performance.md); these are not hardware NPU timings.

`bash scripts/inspect_compilation.sh [--variant streamed|blocked|baseline]` exports
host compile commands/LLVM IR and real cached SFPI ELF, symbols, sections and disassembly.
Default streamed inspection passed for five device ELF files. Existing cache/upstream
are read only; artifacts are saved under `results/compile_inspection/<UTC>/`.
See [14](../docs/14-compiler-inspection-and-tt-explorer.md). TT-Explorer is not installed.

`bash scripts/run_multicore_matmul.sh` builds the existing MatMul host target and
runs baseline depths1/2 plus independent multicore requests1/2/4 at depths1/2.
All 160 cases passed; original 188 cases also pass. Full logs, core tile assignments,
source hashes and coverage checks are under `results/matmul_multicore/<UTC>/`.
See [15](../docs/15-independent-multicore-matmul.md). No multicore hardware speedup is measured.

`bash scripts/benchmark_multicore_matmul.sh [--samples 7 --trials 3]` measures
independent multicore1/2/4 and baseline at depths1/2 on four fixtures. Defaults:
32 configurations, 96 process trials, 672 timed samples and 864 dispatches.
Correctness, identical fingerprints, core assignments, raw samples and source hashes
are retained under `results/matmul_multicore_perf/<UTC>/`. See [16](../docs/16-multicore-simulator-performance.md).

`bash scripts/run_mcast_matmul.sh` validates actual NoC A multicast against baseline
and independent multicore: 200 fixture cases plus 42 repeated-workload dispatches.
It checks sender/receiver handshakes, tail A consumption, physical receiver rectangles,
numerical equality and source-based DRAM/delivery models. Logs and hashes are under
`results/matmul_mcast/<UTC>/`. See [17](../docs/17-a-multicast-matmul.md).

`bash scripts/run_mcast2d_matmul.sh` validates row-A and column-B NoC multicast
against baseline, independent multicore and A-only: 240 common fixture cases plus
98 repeated/fallback dispatches. Covers 2×2 corner tails, separate semaphore pairs,
1×4/4×1 maps, CB wrap and K=512. Artifacts: `results/matmul_mcast2d/<UTC>/`.
See [18](../docs/18-2d-multicast-matmul.md).

`bash scripts/benchmark_multicast_matmul.sh [--samples 7 --trials 3]` compares
independent4/A-only4/2D4 plus an independent1 control at matching CB depths.
All primary comparisons use four active cores. Defaults: 32 configurations /
96 process trials / 672 timed samples / 864 submissions. Logs, raw timing,
source hashes and DRAM/receiver/control models: `results/matmul_multicast_perf/<UTC>/`.
See [19](../docs/19-multicast-simulator-performance.md). Different mappings/reader NoCs remain part of the implementation comparison.


## Windows CUDA MatMul

`run_cuda_matmul.ps1` exports validated WSL NPU inputs, builds sm_61 with the installed Windows
CUDA/MSVC tools, and runs naive/tiled correctness and timing trials on the GPU.
For a WSL UNC workspace under RemoteSigned, copy this runner to a local temporary file and pass
`-ProjectRoot` pointing to the workspace. The tested command is in
[21](../docs/21-cuda-matmul-comparison.md). `-ReuseFixturesFrom` keeps existing inputs and creates a new run.
Run `tools/report/summarize_cuda_matmul.py RESULT_DIRECTORY` inside WSL to generate the comparison.


## NPU row-wise sum

`bash scripts/run_reduction.sh` builds only the project host and validates the pinned simulator path.
Two pilot cases precede depth1/2 suites, each with three repeated submissions by default.
Use `--repeats 1..5` to change the repeat count. See [22](../docs/22-row-wise-reduction.md).


## cudaArray and legacy profiler

The CUDA runner accepts `-Variants naive,tiled,texture_tiled`; the default remains naive/tiled.
`tools/report/probe_cupti.py output.json` is a read-only Windows diagnostic for CUDA initialization,
CUPTI activity/callback subscription and legacy Event API support. It does not enable counter permissions
or change drivers. See [23](../docs/23-texture-matmul-and-profiler.md).

## CUDA row-wise reduction

Run `run_cuda_reduction.ps1 -ProjectRoot <WSL UNC project path>` from Windows PowerShell.
Copy the runner locally if RemoteSigned blocks UNC scripts; retain the project path argument.
The runner exports and validates 25 exact NPU fixtures, builds for sm_61, and compares
naive/tree/shuffle plus four larger CUDA-only fixtures. `-Trials` defaults to 3.
Results, raw samples, commands, and hashes go to `results/cuda_reduction/<UTC>/`.
See [24](../docs/24-cuda-reduction.md) for the actual results and timing limitations.

## RMSNorm

`bash scripts/run_rmsnorm.sh` validates the pinned virtual Wormhole implementation with
20 common BF16 fixtures, CB depth 1/2 and two repeated operator submissions per configuration.
The three device stages preserve intermediate data on device DRAM; host readback is validation only.

From Windows, `run_cuda_rmsnorm.ps1 -ProjectRoot <WSL UNC project path>` first validates NPU,
then builds and benchmarks CUDA unfused/fused variants and runs the Python FP64 cross-backend audit.
Copy the script locally if RemoteSigned blocks UNC execution; keep the ProjectRoot argument.
Use a new ResultsName for a fresh run. See [25](../docs/25-rmsnorm.md).

PyTorch CPU reference: install `requirements-pytorch-reference.txt` in the project `.venv`, then run
`.venv/bin/python tools/report/compare_rmsnorm_pytorch.py --results results/rmsnorm/<run>`.
It compares the 20 saved shared correctness fixtures with official FP32/FP64 RMSNorm,
checks four saved backends, and records `pytorch/comparison.json`. It does not benchmark PyTorch.
