# CUDA row-wise sum

`reduction.cu` implements three FP32 row-sum kernels over widened BF16 inputs:

| Variant | Work assignment | Aggregation |
| --- | --- | --- |
| naive | one thread per row | sequential register sum |
| tree | one 256-thread block per row | shared-memory halving tree; 1024 bytes |
| shuffle | one 256-thread block per row | warp shuffle plus eight shared partial sums; 32 bytes |

All variants emit compact row sums. NPU's current prefix-tile output has a different physical layout.
The comparison verifies the logical row sums, not equal transfer volume across architectures.
The tree and shuffle kernels have identical coalesced input assignment. Comparing naive to either
changes both row parallelism and input access pattern; it does not isolate shared-memory latency.
No bank-conflict hardware counters were collected.

Run `scripts/run_cuda_reduction.ps1` from Windows PowerShell, supplying `-ProjectRoot` as the WSL
UNC project directory. If RemoteSigned blocks a UNC script, copy that script to a local directory
and run the local copy with the same `-ProjectRoot`. Do not change execution policy.
`-LinuxProjectRoot` may be supplied explicitly; otherwise it is inferred from the UNC path.

The runner exports the NPU's actual 25 BF16 fixtures and validates their NPU outputs, builds
for Pascal sm_61 with CUDA 12.9/MSVC, and runs correctness and repeated timing comparisons.
Four additional CUDA-only integer inputs cover 128x320, 640x320, 320x4097, and 1024x8193.
Results contain raw samples, commands, validation reports, fixture hashes, and source snapshots.

Warmup: 20 launches. Correctness: 5 samples / batch 1. Performance: 15 samples / batch 20,
three fresh processes per variant. CUDA Event time is divided by the batch size; this includes
GPU scheduling and possible host-submission gaps, so it is not isolated instruction latency.
The separately timed pipeline includes pinned H2D, one kernel, D2H, and stream synchronization.
Allocation, fixture parsing, CPU reference, sentinel initialization, and validation are outside it.
Padding rows remain zero; valid output rows are initialized to NaN before each measured batch.
Finite BF16 test inputs use an FP64 CPU reference; integer cases require exact equality and
fractional cases use atol=1e-4 and rtol=1e-5. NaN/Inf semantics are outside this test scope.

See [24. CUDA Reduction](../../docs/24-cuda-reduction.md) for observed timings and limitations.
