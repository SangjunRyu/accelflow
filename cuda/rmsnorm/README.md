# CUDA unfused and fused RMSNorm

Both variants calculate the same FP32 operator on identical BF16-quantized x and gamma:
`y = x * rsqrt(sum(x*x)/W + epsilon) * gamma`.

| Variant | Kernel launches per operator | Intermediate global buffers | Row input reuse |
| --- | --- | --- | --- |
| unfused | square, inverse_rms, normalize (3) | squared matrix + per-row scale | input reread during normalize |
| fused | fused_rmsnorm (1) | none | entire logical row retained in shared memory |

Both sum-square paths assign a row to one 256-thread block, use strided coalesced reads,
warp shuffle and eight shared partial sums, and calculate one inverse RMS per row.
Both explicitly round each square to FP32 using __fmul_rn before accumulation.
The difference includes launches, intermediate memory traffic, input reuse and resource usage;
it does not isolate the effect of a single optimization or shared-memory latency.

The fused block uses dynamic shared memory for W input values and eight partial sums,
plus one static shared scale: 4*(W+9) bytes. W is limited to 8193 for the validated MX150 path.
The unfused inverse_rms block uses 32 bytes; its other kernels use no shared memory.
No bank-conflict or measured occupancy counters were collected. Larger shared allocations can
affect residency, but its performance contribution was not separately measured.

Host validation uses an FP64 reference, atol=1e-4 and rtol=1e-4, checks all padded output lanes,
and requires deterministic repeated output. Valid rows are reset with a NaN sentinel outside
each measured operator batch. Shapes not divisible by 32/256 and positive epsilon near zero
are covered. Inputs with NaN/Inf, overflow or arbitrary FP32 dynamic range are outside scope.

Timing: 20 warmups, 15 performance samples, 20 operator repetitions per CUDA Event sample,
three fresh processes per variant and performance shape. The event interval for unfused covers
all three kernels. Scheduling and host-submission gaps may be included; this is not isolated
instruction latency. Pipeline timing separately covers pinned input+gamma H2D, one operator,
output D2H and stream completion. Allocation, parsing, sentinel reset and validation are excluded.

From Windows PowerShell, run `scripts/run_cuda_rmsnorm.ps1 -ProjectRoot <WSL UNC project path>`.
Copy the runner locally if RemoteSigned blocks a UNC script; do not change execution policy.
By default it first runs NPU validation on the common fixtures. Use a new result directory for
new experiments. Reports, hashes, source snapshots and cross-backend audit are retained.
See [25](../../docs/25-rmsnorm.md) for commands and measured results.
