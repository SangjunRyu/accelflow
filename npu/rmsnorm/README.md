# Single-core RMSNorm on virtual Wormhole

The operator is `y[h,w] = x[h,w] * rsqrt(sum(x[h,:]^2)/W + epsilon) * gamma[w]`.
Input and gamma are quantized BF16 values, widened exactly to FP32 for transport.
The output and intermediate storage are FP32. This is a three-stage correctness baseline,
not a fused NPU implementation or a measurement of real NPU performance.

Read the code in this order:

1. `rmsnorm.cpp`: load common fixtures, pack tiles, allocate buffers and CBs, submit stages,
   validate square, row sum-square, normalized output, padding and repeat determinism.
2. `kernels/square.cpp`: square on the SFPU and write FP32 tiles to device DRAM.
3. `kernels/reader_sum.cpp`, `kernels/sum.cpp`: split each exact BF16 square into two
   TF32-representable pieces, run two transpose/cumsum scans, then combine final prefixes.
4. `kernels/reader_normalize.cpp`: read row statistics, split each FP32 statistic into three
   exactly transportable pieces, repeat them across columns and feed input/gamma CBs.
5. `kernels/normalize.cpp`: reconstruct statistics on the SFPU, divide by logical W,
   add epsilon, calculate reciprocal square root and multiply by x and gamma.

The RISC-V readers rearrange and split bit patterns; they do not calculate the sum or normalization.
No host readback is used to compute intermediate values. Host reads intermediates only after the
three stages finish, for validation. The padding divisor is logical W, not padded W.

Why split? A FP32 buffer and FP32 destination do not alone guarantee full-precision transport
through Wormhole SrcA/B. In this pinned simulator combination, direct FP32-to-Dest configuration
was rejected as undefined behavior. The final path uses the supported default input mode and
reconstructs values from representable pieces. The explicit transport property audit covers the
actual fixture corpus; this is not a general NaN/Inf/subnormal implementation.

For sum-square, two passes keep each final prefix in a separate DST slot. No other SFPU operation
is interleaved with an active cumsum carry. The two prefixes are added after both scans finish.
The third stage uses hi/middle/low pieces for arbitrary FP32 statistics in the validated range.

CB payload per stage: square/sum = depth*8192 bytes; normalize = (6*depth+1)*4096 bytes.
Peak is 28 KiB at depth 1 and 52 KiB at depth 2, excluding code, stacks and other L1 allocations.
Let T=ceil(H/32)*ceil(W/32), R=ceil(H/32). Intermediate DRAM capacity is (T+R)*4096 bytes.
Source-derived DRAM payload is (7*T+2*R)*4096 bytes, including two reads of the squared input,
gamma reads per input tile and one statistics read per row group. These are not hardware counters.

Run `bash scripts/run_rmsnorm.sh` in the WSL project. The runner checks the pinned TT-Metal
v0.79.0/TTSim v1.11.1, builds only this host executable, and runs a pilot plus depth 1/2 suites.
It does not clone/build TT-Metal or change drivers. See [25](../../docs/25-rmsnorm.md).
