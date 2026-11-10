# Single-core RMSNorm on virtual Wormhole

The operator is `y[h,w] = x[h,w] * rsqrt(sum(x[h,:]^2)/W + epsilon) * gamma[w]`.
Input and gamma are quantized BF16 values, widened exactly to FP32 for transport.
The output and intermediate storage are FP32. The default path is a three-stage correctness
baseline. The optional `fused` path retains intermediate values in L1 in one program.
Neither path measures real NPU performance.

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

## L1-retained fusion

`rmsnorm fixture-list.txt report.json depth repeats [unfused|fused] [cores]` keeps the old CLI valid;
omitting the final argument still selects unfused. `bash scripts/run_rmsnorm_fused.sh` builds
the host, validates both short pilots, runs the fused depth 1/2 suites, compares bitwise
outputs with the frozen 3-stage baseline, and tests the width resource guard.

Read `kernels/fused_compute.cpp` in square_row -> sum_row -> normalize_row order.
One 32-row group is processed at a time. The reader in `kernels/fused_reader.cpp` loads x once,
retains it in CB24, consumes squared values from CB25, feeds two precision-preserving scans,
and passes the CB26 prefix statistics to normalization. Gamma is streamed from DRAM.
Only x, gamma and y have device DRAM buffers. Intermediate host readback is disabled for fused;
the CPU FP64 reference, frozen bitwise outputs and independent PyTorch audit validate its result.

| CB | Producer | Consumer that pops it | Lifetime |
| --- | --- | --- | --- |
| 24 | reader | compute UNPACK | original row group retained through normalization |
| 25 | compute PACK | reader | full squared row group retained through both scans |
| 26 | compute PACK | reader | final prefix retained through normalization input preparation |
| 0,1,2,4,5 | reader | compute UNPACK | streamed scan/normalization operands |
| 16 | compute PACK | writer | final output tiles |

Each CB has one producer and one pointer-updating consumer. CB24/25 capacity equals the full
row-group width; reserving/popping that width returns the pointer to its base on every group.
No host synchronization separates arithmetic phases inside the fused program.

Let Wt=ceil(W/32), Ht=ceil(H/32), T=Ht*Wt. Fused CB payload is
`(2*Wt + 6*depth + 1)*4096` bytes. At W=4097 and depth 2 it is 1,110,016 bytes (1084 KiB),
versus the unfused 53,248-byte peak. Code, stacks, runtime allocations and full L1 residency
are not included in the payload model. Widths above 4097 are rejected before data-buffer allocation;
a separate conservative payload cap of 1200 KiB is also checked. This is not a full-width streaming
implementation and does not overlap two row groups.

Source-derived DRAM payload changes from `(7*T+2*Ht)*4096` to `3*T*4096` bytes; squared and
statistics DRAM capacities become zero. Local L1 copies, broadcasts, precision splitting and
two cumsum passes remain. This reduction does not establish actual chip speedup, bandwidth,
occupancy, or simulator timing gains. See [26](../../docs/26-npu-rmsnorm-fusion.md).

## Independent multicore row partitioning

The optional final `cores` argument accepts 1/2/4 for fused; default remains one.
The host splits contiguous 32-row groups over min(requested cores, group count) workers.
Empty workers receive no CB or kernel allocation. Reader receives a global row-group offset;
compute receives only its local group count; fused_writer writes a disjoint global tile interval.
The compute arithmetic is unchanged. No peer partial sums, multicast or intercore semaphore was added.
Input/gamma/output DRAM buffers are shared, with immutable inputs and disjoint output ranges.
All CBs and caches are private to each active core. Per-core payload remains unchanged while
aggregate CB payload grows with active cores. Total DRAM request payload remains 3*T*4096 bytes.

`bash scripts/run_rmsnorm_multicore.sh` validates balanced, uneven, and idle-worker cases.
Then run `.venv/bin/python tools/report/audit_rmsnorm_multicore.py --results <new-output>`.
See [27](../../docs/27-multicore-rmsnorm.md) for exact scope, assignments and simulator timing limits.

## 열 분할과 부분합 교환

variant `column`은 타일 열을 연속 구간으로 나누며 요청 코어는 1·2·4입니다.
모든 코어가 같은 32행 묶음을 처리하고, leader가 FP32 부분합을 더한 뒤 전체 합을 돌려줍니다.
dataflow kernel은 비트 배치와 NoC 이동을 맡고 덧셈·정규화는 compute에서 수행합니다.
CLI는 `rmsnorm list.txt report.json depth repeats [unfused|fused|column] [cores]`입니다.

```bash
bash scripts/run_rmsnorm_column.sh
.venv/bin/python tools/report/audit_rmsnorm_column.py --results <RESULTS>
```

분할·semaphore 수명·검증 결과·L1/통신 모델은 [28번](../../docs/28-column-sharded-rmsnorm.md)에 있습니다.
