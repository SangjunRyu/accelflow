# Single-core row-wise sum

Logical operation: y[h] = sum_w X[h,w]. Input BF16, FP32 destination and output.
The validated baseline transposes each 32x32 input tile and uses the pinned SFPU cumsum API.
Only the last prefix tile per 32-row group is packed. Host extraction selects its last row;
the device has already computed the sums. Output is not yet a compact H-element device buffer.

```bash
bash scripts/run_reduction.sh
```

See [22. Implementation and numerical investigation](../../docs/22-row-wise-reduction.md).
CB depth1/2 uses 6/12KiB, independent of W. A scaler tile is unnecessary in this scan path.
The initial native reduce_tile implementation failed a cancellation case under the pinned simulator;
its source snapshots and logs remain in results/reduction and are linked from the report.
This implementation covers row-sum only; CUDA, multicore collectives, mean/max and RMSNorm are future work.
