# 92. Measurement policy — 참고 문서

[진행 순서와 문서 안내](README.md)

CUDA measurements require actual GPU execution, CUDA events, separated warmup,
iteration count, shape, datatype, hardware/software versions, and summary statistics.
Report transfers and kernel-only latency separately.

TTSim is for correctness and execution-structure analysis. Record tile counts,
buffers, data movement, synchronization, layouts, core counts, and numerical errors.
Simulator wall-clock runtime may diagnose setup issues but is never NPU latency,
throughput, TOPS, or evidence of BOS Eagle-N performance.

Archive command, exit status, logs, versions, parameters, and reference-validation
results for every experiment. A successful build is not a successful device test.

Simulator execution-cost experiments may compare host `steady_clock` timing of
workload submission plus completion, with transfers/CPU reference outside the interval,
warmup and raw samples retained, and each output validated. Label them host/simulator
execution cost rather than NPU kernel latency, bandwidth, TOPS or hardware speedup.
Keep source-derived traffic/storage models separate from measured elapsed time.
See [13](13-matmul-simulator-performance.md).
