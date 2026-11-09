# Nsight Systems CUDA MatMul captures

Open tiled.nsys-rep and texture_tiled.nsys-rep with Nsight Systems 2025.1.3 (File > Open).
Both use M=640, K=320, N=640 on the Windows NVIDIA GeForce MX150.
Each capture contains 16 GPU kernel executions and passed exact output validation.

Expand CUDA GPU kernel/memory lanes and CUDA API rows. Compare kernel launches,
host-to-device/array copies, device-to-host copies, and synchronization calls.
The texture path includes cudaMallocArray, cudaCreateTextureObject, and
cudaMemcpy2DToArrayAsync. The current workload uses a sequential stream.

These reports do not contain shared-memory bank-conflict counters, warp-stall
metrics, or measured occupancy. A longer kernel alone does not identify its cause.
No NVTX phase labels or CPU sampling were collected.

Capture overhead is present, including enabled CUDA Event completion tracing.
Use results/cuda_matmul/20261109T0916271819699Z for the earlier unprofiled statistical
comparison; do not replace those numbers with this short instrumented capture.

The *-stats_*.csv files summarize kernels, GPU memory operations, and CUDA API calls.
The SQLite exports allow further analysis without the GUI. Validation JSON,
capture logs, commands, tool version, and SHA-256 hashes are included.

Official documentation:
- https://docs.nvidia.com/nsight-systems/UserGuide/
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/
