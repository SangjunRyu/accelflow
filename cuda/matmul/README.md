# CUDA MatMul

`matmul.cu` contains three FP32 variants for the MX150 (sm_61): naive, 16x16 shared-memory tiled, and cudaArray point-texture tiled.
Both read the exact BF16-valued input exported by the TT-Metal host, widened to FP32.
CUDA storage/output remains FP32; a host BF16-rounded fingerprint supports numerical comparison.

See [21. CUDA implementation, measurements and limitations](../../docs/21-cuda-matmul-comparison.md)
for the tested Windows runner command, event and host timing scopes, raw results, and input format.
Build dependencies: Windows CUDA 12.9, MSVC x64 Build Tools, Python; the fixture producer uses
the existing WSL TT-Metal/TTSim installation. No Python package dependency is required.

The runner builds in LOCALAPPDATA/AccelFlow/cuda-matmul and archives reports in results/cuda_matmul.
Use -ReuseFixturesFrom to benchmark already validated fixtures without rerunning TTSim.
Two block barriers surround each shared-memory K tile; the kernels do not use double buffering,
register tiling, Tensor Cores or cuBLAS. Integer reference checks require exact results.

`-Variants naive,tiled,texture_tiled` enables the new control comparison in the Windows runner.
Texture sampling is point/element type with unnormalized texel-center coordinates, preserving values.
Read path, upload cost and resource creation are recorded separately. See
[23. Texture path and profiler diagnosis](../../docs/23-texture-matmul-and-profiler.md).
This does not implement shared-memory swizzling or an automatic GEMM library.
