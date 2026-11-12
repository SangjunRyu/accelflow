# AccelFlow project specification

The canonical project specification is [AccelFlow_PROJECT.md](AccelFlow_PROJECT.md).
This pointer preserves the original IDE document without maintaining two copies.
Current status: Phase 4 single-core NPU RMSNorm and CUDA unfused/fused RMSNorm validated; PyTorch CPU reference comparison passed; Single-core NPU fusion passed; row- and column-partitioned 1/2/4-core RMSNorm passed; Metalium checkpoint summarized in docs/29-metalium-results.md; TT-NN comparison remains. See [progress](docs/README.md).

Public portfolio tables and evidence/simulator entry point: [docs/30-portfolio.md](docs/30-portfolio.md).

Separate FP32-to-Dest precision experiment: native API path rejected by pinned simulator/LLK combination; local compatibility path passed, without residual-bit splitting. See [31](docs/31-fp32-to-dest-rmsnorm.md).
