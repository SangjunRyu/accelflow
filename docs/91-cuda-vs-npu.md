# 91. CUDA versus NPU — 참고 문서

[진행 순서와 문서 안내](README.md)

| CUDA | TT-Metal concept | Scope |
| --- | --- | --- |
| Global memory | DRAM | Bulk storage |
| Shared memory | L1 / circular buffers | Local reuse; different management |
| Async copies | Reader/writer NoC transfers | Different programming abstraction |
| Block synchronization | Buffer/kernel synchronization | Different scope and semantics |

Detailed comparisons will follow verified kernels; these are conceptual mappings.
