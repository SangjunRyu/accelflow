# Vector Add design assessment

Evidence: `results/vector_add/20261006T105510834955Z`

| Implementation | Passed cases |
| --- | --- |
| uint32 RISC-V | 10 |
| BF16 CB depth 1 | 14 |
| BF16 CB depth 2 | 14 |

Counts below are analytical API/payload models, not profiler counters.

| BF16 N | Tiles | Occupancy | Padding amplification | Transfer API calls | Read/write barriers |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | 0.391% | 256.0000x | 3 | 2 |
| 33 | 1 | 3.223% | 31.0303x | 3 | 2 |
| 1 | 1 | 0.098% | 1024.0000x | 3 | 2 |
| 31 | 1 | 3.027% | 33.0323x | 3 | 2 |
| 32 | 1 | 3.125% | 32.0000x | 3 | 2 |
| 1023 | 1 | 99.902% | 1.0010x | 3 | 2 |
| 1024 | 1 | 100.000% | 1.0000x | 3 | 2 |
| 1025 | 2 | 50.049% | 1.9980x | 6 | 4 |
| 2048 | 2 | 100.000% | 1.0000x | 6 | 4 |
| 2051 | 3 | 66.764% | 1.4978x | 9 | 6 |
| 8193 | 9 | 88.900% | 1.1249x | 27 | 18 |
| 65537 | 65 | 98.463% | 1.0156x | 195 | 130 |

CB depth 1: 6144 B; depth 2: 12288 B; payload unchanged.
Correctness and CB-slot reuse passed. Actual overlap, stalls and hardware speedup remain unmeasured.
