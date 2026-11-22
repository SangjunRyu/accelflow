# RTX 4060 Ti CUDA MatMul 프로파일

2026-11-22 · AccelFlow Naive와 16×16 Tiled · GeForce RTX 4060 Ti (SM 8.9)

[측정 결과와 해석](../../../docs/33-rtx4060ti-matmul-nsight.md) · [재현 절차](reproduce/README.md)

| 도구와 수집 방식 | Naive | Tiled |
|---|---|---|
| Nsight Compute · 카운터 상세 1회 | [원본](compute/naive.ncu-repz) · [원시 CSV](compute/naive-compute-raw.csv) · [상세 텍스트](compute/naive-compute-details.txt) | [원본](compute/tiled.ncu-repz) · [원시 CSV](compute/tiled-compute-raw.csv) · [상세 텍스트](compute/tiled-compute-details.txt) |
| Nsight Compute · GUI 40회 | [원본](compute/naive-gui.ncu-repz) · [Summary PDF](exports/naive-gui-summary.pdf) · [Details PDF](exports/naive-gui-details.pdf) · [Raw PDF](exports/naive-gui-raw.pdf) | [원본](compute/tiled-gui.ncu-repz) · [Summary PDF](exports/tiled-gui-summary.pdf) · [Details PDF](exports/tiled-gui-details.pdf) · [Raw PDF](exports/tiled-gui-raw.pdf) |
| Nsight Systems · 64회 타임라인 | [원본](systems/naive.nsys-rep) · [SQLite](systems/naive.sqlite) · [커널 CSV](systems/naive-systems_cuda_gpu_kern_sum.csv) | [원본](systems/tiled.nsys-rep) · [SQLite](systems/tiled.sqlite) · [커널 CSV](systems/tiled-systems_cuda_gpu_kern_sum.csv) |

원본 리포트는 각 Nsight 도구의 **File → Open**에서 연다. 카운터 상세 캡처는 640×320×640 워밍업 이후 커널 1회를 선택했다. GUI 캡처는 워밍업과 측정 실행 각각 20회를 포함한다.

[validation](validation/)에는 정답 검증과 프로파일러를 끈 반복 측정 JSON을 보존했다. Systems의 복사 및 CUDA API 집계 CSV는 [systems](systems/)에 있다.
