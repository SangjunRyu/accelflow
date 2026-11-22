# RTX 4060 Ti CUDA MatMul Nsight 분석

2026-11-22 · GeForce RTX 4060 Ti (Ada, SM 8.9) · Windows · CUDA NVRTC 13.5 · Nsight Systems 2026.5.2 · Nsight Compute 2026.4.0

**16×16 Tiled 커널은 640×320×640 FP32 행렬곱에서 Naive보다 GPU 이벤트 시간이 23.9% 짧았다.** Nsight Compute에서 L1/TEX 글로벌 로드 요청이 **819.2만 → 51.2만 건**으로 줄었고, 공유 메모리 로드·스토어 뱅크 충돌 카운터는 **0**이었다. DRAM 읽기량은 약 **1.64 MB**로 두 커널이 비슷했다. 이 입력에서 타일링의 효과는 글로벌 로드 명령과 캐시 요청 감소에 뚜렷하게 나타났다.

## 실험 조건과 정합성

AccelFlow 커밋 **84e6306**의 **cuda/matmul/matmul.cu**에서 **naive**와 **tiled<16, false>** 커널 정의를 그대로 사용했다. 이 PC에서 MSVC를 찾지 못해 NVRTC로 SM 8.9 CUBIN을 만들고 Python 3.12와 CUDA Driver API로 실행했다. 두 커널 모두 16×16 스레드 블록, 동일한 CUDA 스트림 구성, 고정 페이지 호스트 메모리를 사용했다.

같은 시드에서 생성한 정수값 −8~8을 FP32로 저장했다. 640×320×640과 128³에서 두 커널의 출력이 FP64 기준과 정확히 일치했다. 프로파일 실행도 모두 출력 검증을 통과했다. 기존 MX150 기록은 BF16에서 펼친 fixture와 C++ 호스트를 사용했으므로, 아래 속도 비교는 RTX 4060 Ti 내에서 해석한다.

## 프로파일러를 끈 기준 측정

각 조건을 별도 프로세스로 3회 실행했다. 프로세스마다 워밍업 20회 후 20개 커널 실행을 CUDA 이벤트로 묶어 표본 15개를 얻었다. 표는 프로세스 내 중앙값을 구한 뒤 그 세 값의 중앙값이다. 호스트 파이프라인은 이미 할당한 버퍼로 **H2D → 커널 → D2H → 스트림 완료 대기**를 15회 반복했다.

| M×K×N | Naive GPU 이벤트 (ms) | Tiled GPU 이벤트 (ms) | Naive 호스트 파이프라인 (ms) | Tiled 호스트 파이프라인 (ms) |
|---|---:|---:|---:|---:|
| 640×320×640 | 0.21898 | 0.16660 | 0.5426 | 0.5354 |
| 128×128×128 | 0.00609 | 0.00538 | 0.0894 | 0.0927 |

큰 입력에서 Tiled 커널은 세 프로세스 쌍 모두 빨랐다. 호스트 파이프라인 대표값의 감소는 약 **1.3%**였고, 프로세스별 값은 Naive **0.5415~0.5864 ms**, Tiled **0.4873~0.5513 ms**로 겹쳤다. 작은 입력에서도 호스트 시간 차이는 수 µs였다.

## Nsight Systems 타임라인

640×320×640에서 커널별로 워밍업 20회, CUDA 이벤트 구간 40회, H2D·D2H를 포함한 파이프라인 4회를 캡처했다. SQLite에서 커널과 복사가 같은 스트림에 기록된 것을 확인했다.

| GPU 타임라인 지표 | Naive | Tiled |
|---|---:|---:|
| 커널 실행 수 | 64 | 64 |
| 커널 중앙값 | 199.30 µs | 152.06 µs |
| H2D 복사 1건 중앙값 | 67.52 µs | 67.56 µs |
| D2H 복사 1건 중앙값 | 124.86 µs | 124.89 µs |

캡처에서도 Tiled의 커널 시간이 짧았고, 입력 복사 두 건과 출력 복사 한 건은 두 구현에서 거의 같은 시간이 걸렸다. 커널 시간의 차이가 복사와 완료 대기를 포함한 호스트 파이프라인에서는 작아지는 결과와 일관된다. 표의 중앙값은 활동별 분포에서 따로 집계했다.

## Nsight Compute 하드웨어 카운터

640×320×640에서 워밍업 20회 뒤 커널 실행 한 건씩을 선택했다. 도구는 카운터 그룹을 모으기 위해 각각 38패스로 재실행했다. 기본 캐시 초기화 옵션을 사용했고 GPU 클럭 고정은 끈 상태로 수집했다.

| 커널당 지표 | Naive | Tiled |
|---|---:|---:|
| GPU 커널 시간 | 200.80 µs | 153.70 µs |
| L1/TEX 글로벌 로드 요청 | 8,192,000 | 512,000 |
| L1/TEX 글로벌 로드 섹터 | 16,387,524 | 2,038,884 |
| 글로벌 로드 섹터 중 L1/TEX 미스 | 1,765,541 | 1,829,118 |
| DRAM 읽기 | 1.644 MB | 1.642 MB |
| 정적 공유 메모리 / 블록 | 0 | 2 KiB |
| 공유 메모리 로드 / 스토어 뱅크 충돌 | 0 / 0 | 0 / 0 |
| 달성 점유율 | 94.66% | 95.05% |

타일을 블록당 한 번씩 읽고 공유 메모리에서 재사용하면서 글로벌 로드 요청은 **16분의 1**, L1/TEX 글로벌 로드 섹터는 **약 8분의 1**이 됐다. Naive의 반복 로드는 L1/TEX가 상당 부분 흡수했다. 두 커널의 L1/TEX 미스 섹터 수와 DRAM 읽기량이 비슷한 이유를 설명해 준다.

두 커널의 달성 점유율은 약 95%였다. Nsight의 워프 상태 요약은 Naive에서 글로벌·로컬 메모리 명령 큐 대기인 **LG throttle**, Tiled에서 공유 메모리 등 MIO 명령 큐 대기인 **MIO throttle**을 주요 항목으로 표시했다. Tiled에서 공유 메모리 명령을 추가하고도 뱅크 충돌 카운터는 0이었다. MX150에서 주소 패턴으로 추정했던 16×16 타일의 뱅크 충돌 가설을 RTX 4060 Ti의 해당 실행에서 실측한 결과다.

## Nsight Compute GUI 재현

Start Activity에서 Naive와 Tiled를 각각 실행했다. 두 GUI 캡처 모두 워밍업 20회와 측정 20회를 포함한다. 아래 표는 각 리포트에서 앞 20건을 제외한 커널 시간이다.

| 커널 | 중앙값 (µs) | 최소~최대 (µs) |
|---|---:|---:|
| Naive | 200.752 | 200.320~201.088 |
| Tiled | 153.616 | 153.504~153.792 |

**Tiled는 47.136 µs(23.5%) 짧았고 두 구현의 출력은 FP64 기준과 정확히 일치했다.** GUI 캡처에는 글로벌 로드 요청과 뱅크 충돌 원시 카운터가 포함되지 않았다. 본문의 카운터는 해당 지표를 지정한 1건 상세 캡처에서 집계했다.

## 파일 열기와 재현

- **Nsight Systems**에서 두 **.nsys-rep**를 열고 CUDA GPU의 Kernels와 Memory, CUDA API 행을 함께 확인한다.
- **Nsight Compute**에서 두 **.ncu-repz**를 열고 Memory Workload Analysis, Warp State Statistics, Raw를 확인한다. CSV에는 본문 지표의 원래 카운터 이름과 단위가 들어 있다.
- [실험 산출물](../results/profiling/20261122-rtx4060ti-matmul/README.md)에 Compute 원본 및 PDF, Systems SQLite·CSV, 반복 측정과 검증 JSON을 모았다. [재현 절차](../results/profiling/20261122-rtx4060ti-matmul/reproduce/README.md)에서 실행 명령을 확인할 수 있다.
- CUDA 커널 실행과 두 캡처는 정상 완료됐다. 이 세션의 별도 **nvidia-smi** 호출은 NVML 초기화 오류를 반환해 온도와 열 제한 이벤트는 이번 MatMul 측정에 포함하지 않았다.

[Nsight Compute GPU 지원표](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/gpu-support.html) · [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) · [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
