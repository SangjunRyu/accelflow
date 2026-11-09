# 13. MatMul 실행 비용 측정 — WSL/TTSim

상태: **실제 host 시간 측정·비교 완료**. 작업 날짜: 2026-10-07.
[진행 순서](README.md) · 이전: [12. K-streamed MatMul](12-k-streamed-matmul.md)

## 무엇을 측정했는가

물리 Tenstorrent 장치가 없으므로 이 결과는 **현재 WSL/TTSim의 프로그램 제출·완료 대기 비용**입니다.
실물 NPU kernel latency, DRAM bandwidth, TOPS 또는 하드웨어 speedup으로 해석하지 않습니다.
이 측정은 구조 모델과 별개로 실제 host 경과 시간을 관찰한 실험입니다.

| 구간 | 측정 방식과 포함 범위 |
| --- | --- |
| host build | Python `perf_counter`; configure/compile/link의 경과 시간, 별도 메타데이터 |
| 첫 두 dispatch | warmup 시간 기록; JIT/cache 준비가 포함될 수 있음 |
| 반복 dispatch | C++ `steady_clock`; `EnqueueMeshWorkload` 직전부터 `Finish` 반환 직후 |
| process 전체 | Python `perf_counter`; 장치 초기화·JIT·입력 생성/전송·검증·종료 포함 |

반복 dispatch 측정은 workload 객체 생성, 입력 생성/packing, buffer 할당, A/B 입력 전송,
출력 sentinel 초기화, output readback, CPU reference 계산·검증, JSON 저장을 제외합니다.
slow dispatch와 host runtime/IPC/시뮬레이션/완료 대기 비용은 포함합니다.
호스트 시간을 사용하므로 실물 장치의 kernel-only 시간과 같은 구간이 아닙니다.

첫 dispatch는 디스크의 기존 JIT cache를 재사용할 수 있으므로 **cold JIT 시간**으로 부르지 않습니다.
cache를 지우거나 전체 TT-Metal을 재빌드하지 않았습니다.
host build도 기존 빌드 트리를 사용하는 incremental build이며 clean build와 다릅니다.

## 비교 조건과 검증

- TT-Metal v0.79.0, TTSim v1.11.1 Wormhole, Clang 20.1.2
- Intel i7-8565U / WSL2, 동일 core(0,0), HiFi4, BF16 입력·출력, FP32 destination 누적
- 같은 프로그램/workload·주소·입력을 한 process 안에서 재사용
- **각 구성 2회 warmup + 7회 측정**, 합계 36개 구성 / 324개 dispatch
- 구성 실행 순서를 seed 2026으로 섞고 실행 순서와 명령을 저장
- 모든 반복 전에 출력 NaN sentinel을 다시 쓰고 완료시킨 후 타이머 시작
- 매번 출력 전체를 읽어 첫 결과와 비교; 최종 출력은 기존 CPU reference/padding/finite 검사도 통과
- 동일 shape의 모드 사이 입력/출력 fingerprints 일치 확인

입력은 shape 하나만 선택한 뒤 seed 2026으로 생성합니다. 기존 전체 fixture suite에서 같은 shape가
등장할 때의 RNG 위치와 다르므로 이전 JSON의 input fingerprint와 같을 필요는 없습니다.
benchmark의 동일 shape·입력 종류 사이 비교에서는 입력이 같습니다.
K>128에서는 기존 blocked를 실행하지 않았으며 unsupported를 성공으로 세지 않습니다.

결과는 median/p95/min/max/mean/stddev와 raw samples로 저장합니다.
p95는 정렬된 7개 표본에서 `(n-1)*0.95` 위치를 선형 보간합니다.
표본이 7개뿐이므로 장기 p95, 신뢰구간 또는 통계적 유의성으로 해석하지 않습니다.

## 실측 비교

아래는 **depth2** 결과이며 시간 단위는 ms입니다. streamed는 chunk2입니다.
Read/CB 수치는 실측 counter가 아니라 기존 소스 기반 payload 모델입니다.

| M,K,N | Mode | Median ms | p95 ms | Read KiB model | CB KiB model |
| --- | --- | --- | --- | --- | --- |
| [128, 128, 128] | baseline_depth2 | 84.600 | 119.329 | 256.0 | 12.0 |
| [128, 128, 128] | reuse_a_depth2 | 33.065 | 44.636 | 160.0 | 16.0 |
| [128, 128, 128] | blocked_depth2 | 36.702 | 64.531 | 128.0 | 68.0 |
| [128, 128, 128] | streamed_depth2_chunk1 | 38.629 | 43.299 | 128.0 | 20.0 |
| [128, 128, 128] | streamed_depth2_chunk2 | 39.747 | 41.073 | 128.0 | 36.0 |
| [32, 128, 96] | baseline_depth2 | 14.917 | 17.484 | 48.0 | 12.0 |
| [32, 128, 96] | reuse_a_depth2 | 11.996 | 13.067 | 32.0 | 16.0 |
| [32, 128, 96] | blocked_depth2 | 12.966 | 14.021 | 40.0 | 68.0 |
| [32, 128, 96] | streamed_depth2_chunk1 | 19.707 | 20.429 | 40.0 | 20.0 |
| [32, 128, 96] | streamed_depth2_chunk2 | 13.366 | 17.038 | 40.0 | 36.0 |
| [64, 512, 64] | baseline_depth2 | 52.793 | 56.871 | 256.0 | 12.0 |
| [64, 512, 64] | reuse_a_depth2 | 59.288 | 69.704 | 192.0 | 40.0 |
| [64, 512, 64] | streamed_depth2_chunk1 | 37.822 | 38.667 | 128.0 | 20.0 |
| [64, 512, 64] | streamed_depth2_chunk2 | 30.774 | 53.435 | 128.0 | 36.0 |
| [33, 257, 65] | baseline_depth2 | 60.289 | 75.082 | 216.0 | 12.0 |
| [33, 257, 65] | reuse_a_depth2 | 42.699 | 50.411 | 144.0 | 26.0 |
| [33, 257, 65] | streamed_depth2_chunk1 | 38.535 | 40.526 | 126.0 | 20.0 |
| [33, 257, 65] | streamed_depth2_chunk2 | 31.235 | 51.085 | 126.0 | 36.0 |

### 결과에서 읽을 수 있는 것

128×128에서 A 재사용의 중앙값은 baseline보다 짧았지만, read 모델이 더 작은 blocked가
A 재사용보다 더 짧지는 않았습니다. 메모리 전송량 모델 하나만으로 host/simulator 실행 시간을
예측할 수 없다는 관찰입니다. 이를 실물 칩의 bottleneck 분석으로 일반화하지 않습니다.

32×128×96에서는 출력 tile row가 하나여서 B의 행 방향 재사용이 없습니다.
A 재사용이 입력 read 모델 32KiB, blocked/streamed가 40KiB입니다.
같은 shape에서 A 재사용이 가장 짧은 depth2 중앙값을 보였습니다.

K=512에서는 streamed chunk2/depth2가 36KiB CB로 실행됐고 baseline depth2보다 짧은
관측 중앙값을 보였습니다. 동일 입력 read 모델인 chunk1/2 사이에도 시간이 다릅니다.
이는 chunk 크기와 host/simulator 동기화 비용을 함께 검토할 이유이며 물리 overlap의 증거는 아닙니다.

### 변동과 제한

128×128 baseline depth2의 min/max는 약 55.9/120.7ms로 차이가 큽니다.
blocked depth2도 median 약 36.7ms에 p95 약 64.5ms입니다.
WSL scheduling, host CPU 부하, simulator/IPC 처리 등이 측정에 영향을 줄 수 있습니다.
실험 전후 `/proc/meminfo`, `/proc/loadavg`, CPU/OS/compiler 정보를 저장했지만
CPU affinity·frequency·백그라운드 부하를 고정하지 않았습니다.

샘플은 각 구성별 한 process의 연속 반복입니다. 구성을 섞었어도 구성 사이 장기 시간 변화와
process 초기화 차이를 완전히 제거한 실험은 아닙니다. 일부 depth1/2 결과는 순서가 뒤바뀌므로
모든 shape에서 depth2가 더 빠르다고 주장하지 않습니다.
`baseline median / variant median` 비율은 이 측정 환경의 관측 비율이며 실물 NPU speedup이 아닙니다.

## 코드와 재현

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/benchmark_matmul.sh
# 필요하면 반복 표본 수 변경 (5..25)
bash scripts/benchmark_matmul.sh --samples 15
```

- [benchmark.hpp](../npu/matmul/benchmark.hpp): 반복 제출, steady_clock 구간, 출력 재검사
- [benchmark_matmul.py](../tools/report/benchmark_matmul.py): pinned 환경, 구성 순서, 메타데이터, 요약 통계
- [matmul.cpp](../npu/matmul/matmul.cpp): 기존 정확성 suite와 별도의 benchmark 경로

직접 실행 파일의 benchmark 선택자는 `name:M,K,N`입니다. 예:
`matmul report.json streamed 2 2 --benchmark 7 signed_random:128,128,128`.
장치 환경 변수/descriptor는 wrapper가 기존 공식 slow-dispatch simulator 설정으로 준비합니다.

- [메타데이터·실행 순서·process/build 시간](../results/matmul_perf/20261007T105114756148Z/metadata.json)
- [전체 36개 비교 JSON](../results/matmul_perf/20261007T105114756148Z/comparison.json)
- [전체 36개 비교 표](../results/matmul_perf/20261007T105114756148Z/comparison.md)
- [128×128 baseline raw timing](../results/matmul_perf/20261007T105114756148Z/128_128_128_baseline_depth2.json)
- [128×128 A 재사용 raw timing](../results/matmul_perf/20261007T105114756148Z/128_128_128_reuse_a_depth2.json)
- [128×128 streamed raw timing](../results/matmul_perf/20261007T105114756148Z/128_128_128_streamed_depth2_chunk2.json)
- [전체 sample 실행 로그 예](../results/matmul_perf/20261007T105114756148Z/128_128_128_streamed_depth2_chunk2.log)
- [host 빌드 로그](../results/matmul_perf/20261007T105114756148Z/build.log)

측정 경로 추가 후 기존 **188개 정확성 suite도 다시 통과**했습니다.
[정확성 회귀 검증 메타데이터](../results/matmul/20261007T105319808353Z/metadata.json)와
[회귀 설계 평가](../results/matmul/20261007T105319808353Z/design-assessment.json)를 보존했습니다.
일반 실행에서는 반복 benchmark를 수행하지 않습니다.

## 다음 실측 단계

실물 Tenstorrent 장치에서 profiler/device timing을 사용할 수 있게 되면 hardware kernel/transfer를
분리해 측정해야 합니다. CUDA 비교도 실제 GPU의 CUDA events로 측정하고 datatype/누적 정밀도/
전송 포함 범위를 맞춰야 합니다. 이 문서의 simulator 시간과 CUDA kernel 시간으로 장치 간
속도 비율을 계산하지 않습니다.
