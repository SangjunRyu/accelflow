# 16. Multicore MatMul — TTSim 수준의 실행 비용 비교

상태: **Multicore TTSim 반복 측정·독립 재측정 완료**. 작업 날짜: 2026-10-19.
[진행 순서](README.md) · 이전: [15. 독립 multicore 구현](15-independent-multicore-matmul.md)

## 이번 측정으로 확인하려는 것

단순 multicore의 작업 분배가 현재 WSL/TTSim의 관측 실행 시간에 어떤 차이를 만드는지 측정합니다.
정확성 검사만으로 끝내지 않고, 같은 입력에서 코어 수·CB 깊이를 바꾼 결과와 변동을 보존합니다.

아래 세 종류의 근거를 구분합니다.

| 근거 | 확인 가능한 내용 |
| --- | --- |
| CPU/출력 검증 | 분배한 일을 실제로 완료하고 같은 결과를 냈는가 |
| host steady_clock 실측 | 현재 host/runtime/simulator의 제출·완료 비용이 어떻게 달라졌는가 |
| 코드/shape 기반 모델 | 코어별 최대 작업량, 전체 DRAM payload, CB 저장 용량이 어떻게 바뀌었는가 |

시뮬레이터의 host 경과 시간에서 물리 TT 칩의 cycle/메모리 대역폭/병렬 speedup을 추론하지 않습니다.
하지만 같은 환경에서 실행 비용의 변화·변동·구성별 차이를 관찰하는 용도로는 사용할 수 있습니다.

## 측정 구간

기존 [benchmark.hpp](../npu/matmul/benchmark.hpp)를 multicore에도 사용합니다.

```text
출력 NaN sentinel 초기화 + Finish                    [측정 밖]
    start = host steady_clock
    EnqueueMeshWorkload(같은 workload)
    Finish(전체 코어의 workload 완료 대기)
    end = host steady_clock                         [측정 종료]
출력 readback + 모든 반복 결과 비교                  [측정 밖]
최종 CPU reference / finite / padding 검사           [측정 밖]
```

입력 생성·packing·업로드, buffer/CB/커널/workload 생성, 출력 전송·CPU 검증은 timing 구간 밖입니다.
단, slow dispatch, runtime, host IPC, simulator 처리와 완료 대기는 포함됩니다.
첫 두 제출은 warmup이며 저장된 JIT cache를 재사용할 수 있어 cold JIT 측정은 아닙니다.
별도 process 전체 시간과 incremental host build 시간도 메타데이터에 저장합니다.

## 실험 설계

| 항목 | 설정 |
| --- | --- |
| 입력 | 아래 4개 fixture, 각 process에서 동일 seed 2026 |
| 구성 | baseline + multicore 요청 1/2/4 × CB depth1/2 = 8개 |
| 독립 process 반복 | 구성별 3회 |
| process 내부 | 2회 warmup 후 7회 timing |
| 실행 순서 | 각 trial 안에서 seed 2026+trial로 섞음; 작업은 순차 실행 |
| 동일 조건 | TT-Metal v0.79.0 / TTSim v1.11.1 / HiFi4 / BF16 입출력 / FP32 누적 |

총 **32개 고유 구성, 96개 process trial, 672개 timing 표본, 864개 dispatch**입니다.
모든 반복마다 출력을 초기화하고 결과를 읽어 이전 반복과 대조합니다.
최종 출력은 CPU FP64 reference·공식 tilizer·padding/finite 검사도 통과해야 합니다.
입력/출력 fingerprints는 구성과 process trial 사이에서도 같아야 합니다.

| M,K,N | 목적 | 출력 타일 수 |
| --- | --- | --- |
| 128,128,128 | 4코어 균등 분배 | 16 |
| 64,64,96 | 4코어로 나누면 2/2/1/1, 출력 행 경계 포함 | 6 |
| 64,512,64 | 큰 K 누적 | 4 |
| 2,3,2 | 출력 한 타일, 활성 코어를 늘릴 수 없는 대조군 | 1 |

`multicore 요청 1코어`를 2/4코어의 주 비교 기준으로 삼습니다.
기존 baseline도 함께 측정하지만 커널 소스가 다르므로 baseline→multicore의 변화에는
작업 분배 이외의 코드 차이도 섞입니다. 코어 수 비교에서는 같은 CB 깊이를 맞춥니다.

## 대표 시간과 통계

대표 시간은 **각 process의 7개 timing 중앙값을 구한 뒤, 독립 process 3개 중앙값의 중앙값**입니다.
3개 trial 중앙값도 모두 표에 남깁니다. p95/min/max는 21개 timing을 모은 기술 통계입니다.
p95는 `(n-1)*0.95` 위치를 선형 보간합니다.

한 process 안의 표본은 서로 관련될 수 있고 독립 trial도 3개뿐이므로 통계적 유의성이나
장기 tail latency를 입증한 것으로 보지 않습니다. CPU affinity/frequency·Windows background load는
고정하지 않았으며 process 순서/부하와 host 상태를 저장합니다.

## 실측 결과

아래 표는 **CB depth2**, 단위 ms입니다. `MC1 / 현재`는 같은 depth의 multicore 1코어
대표 시간을 현재 대표 시간으로 나눈 관측 비율입니다. 1보다 크면 이번 TTSim/host에서 더 짧았습니다.
물리 NPU speedup이나 코어 효율 지표를 뜻하지 않습니다.

| M,K,N | 구성 | 실제 활성 코어 | 대표 ms | Trial 중앙값 ms | pooled p95 ms | MC1 / 현재 |
| --- | --- | --- | --- | --- | --- | --- |
| [128, 128, 128] | baseline_cores1_depth2 | 1 | 53.058 | 47.59/57.02/53.06 | 59.311 | 1.04 |
| [128, 128, 128] | multicore_cores1_depth2 | 1 | 55.057 | 53.76/55.06/59.82 | 64.714 | 1.00 |
| [128, 128, 128] | multicore_cores2_depth2 | 2 | 40.317 | 38.40/40.32/41.22 | 47.161 | 1.37 |
| [128, 128, 128] | multicore_cores4_depth2 | 4 | 38.416 | 34.23/38.42/39.08 | 43.200 | 1.43 |
| [64, 64, 96] | baseline_cores1_depth2 | 1 | 14.110 | 12.91/14.11/15.56 | 15.984 | 0.98 |
| [64, 64, 96] | multicore_cores1_depth2 | 1 | 13.791 | 13.70/14.02/13.79 | 14.899 | 1.00 |
| [64, 64, 96] | multicore_cores2_depth2 | 2 | 11.370 | 10.73/11.37/12.22 | 15.262 | 1.21 |
| [64, 64, 96] | multicore_cores4_depth2 | 4 | 11.192 | 10.20/12.74/11.19 | 13.229 | 1.23 |
| [64, 512, 64] | baseline_cores1_depth2 | 1 | 50.330 | 50.20/50.33/56.31 | 57.098 | 1.00 |
| [64, 512, 64] | multicore_cores1_depth2 | 1 | 50.266 | 50.27/49.63/54.05 | 56.089 | 1.00 |
| [64, 512, 64] | multicore_cores2_depth2 | 2 | 40.691 | 36.16/40.69/42.61 | 47.309 | 1.24 |
| [64, 512, 64] | multicore_cores4_depth2 | 4 | 35.756 | 34.20/35.76/38.09 | 39.630 | 1.41 |
| [2, 3, 2] | baseline_cores1_depth2 | 1 | 5.574 | 4.96/5.57/6.57 | 6.671 | 0.90 |
| [2, 3, 2] | multicore_cores1_depth2 | 1 | 5.039 | 5.04/4.68/6.03 | 6.715 | 1.00 |
| [2, 3, 2] | multicore_cores2_depth2 | 1 | 4.982 | 5.28/4.76/4.98 | 5.592 | 1.01 |
| [2, 3, 2] | multicore_cores4_depth2 | 1 | 4.861 | 4.79/5.15/4.86 | 5.730 | 1.04 |

### 이번 재측정의 대표 관찰

128×128, depth2에서 동일 multicore 커널은 1코어 55.057ms, 2코어 40.317ms,
4코어 38.416ms였습니다. 1→4코어의 관측 시간 감소는 약 30.2%, 시간 비율은 약 1.43입니다.
코어별 최대 출력 작업량 모델은 16→4타일로 줄지만, 관측 시간이 4배 줄지는 않았습니다.

K=512 입력에서는 50.266→40.691→35.756ms로, 1→4코어 시간 비율이 약 1.41이었습니다.
6개 출력 타일 입력은 13.791→11.370→11.192ms로 2→4코어의 추가 감소가 작았습니다.
분배상 최대 작업량은 6→3→2타일이며 남는 타일의 불균형과 전체 DRAM payload 유지도 함께 봐야 합니다.
이 관찰만으로 특정 병목의 원인을 확정하지는 않습니다.

한 타일 대조군은 실제 활성 코어가 모두 1개이고 관측 대표 시간이 약 5ms였습니다.
이를 순수 launch overhead로 단정하거나 다른 측정에서 일괄 차감하지 않습니다.

### 재측정과 관측 변동

첫 번째 세트에서 단일 타일 대조군과 일부 process에 큰 변동이 관측돼 동일한 설정으로 전체를 재측정했습니다.
첫 세트에는 환경 확인용 별도 WSL 명령이 실행된 구간도 있어 host 부하 영향을 배제할 수 없습니다.
재측정에서는 추가 환경 조사를 멈추고 benchmark 작업을 순차 실행했습니다. Windows background 부하는 여전히 통제하지 않았습니다.
위 표는 두 번째 세트이며 첫 번째 세트도 아래 링크와 비교 표에 보존합니다. 더 짧은 결과만 골라 합치지 않았습니다.
두 세트 합계는 **192개 process trial / 1,344개 timing 표본 / 1,728개 dispatch**입니다.

| M,K,N | depth | 첫 세트 MC1 / MC4 관측 시간 비율 | 재측정 MC1 / MC4 비율 |
| --- | --- | --- | --- |
| [128, 128, 128] | 1 | 1.98 | 1.52 |
| [128, 128, 128] | 2 | 0.84 | 1.43 |
| [64, 64, 96] | 1 | 1.34 | 1.37 |
| [64, 64, 96] | 2 | 0.55 | 1.23 |
| [64, 512, 64] | 1 | 1.75 | 1.57 |
| [64, 512, 64] | 2 | 1.11 | 1.41 |
| [2, 3, 2] | 1 | 0.91 | 0.89 |
| [2, 3, 2] | 2 | 1.03 | 1.04 |

비율 1 초과는 해당 세트의 4코어 관측 시간이 짧았다는 뜻이며, 1 미만은 더 길었다는 뜻입니다.
단일 타일 대조군의 요청 1/2/4는 실제로 모두 한 코어만 사용합니다. 그 사이 차이는 병렬화 이득이 아닙니다.
큰 입력에서 코어별 최대 작업량은 분명 줄지만 시간 변화가 그 비율을 따르지는 않습니다.
TTSim/host의 실행 경로와 부하·프로세스별 변동이 함께 포함돼 있어, 개선 폭이 작거나 세트 간 방향이 바뀌면
안정적인 성능 우열을 확정하지 않습니다. CB 깊이가 커지면 항상 더 짧아진다는 주장도 하지 않습니다.
특정 원인을 stall/NoC 병목으로 단정할 장치 counter는 이번 실험에 없습니다.


## 시간과 별도로 확인한 작업량·메모리

`T=Mt×Nt`, 활성 코어 수 `P=min(requested,T)`, 코어별 최대 출력 수 `L=ceil(T/P)`입니다.
동일 출력의 K 전체는 한 코어에서 계산하며 코어 간 부분합/입력 교환은 없습니다.

| M,K,N | P=1 최대 타일/코어 | P=2 | P=4 | 4코어의 작업 분배 계수 T/L |
| --- | --- | --- | --- | --- |
| 128,128,128 | 16 | 8 | 4 | 4 |
| 64,64,96 | 6 | 3 | 2 | 3 |
| 64,512,64 | 4 | 2 | 1 | 4 |
| 2,3,2 | 1 | 1 | 1 | 1 |

작업 분배 계수는 코어별 최대 산술 작업량이 얼마나 줄었는지를 나타내는 **모델**입니다.
메모리 병목·동기화·실행 비용을 포함하지 않으므로 예측 speedup으로 부르지 않습니다.
특히 6타일/4코어는 일부 코어가 두 타일을 담당하므로 이 계수가 4가 아닌 3입니다.

전체 DRAM read payload는 모든 코어 수에서 `4096×T×Kt`로 같습니다.
CB depth2의 payload 용량은 코어별 12KiB, 합계는 활성 1/2/4코어일 때 12/24/48KiB입니다.
합계는 서로 다른 코어의 L1에 나뉘며 전체 L1 사용량을 뜻하지 않습니다.
코어별 최대 matmul 호출 수 `L×Kt`, 전역 read 모델, CB 합계, 타일 구간을 실제 보고서와 대조했습니다.

이 비교는 sender/receiver 도입 전의 기준선입니다. 코어 수만 늘리는 경우에는 전체 입력 전송량이
줄지 않으며, 입력 공유 방식은 별도의 NoC/semaphore 동기화를 추가해 비교해야 합니다.

## 재현과 증거

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/benchmark_multicore_matmul.sh
# 간단 재실행: process trial 1회, timing 7회
bash scripts/benchmark_multicore_matmul.sh --trials 1 --samples 7
```

`--samples`는 5..25, `--trials`는 1..5입니다.
실행 파일 직접 인자는 예를 들어
`matmul report.json multicore 2 4 --benchmark 7 signed_random:128,128,128`입니다.
wrapper가 기존 공식 simulator 환경 변수·SoC descriptor를 준비합니다.

- [재측정 메타데이터·실행 순서·버전·SHA-256](../results/matmul_multicore_perf/20261019T105537249027Z/metadata.json)
- [재측정 전체 비교 JSON](../results/matmul_multicore_perf/20261019T105537249027Z/comparison.json)
- [재측정 전체 비교 표](../results/matmul_multicore_perf/20261019T105537249027Z/comparison.md)
- [128×128 1코어 trial1 원시 표본](../results/matmul_multicore_perf/20261019T105537249027Z/128_128_128_multicore_cores1_depth2_trial1.json)
- [128×128 4코어 trial1 원시 표본](../results/matmul_multicore_perf/20261019T105537249027Z/128_128_128_multicore_cores4_depth2_trial1.json)
- [4코어 trial1 전체 로그](../results/matmul_multicore_perf/20261019T105537249027Z/128_128_128_multicore_cores4_depth2_trial1.log)
- [첫 측정 세트의 전체 비교](../results/matmul_multicore_perf/20261019T105310793162Z/comparison.json)
- [첫 측정 메타데이터](../results/matmul_multicore_perf/20261019T105310793162Z/metadata.json)

원시 JSON의 `timing.samples_ms`/`warmup_ms`, 코어 분배, CPU 비교 결과와 각 trial의 전체 로그를
보존했습니다. compiler/CPU/OS, 명령·실행 순서·exit code·source/executable SHA-256도 저장합니다.
모든 구성은 같은 실행 파일을 사용합니다.

## 해석 범위

이 실험으로 현재 구현의 TTSim 실행 비용을 비교하고, 작업량 감소와 관측 시간 변화가
항상 같은 비율은 아니라는 점을 평가할 수 있습니다. 실제 TT 칩의 kernel latency, NoC bandwidth,
stall/overlap 또는 4코어 물리 병렬 효율은 장치 profiler/timing이 필요합니다.
결과는 10월 7일 단일 코어 측정과 합쳐 속도 비율을 만들지 않고 이번 같은 실험 안에서 비교합니다.
