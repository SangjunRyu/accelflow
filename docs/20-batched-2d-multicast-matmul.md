# 20. K 배치 기반 2D Multicast MatMul

상태: **구현·정확성·320/640 크기 반복 비교 완료**. 날짜: 2026-10-19.
[진행 순서](README.md) · 이전: [19. Multicast 비교](19-multicast-simulator-performance.md)

## 변경한 데이터 흐름

기존 2D는 A/B 각각 K 타일마다 `receiver ready → payload multicast → valid`를 수행했습니다.
새 모드는 K 방향의 1/4/8타일을 묶어 이 교환을 배치당 한 번 수행합니다.

1. sender/receiver가 CB에서 batch 슬롯을 확보합니다.
2. sender가 여러 DRAM page read를 발행하고 한 번의 read barrier로 완료를 기다립니다.
3. receiver ready를 기다린 뒤 실제 타일 수 × 2048 bytes를 연속 L1 payload로 multicast합니다.
4. payload write barrier 이후 valid를 보내고 배치 전체를 CB에 publish합니다.
5. compute는 batch 전체가 준비되면 타일 인덱스 i를 순서대로 곱하고 배치 슬롯을 pop합니다.

A는 K 방향으로 연속 tile ID, B는 K 방향으로 Nt stride인 tile ID입니다.
따라서 B는 DRAM의 한 연속 덩어리를 읽는 것이 아니라 여러 page read로 연속 L1 영역에 모읍니다.
DRAM page read 개수·payload·수치 연산량은 기존 2D와 같습니다.
줄어드는 것은 read barrier, semaphore/CB 교환 및 payload multicast API 호출 횟수입니다.
multicast API 호출 하나가 NoC packet 하나라는 뜻은 아닙니다. 고정 revision의
`tt_metal/hw/inc/api/dataflow/dataflow_api.h`는 큰 전송을 any-length 경로로 처리합니다.

모든 코어는 기존과 같은 A→B 순서, 2×2 출력 분할과 reader NoC0을 유지합니다.
compute는 output tile당 destination 한 칸을 잡고 전체 K를 같은 순서로 FP32 누적합니다.
중간 결과를 BF16으로 pack하지 않으며 최종 BF16 출력만 pack합니다.
여러 output tile 사이의 추가 A/B 재사용을 도입한 구현은 아닙니다.

## 마지막 배치와 버퍼 크기

K=320이면 Kt=10입니다. batch4는 4+4+2, batch8은 8+2타일을 전송합니다.
마지막에도 고정 batch 슬롯을 reserve/push/pop하여 모든 코어의 ring 주소를 맞춥니다.
미사용 슬롯은 전송하거나 계산하지 않습니다. M/N 경계 코어도 dummy 슬롯을 같은 수 소비합니다.
기존 `dummy_*_slots_model`은 실제 K 위치 기준의 논리 dummy 수이며 마지막 배치의 여분 예약은 제외합니다.
실제 예약은 phase 수 × batch입니다.

입력 CB는 각각 batch×depth타일, 출력 CB는 depth타일입니다.
코어당 CB bytes = 2048×depth×(2×batch+1). depth2에서는 12/36/68 KiB입니다.
이는 커널·semaphore 등 전체 L1 사용량이 아닌 CB payload 용량입니다.
배치4/8은 원래 depth2보다 더 많은 입력 슬롯을 가지므로 시간 변화는 신호 교환 감소와
버퍼·read 발행·CB 사용 방식 변화가 결합된 결과입니다.

## 측정 조건

TT-Metal v0.79.0 / TTSim v1.11.1, BF16 입출력 / FP32 destination / HiFi4입니다.
실제 4코어, depth2에서 독립·A-only·기존2D·새 batch1/4/8을 비교했습니다.
batch1은 새로운 reader/compute 루프 자체의 영향을 드러내는 대조군입니다.
주 비교 입력은 A[320,320]×B[320,320], A[640,320]×B[320,640]입니다.

각 구성은 process 3회, process마다 warmup2+측정5회입니다.
2 shapes × 6 구성 × 3 process = 36 process / 180 timing / 252 dispatch입니다.
trial마다 seed 2026+trial로 순서를 섞고 순차 실행했습니다.
표는 process별 5회 중앙값을 다시 세 process 사이에서 중앙값으로 요약했습니다.
모든 raw samples와 warmup, process별 로그를 보존했습니다.

측정 구간은 C++ steady_clock 기준 EnqueueMeshWorkload 직전부터 Finish 반환까지입니다.
입력 전송/packing, NaN 출력 초기화, readback과 CPU 검증은 제외합니다.
host runtime, slow-dispatch, simulator 처리와 대기는 포함합니다. JIT cache는 삭제하지 않았습니다.
따라서 실물 TT 칩 latency·bandwidth·TOPS와 구분해야 합니다.
독립 구현은 reader NoC1 및 다른 출력 분할을 사용하므로 2D끼리보다 통제 수준이 낮습니다.

## 실행 시간 결과

단위 ms. 비율이 1보다 크면 해당 기준보다 시간이 짧습니다.

| M,K,N | 구현 | 대표 ms | Trial 중앙값 ms | 기존2D/현재 | 독립4/현재 |
| --- | --- | --- | --- | --- | --- |
| 320,320,320 | 독립4 | 512.976 | 510.81/579.24/512.98 | 2.27 | 1.00 |
| 320,320,320 | A-only4 | 882.779 | 902.92/816.26/882.78 | 1.32 | 0.58 |
| 320,320,320 | 기존 2D | 1162.351 | 1307.53/1067.29/1162.35 | 1.00 | 0.44 |
| 320,320,320 | 배치1 대조군 | 1393.987 | 1583.93/1391.59/1393.99 | 0.83 | 0.37 |
| 320,320,320 | 배치4 | 746.244 | 732.26/746.24/751.00 | 1.56 | 0.69 |
| 320,320,320 | 배치8 | 780.125 | 780.12/729.91/903.52 | 1.49 | 0.66 |
| 640,320,640 | 독립4 | 2329.654 | 2029.96/2429.70/2329.65 | 1.99 | 1.00 |
| 640,320,640 | A-only4 | 2806.735 | 2806.73/3199.80/2757.98 | 1.65 | 0.83 |
| 640,320,640 | 기존 2D | 4629.212 | 5223.90/4629.21/4410.14 | 1.00 | 0.50 |
| 640,320,640 | 배치1 대조군 | 6026.675 | 6488.86/6026.67/5401.11 | 0.77 | 0.39 |
| 640,320,640 | 배치4 | 3241.892 | 4157.12/3029.07/3241.89 | 1.43 | 0.72 |
| 640,320,640 | 배치8 | 2975.694 | 2609.17/3178.76/2975.69 | 1.56 | 0.78 |

320,320,320에서는 배치4가 배치4/8 중 짧았고, 기존2D보다 35.8% 줄었습니다. 독립4보다 시간은 길었습니다.

640,320,640에서는 배치8이 배치4/8 중 짧았고, 기존2D보다 35.7% 줄었습니다. 독립4보다 시간은 길었습니다.

배치1 대조군은 기존2D보다 느렸습니다. 새 배치 루프의 런타임 인덱스·반복 및 CB 사용 경로에도 비용이 있습니다.
따라서 배치4/8의 결과는 신호 교환뿐 아니라 전체 배치 구현의 효과로 해석합니다.

## 전송·동기화 모델

아래 값은 소스 루프에서 계산한 모델이며 hardware counter가 아닙니다.
ready atomic은 receiver→sender 증가, valid multicast는 payload multicast와 같은 호출 수입니다.
sender wait/reset 및 write barrier 역시 배치당 수행하므로 함께 감소합니다.

| M,K,N | 구현 | DRAM read KiB | A+B payload multicast 호출 | ready atomic | read barrier | 코어당 CB KiB |
| --- | --- | --- | --- | --- | --- | --- |
| 320,320,320 | 기존 2D | 2000 | 1000 | 1000 | 1000 | 12 |
| 320,320,320 | 배치4 | 2000 | 300 | 300 | 300 | 36 |
| 320,320,320 | 배치8 | 2000 | 200 | 200 | 200 | 68 |
| 640,320,640 | 기존 2D | 8000 | 4000 | 4000 | 4000 | 12 |
| 640,320,640 | 배치4 | 8000 | 1200 | 1200 | 1200 | 36 |
| 640,320,640 | 배치8 | 8000 | 800 | 800 | 800 | 68 |

K=320에서 기존 대비 배치4/8의 handshake 수는 각각 70%/80% 감소합니다.
DRAM read bytes와 receiver payload는 변하지 않고 최종 짧은 배치는 실제 2타일만 전달됩니다.
관측된 시간 차이는 잦은 동기화가 비용에 기여한다는 가설을 평가하는 근거입니다.
장치 cycle/semaphore stall counter를 수집하지 않았으므로 semaphore만이 유일한 병목이라고
단정하거나 신호 감소율만큼 실물 성능이 개선된다고 주장하지 않습니다.
batch8도 모든 구현보다 항상 빠른 최적 설정이라는 보장은 없습니다.

## 정확성과 재현

새 batch1/4/8 × depth1/2 × 4코어의 120개 사례와 batch8의 1/2코어 fallback 80개,
총 200개 정확성 사례가 통과했습니다. 정수 exact, 소수 허용오차, K<=512,
마지막 배치, padding, NaN sentinel, 공식 tilizer 대조를 포함합니다.
큰 입력의 모든 252 dispatch도 출력 일치를 확인했고 최종 출력은 CPU FP64 기준과 exact 비교했습니다.
모드/trial 사이 입력/출력 fingerprint와 전송·CB 모델을 대조했습니다.
기존 baseline/reuse/blocked/streamed 회귀 검증 188개도 모두 통과했습니다.

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/benchmark_batched_matmul.sh
# 기존 연산 회귀 검증
bash scripts/run_matmul.sh
```

새 mode: `mcast2d_batch1`, `mcast2d_batch4`, `mcast2d_batch8`.
실행 인자 구조와 공식 simulator 환경 설정은 기존 모드와 같으며 위 스크립트가 구성합니다.
코드 읽는 순서:
`mcast2d.hpp → kernels/reader_mcast2d_batched.cpp → kernels/compute_mcast2d_batched.cpp`.
writer는 기존 `writer_mcast2d.cpp`를 재사용합니다.

이번 결과:

- [성능·정확성 결과](../results/matmul_batched_perf/20261019T113209749850Z/summary.json)
- [모든 실행 명령·소스 SHA256·환경](../results/matmul_batched_perf/20261019T113209749850Z/metadata.json)
- [개별 timing과 사례](../results/matmul_batched_perf/20261019T113209749850Z/measurements.json)
- [기존 연산 회귀 결과](../results/matmul/20261019T114328718474Z/metadata.json)
