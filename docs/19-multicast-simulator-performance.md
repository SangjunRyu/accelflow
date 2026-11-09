# 19. 독립·A-only·2D Multicast MatMul 실행 비용 비교

상태: **32개 구성·96개 process trial 측정 및 동일 출력 검증 완료**. 작업 날짜: 2026-10-19.
[진행 순서](README.md) · 이전: [18. 2D multicast](18-2d-multicast-matmul.md)

## 무엇을 비교했는가

같은 입력·정밀도·실제 활성 코어 수·CB 깊이에서 다음 세 구현의 TTSim 실행 비용을 비교했습니다.

- `multicore`: 각 코어가 A/B를 DRAM에서 직접 읽는 독립 방식
- `mcast`: A를 행 방향으로 공유하고 B는 직접 읽는 A-only 방식
- `mcast2d`: A 행 공유 + B 열 공유의 2차원 방식

주 비교의 세 전략은 **모두 실제 4개 활성 코어**를 사용합니다.
독립 multicore 1코어도 대조군으로 측정하되, 4코어 전략 간 비율의 기준은 같은 depth의 독립 4코어입니다.

이 결과는 기존 구현 전체의 비교입니다. 독립 버전은 reader RISCV_1/NoC1,
multicast 버전은 reader RISCV_0/NoC0이며 출력 타일 분배도 다릅니다.
따라서 multicast 사용 여부만 바꾼 실험이라고 주장하지 않습니다.
마찬가지로 host 시간을 물리 TT 칩 kernel latency·DRAM bandwidth·speedup으로 해석하지 않습니다.

## 측정 조건과 검증

| 항목 | 설정 |
| --- | --- |
| 환경 | 고정 TT-Metal v0.79.0 / TTSim v1.11.1 Wormhole |
| 수치 조건 | BF16 입력·출력, FP32 destination 누적, HiFi4 |
| 구성 | 독립1/독립4/A-only4/2D4 × CB depth1/2 = 8개 |
| 입력 | 아래 4개 shape, 동일 seed 2026의 signed random |
| 독립 반복 | 구성마다 별도 process 3회 |
| process 내부 | warmup 2회 + 측정 7회 |
| 순서 | trial별 seed 2026+trial로 섞고 순차 실행 |

합계 **32개 구성 / 96개 process trial / 672개 timing 표본 / 864개 dispatch**입니다.
입력 종류는 모든 구현에서 같으며 같은 실행 파일을 사용합니다.

| M,K,N | 목적 | 주 비교의 활성 코어 |
| --- | --- | --- |
| 128,128,128 | 세 방식 모두 코어별 출력 수가 균등 | 모두 4 |
| 96,64,160 | 홀수 M/N 타일 수와 tail·불균형 | 모두 4 |
| 65,257,97 | Kt=9와 행 경계, 출력 분배 차이 | 모두 4 |
| 32,64,128 | 2D가 1×4로 축소되어 A-only와 같은 공유 방향 | 모두 4 |

각 반복 전에 NaN sentinel을 출력에 쓰고 완료시킨 후 타이머를 시작합니다.
`EnqueueMeshWorkload` 직전부터 `Finish` 반환까지 C++ `steady_clock`으로 측정합니다.
입력 생성/packing/전송, 출력 초기화/readback, CPU 검증은 측정 구간 밖입니다.
runtime/slow-dispatch/host IPC/시뮬레이터 처리·완료 대기 비용은 포함합니다.
처음 두 dispatch는 기록하되 통계에서 제외합니다. JIT cache를 삭제하지 않았으므로 cold JIT 측정은 아닙니다.

모든 반복의 출력이 같아야 하고, 마지막 출력은 CPU FP64 reference·finite·padding·tilizer 검사도
통과해야 합니다. 모드/trial 사이 input/output fingerprints와 타일 분배, 읽기/전달/CB 모델도 대조합니다.

## 통계와 실측 결과

대표 시간은 process마다 구한 7개 timing의 중앙값을 다시 세 process 사이에서 중앙값으로 요약합니다.
각 process 중앙값과 raw samples도 보존했습니다.
p95는 21개 표본을 모아 `(n-1)*0.95`에서 선형 보간한 기술 통계입니다.
한 process의 표본은 관련될 수 있고 독립 trial도 3개뿐이므로 통계적 유의성·장기 tail 보장은 주장하지 않습니다.

아래는 **실제 4코어 / CB depth2**, 단위 ms입니다.
`독립4 / 현재`는 이번 환경에서 같은 depth의 독립 4코어 대표 시간을 현재 대표 시간으로 나눈 값입니다.
1보다 크면 이번 측정에서 더 짧았고, 1 미만이면 더 길었습니다.

| M,K,N | 방식 | 대표 ms | Trial 중앙값 ms | pooled p95 ms | 독립4 / 현재 |
| --- | --- | --- | --- | --- | --- |
| [128, 128, 128] | 독립4 | 61.946 | 61.95/68.93/43.56 | 95.885 | 1.00 |
| [128, 128, 128] | A-only4 | 58.090 | 54.87/62.43/58.09 | 89.153 | 1.07 |
| [128, 128, 128] | 2D4 | 90.676 | 107.54/86.35/90.68 | 113.711 | 0.68 |
| [96, 64, 160] | 독립4 | 24.242 | 24.24/25.90/23.08 | 28.287 | 1.00 |
| [96, 64, 160] | A-only4 | 32.315 | 32.31/32.93/31.02 | 35.282 | 0.75 |
| [96, 64, 160] | 2D4 | 48.350 | 48.35/51.07/45.60 | 75.826 | 0.50 |
| [65, 257, 97] | 독립4 | 62.647 | 96.32/62.65/55.49 | 99.982 | 1.00 |
| [65, 257, 97] | A-only4 | 81.722 | 79.82/82.63/81.72 | 90.327 | 0.77 |
| [65, 257, 97] | 2D4 | 137.839 | 124.75/145.36/137.84 | 160.657 | 0.45 |
| [32, 64, 128] | 독립4 | 10.589 | 9.52/48.90/10.59 | 61.352 | 1.00 |
| [32, 64, 128] | A-only4 | 14.813 | 14.81/10.64/18.77 | 19.362 | 0.71 |
| [32, 64, 128] | 2D4 | 18.462 | 21.54/18.46/11.98 | 26.226 | 0.57 |

### 이번 측정의 관찰

이번 TTSim/host 환경에서는 2D가 세 방식 중 가장 짧지 않았습니다.
128×128, depth2에서 독립 61.946ms,
A-only 58.090ms, 2D 90.676ms였습니다.
2D는 독립 방식보다 대표 시간이 약 1.46배 길었지만 입력 DRAM read 모델은 절반입니다.
전역 읽기량만 줄이는 것이 이 환경의 시간 개선을 보장하지 않는다는 관찰입니다.

A-only의 128×128/depth2 대표 시간은 독립보다 조금 짧지만,
trial별 독립/A-only 시간 비율은 **1.13/1.10/0.75**로 방향이 바뀌었습니다.
따라서 약간의 대표 시간 차이를 안정적인 개선으로 결론 내리지 않습니다.

65×257×97의 2D 대표 시간은 독립보다 약 2.20배 길었습니다.
이 입력에서는 2D의 최대 코어별 matmul 호출 수가 36회, 독립/A-only는 27회입니다.
DRAM 절약과 별개로 출력 분배의 불균형도 비교에 포함됩니다.

32×64×128은 A-only와 2D 모두 실제 1×4로 A를 공유하지만 관측 시간이 같지 않았습니다.
커널의 제어 흐름·semaphore 설정 등 구현 차이와 host 변동도 포함되기 때문이며,
이 실험으로 그중 하나를 단독 원인으로 지목하지 않습니다.
독립 depth2의 trial 중앙값에도 약 9.52/48.90/10.59ms의 큰 변동이 있어 p95와 개별 trial을 함께 봐야 합니다.

일부 결과가 예상보다 느리다는 이유로 구성이나 trial을 제외하지 않았습니다.
독립1코어 대조군과 CB depth1/2의 전체 32개 결과는 아래 원시 비교 표에 보존합니다.


## 시간과 함께 볼 DRAM·NoC·부하 모델

아래 값은 실행 counter가 아닌 source/shape 기반 모델입니다.
receiver data는 원격 L1에 전달된 payload의 합이며 NoC packet 수나 link별 실제 이동 bytes는 아닙니다.
추가 atomic·flag와 protocol overhead도 data payload에 포함하지 않습니다.

| M,K,N | 방식 | DRAM read KiB | Receiver data KiB | Payload mcast calls | Ready atomic increments | 최대 matmul 호출/코어 |
| --- | --- | --- | --- | --- | --- | --- |
| [128, 128, 128] | 독립4 | 256.0 | 0.0 | 0 | 0 | 16 |
| [128, 128, 128] | A-only4 | 160.0 | 96.0 | 16 | 48 | 16 |
| [128, 128, 128] | 2D4 | 128.0 | 128.0 | 64 | 64 | 16 |
| [96, 64, 160] | 독립4 | 120.0 | 0.0 | 0 | 0 | 8 |
| [96, 64, 160] | A-only4 | 84.0 | 72.0 | 12 | 36 | 12 |
| [96, 64, 160] | 2D4 | 76.0 | 76.0 | 38 | 38 | 12 |
| [65, 257, 97] | 독립4 | 432.0 | 0.0 | 0 | 0 | 27 |
| [65, 257, 97] | A-only4 | 270.0 | 162.0 | 27 | 81 | 27 |
| [65, 257, 97] | 2D4 | 252.0 | 252.0 | 126 | 126 | 36 |
| [32, 64, 128] | 독립4 | 32.0 | 0.0 | 0 | 0 | 2 |
| [32, 64, 128] | A-only4 | 20.0 | 12.0 | 2 | 6 | 2 |
| [32, 64, 128] | 2D4 | 20.0 | 12.0 | 2 | 6 | 2 |

### 균등한 128×128에서도 전역 읽기량만으로 시간을 예측할 수 없는 이유

세 버전의 코어별 유효 matmul 호출 수는 모두 16회입니다.
하지만 입력 전달 경로와 sender의 책임은 다릅니다.

| source 기반 지표 | 독립4 | A-only4 | 2D4 |
| --- | --- | --- | --- |
| 전역 A/B DRAM tile reads 합계 | 128 | 80 | 64 |
| 가장 많이 읽는 코어의 DRAM tile reads | 32 | 32 | 32 |
| 전역 payload multicast 호출 | 0 | 16 | 64 |
| 가장 많이 전파하는 코어의 payload multicast 호출 | 0 | 16 | 32 |
| ready atomic increment 합계 | 0 | 48 | 64 |
| receiver data payload | 0 | 96KiB | 128KiB |

A-only sender는 A 16회 + 자기 B 16회를 읽습니다.
2D의 왼쪽 위 코어는 A 16회 + B 16회를 읽고 두 방향으로 전파합니다.
따라서 전역 DRAM 읽기량이 줄어도 가장 바쁜 sender의 read 수가 같은 경우가 있고
그 sender의 multicast/완료 대기 책임은 오히려 늘어납니다.
이는 구조적 tradeoff이며 특정 runtime stall이나 병목이 실제 발생했다는 증거는 아닙니다.

비정방형 입력에서는 코어별 출력 수 차이도 있습니다.
예를 들어 Mt=3,Nt=5의 독립4 분배는 4/4/4/3타일이지만, A-only는 6/3/3/3,
2D는 6/4/3/2타일입니다. 총 계산량이 같아도 최대로 담당하는 코어의 작업량은 다릅니다.

## 재현성과 해석 제한

trial 안에서 순서를 섞었지만 CPU affinity/frequency·Windows background load는 고정하지 않았습니다.
host의 CPU/OS/compiler, 실행 전후 meminfo/loadavg, process별 loadavg와 전체 process 시간도 남겼습니다.
프로세스별 차이가 크거나 trial마다 비율 방향이 바뀌면 안정적인 우열을 확정하지 않습니다.

실물 NoC bandwidth·DRAM stall·연산/통신 overlap은 이번 host 측정으로 분리할 수 없습니다.
같은 active core 수·CB depth라는 조건 외에 출력 배치와 reader NoC 차이가 포함돼 있으므로
순수 multicast 비용을 분리하려면 동일 매핑/NoC의 직접 읽기 대조 구현이 추가로 필요합니다.
이번 데이터와 과거 10월 7일/앞선 multicore 측정을 섞어 속도 비율을 만들지 않습니다.

## 실행과 증거

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/benchmark_multicast_matmul.sh
# 간단 재실행
bash scripts/benchmark_multicast_matmul.sh --trials 1 --samples 7
```

`--samples`는 5..25, `--trials`는 1..5입니다.
[benchmark_multicast_matmul.py](../tools/report/benchmark_multicast_matmul.py)가 고정 버전·환경을 확인하고
실행 순서, 수치 검증, 원시 표본, 통계와 모델 비교를 저장합니다.
기존 C++/장치 커널은 이번 측정을 위해 변경하지 않았습니다.

- [메타데이터·순서·버전·소스 SHA-256](../results/matmul_multicast_perf/20261019T112137744676Z/metadata.json)
- [전체 32개 구성 비교 JSON](../results/matmul_multicast_perf/20261019T112137744676Z/comparison.json)
- [전체 비교 표](../results/matmul_multicast_perf/20261019T112137744676Z/comparison.md)
- [128×128 독립4/depth2 trial1](../results/matmul_multicast_perf/20261019T112137744676Z/128_128_128_multicore_cores4_depth2_trial1.json)
- [128×128 A-only4/depth2 trial1](../results/matmul_multicast_perf/20261019T112137744676Z/128_128_128_mcast_cores4_depth2_trial1.json)
- [128×128 2D4/depth2 trial1](../results/matmul_multicast_perf/20261019T112137744676Z/128_128_128_mcast2d_cores4_depth2_trial1.json)
- [2D trial1 전체 로그](../results/matmul_multicast_perf/20261019T112137744676Z/128_128_128_mcast2d_cores4_depth2_trial1.log)
- [host 빌드 로그](../results/matmul_multicast_perf/20261019T112137744676Z/build.log)

**96개 process가 모두 exit code 0**이며 모든 반복의 동일 출력과 최종 CPU 검증이 통과했습니다.
입출력 fingerprints·실제 4코어·타일 소유권·전송량/CB 모델 대조도 통과했습니다.
측정된 source/executable hash를 기록했고 느린 결과도 포함해 보존했습니다.
