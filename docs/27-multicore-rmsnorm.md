# 27. 행 분할 Multicore RMSNorm

상태: **1·2·4코어 fused RMSNorm 구현·행 분할·PyTorch 대조 완료**. 날짜: 2026-11-10.
[진행 순서](README.md) · 이전: [26. NPU Fusion](26-npu-rmsnorm-fusion.md)

## 코어마다 다른 행을 계산한다

26번의 fused 연산을 재사용하고 코어마다 연속한 32행 묶음을 배정했다.
한 행의 전체 폭은 같은 코어에서 계산하므로 코어 간 부분합이나 데이터 동기화 신호를 주고받지 않는다.
한 프로그램에 여러 코어의 reader/compute/writer를 등록해 한 번 제출한다.

input·gamma·output은 공통 DRAM buffer이며 각 코어의 CB와 입력/제곱 캐시는 private L1에 있다.
gamma는 읽기 전용으로 공유한다. output의 서로 다른 타일 구간에 기록한다.

## 분할과 주소 계산

Ht는 32행 묶음 수, Wt는 32열 타일 수다.
활성 코어 수는 요청한 1·2·4와 Ht 중 작은 값이다. 빈 코어에는 CB·커널을 만들지 않는다.
각 코어에 `Ht // active`개를 배정하고 앞의 `Ht % active`개 코어에 한 묶음씩 더 준다.
따라서 행 묶음 수 차이는 최대 하나다. 이는 padded 타일 작업량 기준의 균형이며
실제 코어별 cycle 균형을 측정한 결과는 아니다.

| runtime 정보 | 처리 |
| --- | --- |
| row_group_start | 해당 코어가 시작할 전역 행 묶음 |
| row_group_count | 해당 코어의 처리 묶음 수 |
| reader 입력 타일 | `(row_group_start + local_row) * Wt + col` |
| compute | 기존 fused 커널에 local row_group_count만 전달 |
| writer 시작 타일 | `row_group_start * Wt` |
| writer 타일 수 | `row_group_count * Wt` |

host와 독립 검증 script에서 구간 전체가 정확히 한 번 덮이는지 확인했다.
논리 코어 좌표는 device의 worker grid에서 고른다. 물리 NoC 좌표를 하드코딩하지 않았다.
같은 코어 안의 producer/consumer CB 동기화는 26번과 같고 peer 통신은 추가하지 않았다.

## 검증 입력과 결과

| 입력 | 목적 |
| --- | --- |
| 256×33 | 짧은 폭과 균등한 8개 행 묶음 |
| 256×257 | 균등 분할·여러 열 타일·각 코어의 반복 row-cache 재사용 |
| 128×1025 | 넓은 폭과 4개 행 묶음·코어별 L1 비용 |
| 65×65 | 3개 행 묶음·마지막 한 행·혼합 크기 값 |
| 1×33 zero | 요청 코어 수보다 작업이 적은 경우·NaN sentinel 쓰기 확인 |

입력과 gamma는 BF16으로 양자화한 뒤 FP32로 확장했다. 누적과 출력은 FP32다.
gamma에는 음수·0·비균일 값을 사용하고 epsilon은 저장된 FP32 1e-5를 명시적으로 전달했다.
검증 허용오차는 기존 `atol=1e-4`, `rtol=1e-4`를 유지했다.

- 5개 입력 × 1·2·4코어 = **15개 구성**을 CB depth 2에서 실행해 모두 통과했다.
- 꼬리 입력 두 개를 4코어로 각각 2회 다시 제출했고 pilot 4회까지 총 **23회 operator 실행**을 통과했다.
- CPP의 FP64 기준값으로 모든 유효 출력과 padding을 검사했다. 출력은 매번 NaN sentinel로 초기화했다.
- 저장된 1·2·4코어 출력의 FP32 바이트가 일치했고 입력·gamma·출력 fingerprint도 일치했다.
- PyTorch 공식 RMSNorm의 FP32·FP64 기준값으로 저장된 출력 15개를 독립 대조했다.

CPU FP64 기준 최대 절대 오차는 1.72526204434e-07였다.
PyTorch FP32 대비 최대 절대 오차는 2.38418579102e-07,
PyTorch FP64 대비는 1.72526204434e-07였다.

H=65에 4코어를 요청하면 활성 코어는 3개이고 H=1이면 1개다.
아래의 배열은 각 활성 코어가 처리하는 행 묶음 수다. CB 단위는 KiB다.

| H×W | 요청 | 활성 | 행 묶음 분배 | 코어당 CB | 활성 코어 CB 합계 |
| --- | --- | --- | --- | --- | --- |
| 256×257 | 1 | 1 | [8] | 124 | 124 |
| 256×257 | 2 | 2 | [4, 4] | 124 | 248 |
| 256×257 | 4 | 4 | [2, 2, 2, 2] | 124 | 496 |
| 128×1025 | 1 | 1 | [4] | 316 | 316 |
| 128×1025 | 2 | 2 | [2, 2] | 316 | 632 |
| 128×1025 | 4 | 4 | [1, 1, 1, 1] | 316 | 1264 |
| 65×65 | 1 | 1 | [3] | 76 | 76 |
| 65×65 | 2 | 2 | [2, 1] | 76 | 152 |
| 65×65 | 4 | 3 | [1, 1, 1] | 76 | 228 |
| 1×33 | 1 | 1 | [1] | 68 | 68 |
| 1×33 | 2 | 1 | [1] | 68 | 68 |
| 1×33 | 4 | 1 | [1] | 68 | 68 |

## 메모리와 전송 모델

코어당 CB 용량은 26번 fused와 같은 `4096 * [2*Wt + 6*depth + 1]` bytes다.
처리할 행 묶음이 여러 개여도 하나씩 재사용하므로 코어당 용량은 H에 따라 늘지 않는다.
활성 코어를 늘리면 private CB가 복제돼 합계는 코어당 값의 active배가 된다.
위 수치는 CB payload이며 code·stack·전체 device L1 사용량은 포함하지 않는다.

128×1025의 코어당 CB는 316 KiB, 4코어 합계는 1264 KiB다.
이 합계는 여러 코어에 분산된 용량이며 각 코어의 CB는 해당 코어의 L1에 놓인다.
중간 DRAM capacity는 모든 구성에서 0이다.

전체 타일 수를 T로 두면 DRAM 요청 payload 모델은 모든 코어 수에서 `3*T*4096` bytes로 같다.
input과 output 구간을 분할하고 gamma를 행 묶음마다 읽는 기존 경로를 유지하기 때문이다.
multicore 자체가 총 DRAM 전송량을 더 줄이지는 않는다. gamma caching이나 multicast는 구현하지 않았다.
이 수치는 source에서 계산한 모델이며 hardware counter 측정값은 아니다.

## TTSim 실행 기록

아래는 각 코어 설정에서 같은 5개 입력을 한 process로 실행한 관측값이다.
metadata의 Python `perf_counter` 기반 process_wall_seconds를 옮겼다.
초기화·JIT·host 검증을 포함하며 설정마다 process 한 번씩이다.

| 요청 코어 수 | 전체 process 시간 seconds |
| --- | --- |
| 1 | 318.488 |
| 2 | 138.802 |
| 4 | 82.965 |

이 실행에서 TTSim process 시간은 줄었지만 반복 통계나 실제 NPU의 가속 배율로 보고하지 않는다.
TTSim의 가상 코어 실행과 CPU scheduling, JIT 상태가 포함된 결과다.
확인한 설계 효과는 행 묶음 분산, 같은 출력, 코어별 L1 비용과 전체 DRAM 모델 유지다.

## 실행과 코드

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_rmsnorm_multicore.sh

.venv/bin/python tools/report/audit_rmsnorm_multicore.py \
  --results results/rmsnorm_multicore/20261110T060505067399Z
```

직접 host를 실행할 때 CLI는 `rmsnorm list.txt report.json depth repeats [unfused|fused] [cores]`다.
마지막 인자를 생략하면 기존처럼 1코어이며 multicore는 fused 경로에서만 지원한다.
이번 multicore 검증 폭은 33·65·257·1025다. 폭 4097의 단일 코어 검증은 26번에 있으며
4코어에서 모든 폭과 입력 조합을 검증했다고 주장하지 않는다.

- [host·행 분할·코어별 인자](../npu/rmsnorm/rmsnorm.cpp)
- [reader·전역 행 offset](../npu/rmsnorm/kernels/fused_reader.cpp)
- [writer·출력 구간](../npu/rmsnorm/kernels/fused_writer.cpp)
- [재사용한 compute](../npu/rmsnorm/kernels/fused_compute.cpp)
- [실행·독립 분할 검증](../tools/report/run_rmsnorm_multicore.py)
- [PyTorch 대조](../tools/report/audit_rmsnorm_multicore.py)
- [검증·작업량·메모리 모델](../results/rmsnorm_multicore/20261110T060505067399Z/summary.json)
- [명령·source/실행 파일 SHA-256·시간](../results/rmsnorm_multicore/20261110T060505067399Z/metadata.json)
- [PyTorch 결과](../results/rmsnorm_multicore/20261110T060505067399Z/pytorch-audit.json)
- [4코어 전체 로그](../results/rmsnorm_multicore/20261110T060505067399Z/cores4.log)

다음 후보는 TT-NN RMSNorm 기준 구현 대조다. 한 행을 여러 코어로 나누고 부분합을 교환하는
열 분할·sender/receiver 방식은 이번 범위에 포함하지 않았다.
