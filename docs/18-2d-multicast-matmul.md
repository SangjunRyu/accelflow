# 18. 2차원 Multicast MatMul — A 행 공유·B 열 공유

상태: **A/B 행·열 multicast 구현·실행 검증 완료**. 작업 날짜: 2026-10-19.
[진행 순서](README.md) · 이전: [17. A-only multicast](17-a-multicast-matmul.md)

## 구현 범위

`mcast2d`는 A를 코어 행 방향으로, B를 코어 열 방향으로 실제 NoC multicast 전송하는 MatMul입니다.
각 출력 타일의 K 전체는 한 코어에서 FP32 destination에 누적하고 BF16으로 pack합니다.
K를 코어 사이에 나누거나 partial sum을 교환하는 방식은 아닙니다.

4코어의 기본 2×2 역할은 다음과 같습니다. 표기는 `(col,row)` 논리 좌표입니다.

```text
                       B ↓                         B ↓
                 (0,0): A/B sender ── A → (1,0): A receiver / B sender
                       │                           │
                       B                           B
                       ↓                           ↓
                 (0,1): A sender / B receiver ─ A → (1,1): A/B receiver
```

각 행의 col=0 코어가 해당 A를 읽고 같은 행의 receiver에 보냅니다.
각 열의 row=0 코어가 해당 B를 읽고 같은 열의 receiver에 보냅니다.
sender 자신도 자기 L1 타일로 계산합니다. 코어마다 서로 다른 출력 타일을 씁니다.

## 코드 읽는 순서

1. [mcast2d.hpp](../npu/matmul/mcast2d.hpp): core grid, 물리 좌표, CB/semaphore와 sender 구성
2. [reader_mcast2d.cpp](../npu/matmul/kernels/reader_mcast2d.cpp): `Channel`, `share_tile`, A→B 전송 순서
3. [compute_mcast2d.cpp](../npu/matmul/kernels/compute_mcast2d.cpp): 유효 출력 누적 및 경계 CB 소비
4. [writer_mcast2d.cpp](../npu/matmul/kernels/writer_mcast2d.cpp): row/column stride에 따른 출력 저장
5. [matmul.cpp](../npu/matmul/matmul.cpp): 기존 입력 생성·CPU 검증과 `mcast2d` 연결

기존 A-only 및 독립 multicore 커널을 바꾸지 않고 별도 구현으로 추가했습니다.

## 출력 분배와 축소 배치

`Mt=ceil(M/32)`, `Kt=ceil(K/32)`, `Nt=ceil(N/32)`입니다.
core grid를 `Pr`행 × `Pc`열로 잡고 코어 `(c,r)`는
`m=r,r+Pr,...`, `n=c,c+Pc,...`에 해당하는 C 타일을 계산합니다.

| 요청 코어 수 | 배치 정책 |
| --- | --- |
| 1 | 1×1 |
| 2 | Nt>1이면 1×2, 아니면 최대 2×1 |
| 4 | Mt,Nt>1이면 2×2 |
| 4, Mt=1 | 최대 1×4로 A를 가로 공유 |
| 4, Nt=1 | 최대 4×1로 B를 세로 공유 |

행/열 크기에 따라 실제 활성 코어가 요청보다 작을 수 있습니다.
물리 worker 좌표를 조회하고, 선택한 사각형이 연속된 worker만 포함하는지 확인합니다.
NoC0를 사용하며 각 multicast의 목적지 수는 source를 제외한 Pc-1 또는 Pr-1입니다.
다른 물리 배치/harvested 구성에 대해서는 이 조건이 맞지 않으면 안전하게 오류로 종료합니다.

## 두 방향의 동기화

공통 CoreRange에 CB0=A, CB1=B, CB16=C와 4개 semaphore를 생성합니다.

| 채널 | sender | ready counter | receiver valid |
| --- | --- | --- | --- |
| A 행 | col=0 | A-ready | A-valid |
| B 열 | row=0 | B-ready | B-valid |

유효한 입력에 대한 채널 흐름은 `reserve → receiver ready atomic → sender data multicast →
write barrier → valid multicast → write barrier → push`입니다.
모든 코어가 각 K 타일에서 **A→B 순서**로 진행합니다. A/B semaphore 쌍은 분리돼 있으므로
행 방향의 준비 통보가 열 방향 counter와 섞이지 않습니다.
reader는 RISCV_0/NoC0, writer는 RISCV_1/NoC1이며 출력 계산은 Tensix에서 수행합니다.

### 홀수 행·열과 dummy 슬롯

전 코어가 `Gm=ceil(Mt/Pr)`, `Gn=ceil(Nt/Pc)`에 대해 같은 수의 group/K 단계를 진행합니다.
한 코어의 타일은 `(mg×Pr+r, ng×Pc+c)`입니다.

- m만 유효하면 A 전송에는 참여하고 B 열의 유효성에 따라 B를 받습니다.
- n만 유효하면 B 전송에는 참여하고 A 행의 유효성에 따라 A를 받습니다.
- A 행 자체가 범위 밖이면 해당 행은 A DMA/handshake를 생략하고 dummy CB 슬롯을 push합니다.
- B 열 자체가 범위 밖이면 해당 열은 B DMA/handshake를 생략하고 dummy CB 슬롯을 push합니다.
- compute는 유효하지 않은 출력에서도 **A/B 둘 다 wait/pop**하고 실제 matmul/출력만 생략합니다.

dummy 슬롯의 값은 읽지 않으므로 초기화하지 않습니다.
반면 유효한 행의 A가 유효하지 않은 출력 열에 도착하거나, 유효한 열의 B가 유효하지 않은
출력 행에 도착하는 tail 전송은 실제 데이터이며 소비 후 버립니다.
모든 CB의 push/pop 횟수를 맞춰 ring offset이 다르게 진행되는 것을 방지합니다.

## DRAM/NoC 비용 모델

`Gm=ceil(Mt/Pr)`, `Gn=ceil(Nt/Pc)`입니다.

| 지표 | 독립 multicore | 2D multicast |
| --- | --- | --- |
| A DRAM tile reads | Mt×Kt×Nt | Mt×Kt×Gn |
| B DRAM tile reads | Mt×Kt×Nt | Nt×Kt×Gm |
| C tile writes | Mt×Nt | 동일 |
| 유효 matmul_tiles 호출 | Mt×Kt×Nt | 동일 |
| A receiver 전달 tile 수 | 없음 | A reads×(Pc-1) |
| B receiver 전달 tile 수 | 없음 | B reads×(Pr-1) |
| 코어별 CB payload | 2048×3×depth | 동일 |

실제 데이터가 전달된 뒤 버려지는 A tail은 `Mt×Kt×(Gn×Pc-Nt)`, B tail은
`Nt×Kt×(Gm×Pr-Mt)`입니다. 입력 자체가 없는 dummy 슬롯과 구분해 기록합니다.
각 코어의 CB 단계 수는 `Gm×Gn×Kt`로 같습니다.

이 수치는 소스 기반 모델이며 물리 DRAM transaction/NoC packet/cycle counter가 아닙니다.
입력 읽기를 코어 간 NoC 전달로 대체하므로 DRAM 읽기가 줄어도 전체 NoC 비용이 같은 비율로
줄어드는 것은 아닙니다. ready atomic·valid 전파·완료 대기 비용도 추가됩니다.

### 128×128 행렬곱, 4개 활성 코어

Mt=Kt=Nt=4입니다. A-only는 1×4, 2D는 2×2 배치입니다.

| 항목 | 독립 multicore | A-only | A/B 2D multicast |
| --- | --- | --- | --- |
| A DRAM reads | 64 tiles | 16 tiles | 32 tiles |
| B DRAM reads | 64 tiles | 64 tiles | 32 tiles |
| 입력 read payload | 256KiB | 160KiB | 128KiB |
| 출력 쓰기 | 32KiB | 32KiB | 32KiB |
| receiver payload 전달 | 없음 | 96KiB | 128KiB |
| depth2 CB payload 합계 | 48KiB | 48KiB | 48KiB |

2D 입력 read 모델은 독립 방식 대비 **50%**, A-only 대비 **20%** 감소합니다.
read+write DRAM payload 모델은 독립 방식 대비 약 44.4% 감소합니다.
대신 A-only보다 receiver payload와 handshake 수가 늘어납니다.
같은 CB 용량이어도 semaphore는 A-only의 2개에서 4개로 늘며, 이 저장 비용과 기타 runtime
메모리는 CB payload 합계에 포함하지 않습니다.

shape가 바뀌면 이 비율도 바뀝니다. 모든 shape에서 2D가 A-only보다 더 적게 읽거나
더 짧게 실행된다고 일반화하지 않습니다.

## 검증 결과

**12개 구성 × 20개 입력 = 240개 검증 모두 PASS**입니다.
별도 반복/fallback 14개 profile에서 **98회 dispatch**도 모두 성공했습니다.
동일 출력·CPU 비교·semaphore 재사용과 실제 1×4/4×1 배치를 확인했습니다.
기존 baseline/reuse_a/blocked/streamed의 **188개 회귀 검증도 통과**했습니다.

20개 공통 입력을 mcast2d 요청1/2/4 × depth1/2, baseline depth1/2,
독립 multicore 요청4 depth1/2, A-only 요청4 depth1/2의 12개 구성에서 비교했습니다.
입력/출력 fingerprints가 일치하며 CPU FP64 reference, 공식 tilizer, NaN sentinel 덮어쓰기,
padding=0, 정수 exact 및 기존 fractional 허용 오차 검사를 통과해야 성공으로 기록합니다.

추가 검증:

- 실제 2×2의 A/B sender 역할, 물리 행/열 목적지, 출력 타일 중복/누락 없음
- 한 축만 있는 배치와 단일 코어 fallback
- 홀수 M/N 타일 수, 두 축 모두 남는 모서리, dummy/tail 소비, CB wrap
- K=512의 2×2 누적 및 fractional 입력
- 같은 workload를 7회씩 재제출해 semaphore/CB 상태 재사용
- 1×4/4×1 입력을 baseline과 별도 비교하고 반복 실행
- DRAM read, multicast 전달, dummy/tail 및 CB 모델의 독립 계산 대조

반복 검사의 timing도 JSON에 남지만, 이번 작업은 동기화·정확성 검증입니다.
독립 방식/A-only/2D 사이의 안정적인 시간 비교나 실물 속도 개선은 아직 측정·입증하지 않았습니다.

## 재현과 증거

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_mcast2d_matmul.sh
```

실행 파일 인자는 `matmul report.json mcast2d depth [requested_cores]`입니다.
depth는 1/2, requested_cores는 1/2/4이고 기본 요청은 4입니다.
선택 실행/반복 검사에서는 기존 `--benchmark` 경로를 사용합니다.
1×4/4×1 전용 두 입력은 선택 실행에 추가했고 일반 suite의 20개 입력은 유지했습니다.

- [버전·명령·소스 SHA-256](../results/matmul_mcast2d/20261019T111449423414Z/metadata.json)
- [grid/분배/모델/coverage 평가](../results/matmul_mcast2d/20261019T111449423414Z/design-assessment.json)
- [2D 요청4/depth2 결과](../results/matmul_mcast2d/20261019T111449423414Z/mcast2d_cores4_depth2.json)
- [2D 요청4/depth2 전체 로그](../results/matmul_mcast2d/20261019T111449423414Z/mcast2d_cores4_depth2.log)
- [A-only 비교 기준](../results/matmul_mcast2d/20261019T111449423414Z/mcast_cores4_depth2.json)
- [독립 multicore 비교 기준](../results/matmul_mcast2d/20261019T111449423414Z/multicore_cores4_depth2.json)
- [모서리 tail 재실행](../results/matmul_mcast2d/20261019T111449423414Z/repeat_96_64_160_depth2.json)
- [4×1 B-only multicast](../results/matmul_mcast2d/20261019T111449423414Z/fallback_mcast2d_128_64_32_depth2.json)
- [1×4 A-only multicast](../results/matmul_mcast2d/20261019T111449423414Z/fallback_mcast2d_32_64_128_depth2.json)
- [기존 188개 회귀 검증](../results/matmul/20261019T111704143636Z/design-assessment.json)

공식 API 기준은 고정된 [v0.79.0 multicast MatMul](https://github.com/tenstorrent/tt-metal/tree/v0.79.0/tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast)과
[sender/receiver 공통 커널](https://github.com/tenstorrent/tt-metal/tree/v0.79.0/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow)입니다.
앞 단계에서 실제 checkout의 ready/data/valid 순서를 확인했으며, grid/stride/tail/dummy 처리는 자체 구현했습니다.
기존 TT-Metal/TTSim revision과 라이브러리를 유지했고 전체 upstream 재빌드는 하지 않았습니다.

## 다음 단계

같은 활성 코어 수·dtype·입력·CB 깊이에서 독립/A-only/2D의 TTSim 실행 비용을 비교하고
추가 handshake와 tail 전송 비용을 평가할 수 있습니다. 이후 출력 block 재사용, 명시적 전송/계산 overlap,
더 큰 grid·비연속 물리 배치 지원 등은 별도 구현·검증 단계입니다.

후속 측정 완료: [19. 독립/A-only/2D 시간 비교](19-multicast-simulator-performance.md)에서 같은 활성 4코어·CB 깊이의 실행 비용을 비교했습니다. 본 문서의 미측정 설명은 최초 구현 단계의 기록입니다.
