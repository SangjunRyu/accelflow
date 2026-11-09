# 17. A Multicast MatMul — sender/receiver 입력 공유

상태: **A multicast sender/receiver 구현·실행 검증 완료**. 작업 날짜: 2026-10-19.
[진행 순서](README.md) · 이전: [16. Multicore 실행 비용](16-multicore-simulator-performance.md)

## 구현한 것

`mcast`는 sender가 A 타일을 DRAM에서 한 번 읽고 **실제 NoC multicast API**로 receiver들의
L1 입력 CB에 전달하는 1차원 multicore MatMul입니다. unicast를 코어 수만큼 호출해 흉내 낸 방식이 아닙니다.
sender도 자기 A를 사용해 계산하고, B는 각 코어가 DRAM에서 직접 읽습니다.

이 버전은 **A만 공유**합니다. A와 B를 모두 공유하는 2차원 행/열 multicast나
코어 간 K 부분합 교환까지 구현한 것은 아닙니다.
출력 한 타일의 K 전체를 해당 코어가 FP32 destination에 누적하고 BF16으로 pack합니다.

```text
                    DRAM A
                       ↓ 한 번 읽기
core0 sender: A CB ── NoC multicast ──┬─ receiver1 A CB
     ↓                              ├─ receiver2 A CB
   compute                          └─ receiver3 A CB
     ↓                                   ↓
  writer                             compute → writer
     └────────── 공유 DRAM C의 서로 다른 타일 ──────────┘

DRAM B는 각 코어가 담당 출력 열에 맞춰 직접 읽음
```

## 파일과 역할

- [mcast.hpp](../npu/matmul/mcast.hpp): 코어/물리 좌표, 공통 CB, semaphore, 커널/runtime args 구성
- [reader_mcast.cpp](../npu/matmul/kernels/reader_mcast.cpp): A sender/receiver handshake, A 전파, local B 읽기
- [compute_mcast.cpp](../npu/matmul/kernels/compute_mcast.cpp): K 누적 및 tail A 소비
- [writer_mcast.cpp](../npu/matmul/kernels/writer_mcast.cpp): 각 코어의 출력 열을 DRAM C에 기록
- [matmul.cpp](../npu/matmul/matmul.cpp): `mcast` 선택, 공통 입력/출력 및 CPU 검증

## 출력 분배와 물리 multicast 범위

`Mt=ceil(M/32)`, `Kt=ceil(K/32)`, `Nt=ceil(N/32)`이며
활성 코어 수는 `P=min(requested,Nt)`입니다. 요청은 1/2/4입니다.
한 논리 코어 행 `(rank,0)`에서 코어 rank가 출력 열 `rank, rank+P, rank+2P, ...`를 담당합니다.
모든 코어는 m행과 열 group을 같은 순서로 진행합니다.

기존 독립 multicore는 `min(requested,Mt×Nt)`개 코어에 연속 출력 구간을 분배했습니다.
여기서는 A 공유를 위해 **열 방향으로만** 분배하므로 실제 활성 코어 수가 다를 수 있습니다.
예를 들어 Nt=3이고 4코어를 요청하면 이 버전은 3개 코어를 사용합니다.
Nt=1이면 m행이 많아도 한 코어만 사용합니다. A 공유와 충분한 코어 활용이 항상 함께 되는 것은 아닙니다.

논리 좌표를 실제 worker 물리 좌표로 변환하고 선택 행이 물리적으로 연속인지 검사합니다.
multicast 사각형은 sender를 제외한 receiver들만 포함하며, destination count는 `P-1`입니다.
물리 행이 연속되지 않으면 사각형 안의 다른 자원을 건드리지 않도록 오류로 종료합니다.
현재 Wormhole에서 최대 4개 코어의 이 선택을 검증했습니다.

## Semaphore와 CB 동기화

전 코어에 같은 CoreRange로 CB0=A, CB1=B, CB16=C와 두 semaphore를 함께 생성합니다.
동일 CB ID/설정과 동일 진행 순서로 multicast 대상 L1 offset이 일치하도록 구성합니다.

| 단계 | sender | receiver |
| --- | --- | --- |
| 슬롯 확보 | A CB reserve | A CB reserve |
| 준비 통보 | A를 DRAM에서 읽음 | local valid=0, sender ready counter에 atomic +1 |
| 준비 확인 | ready=P-1 대기 후 ready=0 | valid=1 대기 |
| 데이터 전달 | `noc_async_write_multicast`, write barrier | multicast가 A 슬롯에 도착 |
| 공개 통보 | `noc_semaphore_set_multicast`, write barrier | valid=1을 관측 |
| 소비자 연결 | A CB push | A CB push |

이후 각 코어가 유효 출력 열의 B를 읽고 B CB에 push합니다.
reader는 RISCV_0/NoC0, writer는 RISCV_1/NoC1을 사용합니다.
학습용 기준 버전은 data와 valid 사이의 완료 순서를 명확히 하기 위해 write barrier를 사용합니다.
공식 예제의 VC 순서 보장에 의존한 barrier 생략 최적화는 적용하지 않았습니다.

### 마지막 열 group의 receiver도 A를 소비해야 하는 이유

Nt가 P의 배수가 아니면 마지막 group에 출력이 없는 rank가 생깁니다.
그 코어도 A reserve/ready/수신/push와 compute의 A wait/pop에는 참여합니다.
B 읽기, matmul, 출력 pack/write만 생략합니다.

만약 이 rank만 A push/pop을 생략하면 다른 코어와 A CB의 ring 위치가 달라져
다음 group/행의 multicast가 잘못된 슬롯에 쓸 수 있습니다.
출력 없는 A 소비는 reader가 임의로 pop하지 않고 **compute 소비자에서** 수행합니다.
그래야 depth2에서 아직 소비되지 않은 앞 타일을 reader가 잘못 제거하지 않습니다.

## 소스 기반 전송량 모델

`G=ceil(Nt/P)`, `R=Mt×Kt×G`라고 하면:

| 항목 | 독립 multicore baseline | A multicast |
| --- | --- | --- |
| A DRAM tile reads | Mt×Kt×Nt | R |
| B DRAM tile reads | Mt×Kt×Nt | 동일 |
| 입력 read payload bytes | 4096×Mt×Kt×Nt | 2048×(R+Mt×Kt×Nt) |
| 유효 matmul_tiles 호출 | Mt×Kt×Nt | 동일 |
| C tile writes | Mt×Nt | 동일 |
| A payload multicast 호출 | 0 | R (P>1), 0 (P=1) |
| receiver에 전달된 A tile 수 | 0 | R×(P-1) |
| 출력 없는 tail에서 소비한 A tile 수 | 0 | Mt×Kt×(G×P-Nt) |

receiver가 DRAM에서 읽는 A를 sender→receiver NoC 전송으로 대체합니다.
따라서 **DRAM read 감소가 전체 NoC 전송량 감소를 의미하지는 않습니다.**
ready atomic, valid 전파와 완료 대기도 추가됩니다.
위 수치는 source/shape 모델이며 물리 transaction/cycle counter가 아닙니다.

### 128×128 행렬곱, 요청/활성 4코어

Mt=Kt=Nt=4, P=4, G=1입니다.

| 항목 | 독립 multicore | A multicast |
| --- | --- | --- |
| A DRAM reads | 64 tiles | 16 tiles |
| B DRAM reads | 64 tiles | 64 tiles |
| 전체 입력 DRAM read payload | 256KiB | 160KiB |
| 출력 쓰기 | 32KiB | 32KiB |
| receiver A 전달 | 없음 | 48 tiles / 96KiB |
| CB depth2 합계 | 48KiB | 48KiB |

입력 read 모델은 **37.5% 감소**합니다. 출력 쓰기까지 합친 DRAM payload 모델은 약 33.3% 감소합니다.
CB payload는 코어당 12KiB이며 semaphore·기타 runtime 메모리는 이 용량에 포함하지 않습니다.
이 감소율을 실측 시간 개선율이나 실제 칩 speedup으로 표현하지 않습니다.

## 검증 결과

**10개 구성 × 20개 입력 = 200개 모두 PASS**, 모든 실행 exit code 0입니다.
추가로 3개 shape × CB depth1/2에서 같은 workload를 각 7회 제출해
**42회 반복 dispatch**의 동일 출력·CPU 검증을 통과했습니다.
기존 단일 코어/blocked/K-streamed 188개 suite도 다시 통과했습니다.

baseline depth1/2, 독립 multicore 요청4 depth1/2, mcast 요청1/2/4 depth1/2를
같은 기존 20개 입력에서 비교했습니다. 입력 종류·CPU FP64 reference·정수 exact 및 fractional
허용 오차 `0.03+0.03×abs(reference)`는 변경하지 않았습니다.

검증 항목:

- 공식 tilizer 대조와 CPU 원소별 비교, padding=0, NaN 출력 sentinel 덮어쓰기
- 모드·코어 수·CB 깊이 사이 동일 입력/출력 fingerprints
- strided C 열 분배의 중복/누락 없음과 정확한 물리 receiver 범위
- Nt=1의 단일 코어 fallback 및 실제 활성 3/4코어
- Nt=5, 요청4에서 tail receiver, 여러 m행, CB wrap
- K=512, fractional, 비정방형/타일 경계
- 같은 workload 반복 제출 시 semaphore/CB 상태 재사용
- DRAM reads, multicast 전달 및 tail 소비 모델을 독립 계산해 보고서와 대조

반복 실행은 정확성·동기화 재사용 검증 목적입니다. raw timing도 남지만 이것만으로
독립 multicore와 multicast의 안정적인 시간 우열을 주장하지 않습니다.
별도 성능 비교 시 실제 활성 코어 수와 출력 매핑 차이, 추가 handshake 비용을 함께 맞춰야 합니다. reader/writer의 RISC-V·NoC 배치도 기존 독립 버전과 다르므로 성능 비교에서 통제하거나 명시해야 합니다.

## 재현과 증거

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_mcast_matmul.sh
```

실행 파일 인자는 `matmul report.json mcast depth [requested_cores]`입니다.
depth는 1/2, requested_cores는 1/2/4, 기본 요청은 4입니다.
기존 benchmark 인자도 지원하지만 이번 wrapper는 반복 동기화 검사에만 사용합니다.

- [버전·명령 구성·소스 SHA-256](../results/matmul_mcast/20261019T110520010148Z/metadata.json)
- [기능·분배·전송량 모델 평가](../results/matmul_mcast/20261019T110520010148Z/design-assessment.json)
- [mcast 요청4/depth2](../results/matmul_mcast/20261019T110520010148Z/mcast_cores4_depth2.json)
- [mcast 요청4/depth2 전체 로그](../results/matmul_mcast/20261019T110520010148Z/mcast_cores4_depth2.log)
- [독립 multicore 요청4 비교 기준](../results/matmul_mcast/20261019T110520010148Z/multicore_cores4_depth2.json)
- [tail 및 CB 재실행 검사](../results/matmul_mcast/20261019T110520010148Z/repeat_96_64_160_depth2.json)
- [기존 188개 회귀 검사](../results/matmul/20261019T110530769490Z/design-assessment.json)

공식 API 기준은 고정된 [v0.79.0 multicast MatMul](https://github.com/tenstorrent/tt-metal/tree/v0.79.0/tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast)과
[공통 sender/receiver 커널](https://github.com/tenstorrent/tt-metal/tree/v0.79.0/tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow)입니다.
실제 checkout에서 API·준비/완료 순서를 확인했습니다. 입력/출력 매핑, tail 처리와 검증은 자체 구현입니다.

기존 TT-Metal/TTSim revision과 prebuilt 라이브러리를 유지했습니다.
빌드 전 C: 여유 약 16.0GiB, WSL 가용 RAM 약 3.3GiB를 확인했습니다.
TT-Metal 전체 재빌드나 물리 장치 component 설치는 하지 않았습니다.

## 남은 단계

같은 활성 코어·유효 입력·정밀도에서 독립 multicore와 A multicast의 TTSim 실행 비용을 반복 비교할 수 있습니다.
다음 구현 확장은 B도 공유하는 2차원 multicast, group별 sender 배치와 tail 낭비 감소입니다.
실물 TT 칩의 DRAM/NoC bandwidth·stall·overlap은 장치 profiler로 별도 검증해야 합니다.

후속 구현 완료: [18. 2D A/B multicast](18-2d-multicast-matmul.md)에서 A 행 공유와 B 열 공유를 별도 채널로 구현했습니다. 본 문서의 A-only 구현과 검증은 비교 기준으로 유지합니다.

후속 비교: [19. Multicast 시간 측정](19-multicast-simulator-performance.md)에 A-only와 독립/2D의 반복 시간과 변동을 기록했습니다.
