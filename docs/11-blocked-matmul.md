# 11. Blocked MatMul — A/B 재사용과 메모리 비용

상태: **단일 코어 2×2 출력 타일 블록 구현·검증 완료**. 작업 날짜: 2026-10-07.
[진행 순서](README.md) · 이전: [10. MatMul 첫 구현](10-matmul.md)

## 이번에 진행한 단계

1. 고정 TT-Metal v0.79.0의 `matmul_tiles` 입력/출력 타일 인덱스 API를 확인했습니다.
2. C: 여유 약 19.9GiB와 WSL 가용 RAM 약 3.2GiB를 확인했습니다.
3. 기존 host와 CPU 기준값을 공유하는 `blocked` 모드를 추가했습니다.
4. baseline/reuse_a/blocked × CB depth 1/2를 동일한 14개 입력으로 검증했습니다.
5. **총 84개 케이스 PASS, 여섯 실행 모두 exit code 0**을 확인했습니다.

TT-Metal 전체를 다시 빌드하거나 upstream 소스를 변경하지 않았습니다.
기존 BF16/HiFi4 단일 코어 설정, TTSim v1.11.1, Clang 20.1.2를 사용했습니다.

## 구현 흐름

기존 baseline은 출력 타일 하나마다 A/B를 다시 읽고, reuse_a는 A의 한 타일 행 전체를
N 방향으로 재사용했습니다. blocked는 출력 타일을 최대 2행×2열로 묶습니다.

```text
출력 블록 C(m:m+2, n:n+2)
    reader: A의 두 타일 행 × Kt, B의 Kt × 두 타일 열을 DRAM에서 읽음
    compute: 최대 4개 destination tile에 K축 부분 곱을 누적
             같은 A는 출력 열에서, 같은 B는 출력 행에서 재사용
    writer: 블록 내 출력 순서를 전역 tile row-major DRAM 주소로 변환
```

출력 타일의 크기는 32×32입니다. 따라서 2×2 출력 블록은 최대 64×64 원소 영역입니다.
`matmul_tiles(a,b,r*Kt+k,k*2+c,r*2+c)`의 마지막 인자가 출력 register tile index입니다.
각 출력의 K축 누적 순서는 기존 버전과 같으며, math fidelity와 dtype도 같습니다.

### 코드 읽는 순서

- [host: matmul.cpp](../npu/matmul/matmul.cpp): variant 선택, CB 크기, 커널 등록, CPU 검증
- [reader_blocked.cpp](../npu/matmul/kernels/reader_blocked.cpp): 입력 블록 DMA와 공개
- [compute_blocked.cpp](../npu/matmul/kernels/compute_blocked.cpp): 입력 재사용과 네 출력 누적
- [writer_blocked.cpp](../npu/matmul/kernels/writer_blocked.cpp): 출력 타일 주소 매핑

## CB와 경계 처리

CB0=A, CB1=B, CB16=C입니다. 입력 블록마다 각 입력 CB의 `2×Kt` 슬롯을 reserve/push합니다.
마지막 블록에 실제 행/열이 하나만 남으면 유효한 슬롯만 DMA로 읽습니다.
compute도 유효한 행/열만 접근하므로 빈 슬롯의 초기값에 의존하지 않습니다.
고정 블록 슬롯 수로 reserve/pop하여 ring wrap 때 블록이 연속된 공간에 놓이도록 합니다.

compute는 입력을 모두 받은 뒤 destination registers를 acquire하고 K축을 누적합니다.
commit/wait 후 유효한 출력 타일을 하나씩 pack/push하며 writer가 소비할 수 있습니다.
출력 CB 깊이가 1이어도 동작합니다. 입력 블록은 모든 재사용이 끝난 후에만 pop합니다.

여기서 depth 1/2는 blocked 입력 CB에 보관할 **블록 수**와 출력 CB의 **타일 수**입니다.
기존 baseline의 입력 depth는 타일 수였습니다. 같은 depth 숫자가 같은 L1 용량을 뜻하지 않습니다.
입력 K 전체를 보관하는 첫 버전이므로 host에서 **Kt <= 4, 즉 padded K <= 128**을 검사합니다.
더 큰 K를 지원하려면 K-block streaming 및 partial-result 처리 설계가 필요합니다.

## 검증 결과

기존 12개 입력에 다음 두 입력을 추가했고 세 모드에 모두 같은 순서로 적용했습니다.

| M,K,N | 목적 |
| --- | --- |
| 96,64,160 | 3×5 출력 타일: 홀수 행/열, 6개 출력 블록, 입력 CB wrap |
| 128,128,128 | 4×4 출력 타일: Kt=4, 최대 네 출력 누적, 반복 블록/CB wrap |

기존 zero/identity/결정적 입력, 비정방형 행렬, 음수 정수, 소수 BF16 입력도 유지했습니다.
모든 모드에서 CPU FP64 기준값 비교, 공식 tilizer 대조, NaN 출력 sentinel 덮어쓰기,
padding 0 검사가 통과했습니다. 정수 fixture는 exact, 소수 fixture는 기존
`abs_error <= 0.03 + 0.03*abs(reference)` 기준을 유지했습니다.

여섯 설정의 입력/출력 FNV fingerprints가 일치했습니다. 이는 동일성 대조용이며 보안 hash가 아닙니다.
각 소수 fixture의 최대 절대 오차도 기존 결과와 동일합니다. 이 검증은 현재 fixture에 대한 결과이며
모든 입력 범위·dtype·NaN/Inf·overflow의 일반적인 GEMM 정확성을 보장하지 않습니다.

## 전송량 모델과 메모리 비용

`Mt=ceil(M/32)`, `Kt=ceil(K/32)`, `Nt=ceil(N/32)`이며
`Mb=ceil(Mt/2)`, `Nb=ceil(Nt/2)`입니다. 경계 슬롯에 대한 불필요한 DMA는 없습니다.

| 지표 | baseline | reuse_a | blocked |
| --- | --- | --- | --- |
| A tile reads | Mt×Kt×Nt | Mt×Kt | Mt×Kt×Nb |
| B tile reads | Mt×Kt×Nt | Mt×Kt×Nt | Nt×Kt×Mb |
| C tile writes | Mt×Nt | 동일 | 동일 |
| matmul_tiles 호출 | Mt×Kt×Nt | 동일 | 동일 |
| NoC read barrier 호출 | 2×Mt×Kt×Nt | Mt+Mt×Kt×Nt | Mb×Nb |
| 최대 destination tile slots | 1 | 1 | 4 |
| 구성 CB payload bytes | 2048×3d | 2048×(Kt+2d) | 2048×(4Kt+1)d |

위 수치는 **소스 기반 모델**이며 실행 중 DRAM counter나 물리 packet/cycle 측정이 아닙니다.
CB payload 용량은 전체 L1 사용량과 다르고 destination registers는 CB 용량에 포함되지 않습니다.
NoC read barrier 한 번이 여러 비동기 read의 완료를 기다릴 수 있습니다.

### 128×128 × 128×128

| 지표 | baseline | reuse_a | blocked |
| --- | --- | --- | --- |
| A/B tile reads | 64/64 | 16/64 | 32/32 |
| 입력 read payload | 256KiB | 160KiB | 128KiB |
| CB 용량, depth2 | 12KiB | 16KiB | 68KiB |

blocked의 입력 payload 모델은 baseline 대비 **50%**, reuse_a 대비 **20%** 감소합니다.
대신 depth2 CB 구성 용량은 68KiB, 최대 destination tile slots는 4개입니다.
출력 쓰기량은 세 버전 모두 32KiB로 같으며 read+write 모델의 baseline 대비 감소는 약 44.4%입니다.

### 더 많은 재사용이 항상 이득은 아닙니다

M=32,K=128,N=96에서는 출력 타일 행이 하나뿐이라 B의 행 방향 재사용이 없습니다.
reuse_a는 A/B를 4/12회, blocked는 8/12회 읽습니다.
blocked가 N 블록마다 A를 다시 읽기 때문에 입력 payload는 **32KiB → 40KiB**로 늘어납니다.
96,64,160에서도 blocked는 38타일, reuse_a는 36타일을 읽어 더 불리합니다.

따라서 이 구현을 모든 shape의 기본값으로 바꾸지 않았습니다. 세 모드를 명시적으로 선택하고,
타일 grid·데이터 이동량·L1 비용을 함께 비교합니다. 자동 shape 기반 선택은 아직 구현하지 않았습니다.
depth2가 실제 overlap/speedup을 제공하는지, 실물 latency/bandwidth는 아직 측정하지 않았습니다.

## 재현과 증거

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_matmul.sh
```

빌드: `results/matmul/build.log`. 각 실행의 전체 `.log`와 exit code, 수치 오차,
compiler/OS/버전/source SHA-256, 구조 비교는 새 `results/matmul/<UTC>/`에 저장됩니다.

- [실행 메타데이터](../results/matmul/20261007T082755057050Z/metadata.json)
- [blocked depth1 결과](../results/matmul/20261007T082755057050Z/blocked_depth1.json)
- [blocked depth2 결과](../results/matmul/20261007T082755057050Z/blocked_depth2.json)
- [비교 평가 JSON](../results/matmul/20261007T082755057050Z/design-assessment.json)
- [비교 표](../results/matmul/20261007T082755057050Z/design-assessment.md)
- [blocked 전체 로그](../results/matmul/20261007T082755057050Z/blocked_depth2.log)

평가 도구는 여섯 설정의 사례 수/입출력 fingerprints/기능 검사와 전송량·CB·register 모델을 대조합니다.
최신 실행이 실패하면 이전 성공 결과를 대신 평가하지 않고 오류로 종료합니다.

## 다음 단계

K-block streaming으로 Kt 범위를 확장하고, shape에 따른 재사용 방식 선택을 검토합니다.
이후 동일 데이터형·입력의 CUDA 비교, 실제 장치 측정, multi-core 분할로 연결할 수 있습니다.
시뮬레이터 KHz나 wall time을 실제 NPU 처리율/latency로 보고하지 않습니다.

후속 완료: [12. K-streamed MatMul](12-k-streamed-matmul.md)에서 일정한 입력 CB 용량으로 K<=512를 검증했습니다. 큰 K의 fractional 누적 오차를 발견해 모든 최종 비교 버전의 destination 누적을 FP32로 바꿨습니다. 이 문서의 84개 결과는 이전 설정의 기록입니다.
