# 15. 단순 Multicore MatMul — 출력 타일 분할

상태: **독립 코어 MatMul 구현·실행 검증 완료**. 작업 날짜: 2026-10-19.
[진행 순서](README.md) · 이전: [14. 컴파일 관찰](14-compiler-inspection-and-tt-explorer.md)

## 구현 범위

`multicore`는 C의 출력 타일을 1·2·4개 코어에 나눠 처리하는 독립 코어 방식입니다.
각 코어가 DRAM A/B에서 직접 읽고 자신의 CB와 Tensix를 사용합니다.
출력 DRAM C는 공유하되 각 코어가 서로 다른 타일 구간을 씁니다.

이 버전은 baseline을 여러 코어로 나눈 기준 구현입니다. A/B 재사용·2×2 blocked·K-chunk 버전을
여러 코어에 확장한 것은 아닙니다. K축은 분할하지 않고 각 출력의 K 전체를 같은 코어가 누적합니다.
sender/receiver, multicast, 코어 간 partial sum 교환은 아직 구현하지 않았습니다.

## 코드 읽는 순서

1. [matmul.cpp](../npu/matmul/matmul.cpp): `multicore` 선택, 공통 입력 생성/DRAM 업로드/CPU 검사
2. [multicore.hpp](../npu/matmul/multicore.hpp): 코어 선택, CB/커널 생성, runtime args, 출력 구간 분배
3. [reader_multicore.cpp](../npu/matmul/kernels/reader_multicore.cpp): 전역 출력 번호에서 A/B 주소 계산
4. [compute_multicore.cpp](../npu/matmul/kernels/compute_multicore.cpp): 코어별 K 전체 누적
5. [writer_multicore.cpp](../npu/matmul/kernels/writer_multicore.cpp): 자기 구간을 전역 DRAM C에 저장

학습 중이던 공식 API도 명시적으로 볼 수 있게 reader/compute/writer 반환값을 `KernelHandle`로
선언했습니다. `DeviceLocalBufferConfig`는 기존 `common.hpp`의 `af::buffer()`에서 계속 사용합니다.

## 출력 타일 분배

`Mt=ceil(M/32)`, `Kt=ceil(K/32)`, `Nt=ceil(N/32)`이고 총 출력 타일은 `T=Mt×Nt`입니다.
실제 활성 코어 수는 `P=min(requested,T)`입니다. 출력이 한 타일이면 코어 4개를 요청해도 하나만
구성합니다. 작업이 없는 코어에는 CB/커널을 생성하지 않습니다.

```text
q = T / P, r = T % P
코어 i의 타일 수 = q + (i < r ? 1 : 0)
start = 앞 코어들이 담당한 타일 수의 합
담당 구간 = [start, start + count)
```

코어별 타일 수 차이는 최대 1입니다. 현재는 작업별 Kt가 같으므로 코어별 matmul 호출 수 차이는
최대 Kt입니다. 논리 core grid 크기는 실제 `MeshDevice::compute_with_storage_grid_size()`에서
조회하고 row-major `(i%grid.x,i/grid.x)`로 선택합니다. 물리 NoC 좌표를 하드코딩하지 않습니다.

예: M=64,K=64,N=96이면 출력 타일은 2×3=6개입니다. 4코어 요청 시:

| 논리 코어 | 전역 C 타일 구간 | C 타일 좌표 (m,n) |
| --- | --- | --- |
| (0,0) | [0,2) | (0,0), (0,1) |
| (1,0) | [2,4) | (0,2), (1,0) |
| (2,0) | [4,5) | (1,1) |
| (3,0) | [5,6) | (1,2) |

위 좌표는 grid.x>=4인 현재 환경의 예입니다. reader는 각 전역 출력 번호 `t`에 대해
`m=t/Nt`, `n=t%Nt`로 복원하므로 담당 구간이 출력 행 경계를 넘어도 정확히 처리합니다.
입력 타일 번호는 A=`m×Kt+k`, B=`k×Nt+n`, 출력은 C=`t`입니다.

## 코어 내부 흐름과 메모리

```text
공유 DRAM A/B
    ├─ core 0: reader → L1 CB0/1 → compute → L1 CB16 → writer → C 구간 0
    ├─ core 1: reader → L1 CB0/1 → compute → L1 CB16 → writer → C 구간 1
    └─ ...
```

각 코어의 reader는 RISCV_1, writer는 RISCV_0에서 동작합니다.
CB0/CB1/CB16이라는 ID는 같아도 L1 버퍼는 코어마다 독립적으로 생성합니다.
각 출력 타일의 K축 누적은 FP32 destination으로 수행하고 최종 출력은 BF16입니다.
HiFi4, 입력 배치, 수치 허용 오차는 기존 baseline과 같습니다.

read DMA 완료 후 CB push, compute 완료 후 output push, write DMA 완료 후 pop의 순서를 지킵니다.
코어 사이의 semaphore는 없습니다. 공유 C에 대한 쓰기 구간이 겹치지 않고 코어 간 입력 의존성이
없으므로 이 버전에는 필요하지 않습니다. 전체 workload의 완료는 host의 `Finish`로 기다립니다.

## 소스 기반 비용 모델

| 지표 | 단일 코어 baseline | 독립 multicore |
| --- | --- | --- |
| 전체 A tile reads | Mt×Kt×Nt | 동일 |
| 전체 B tile reads | Mt×Kt×Nt | 동일 |
| C tile writes | Mt×Nt | 동일 |
| 전체 matmul_tiles 호출 | Mt×Kt×Nt | 동일 |
| 전체 read barrier 호출 | 2×Mt×Kt×Nt | 동일 |
| 코어별 CB payload bytes | 2048×3×depth | 동일 |
| 전체 CB payload bytes | 2048×3×depth | 2048×3×depth×P |

depth2는 코어별 12KiB, 실제 활성 코어가 4개면 합계 48KiB입니다.
이 합계는 코어마다 나뉜 L1 용량이며 한 코어에 48KiB를 배정한다는 의미가 아닙니다.
CB payload는 전체 L1 사용량이나 destination register 크기와 다릅니다.

독립 multicore는 일을 분산하지만 동일 A/B의 코어 간 중복 읽기를 줄이지 않습니다.
위 값은 소스 기반 모델이며 실행 중 DRAM counter나 물리 packet/cycle 측정이 아닙니다.
실물 병렬 speedup이나 TTSim의 multicore 실행 시간은 이번 작업에서 측정하지 않았습니다.

## 검증

baseline depth1/2와 multicore 요청 1/2/4 × depth1/2를 같은 20개 입력으로 실행했습니다.
기존 deterministic, zero, identity, 음수 정수, fractional, 비정방형/타일 경계, K<=512 입력입니다.

**8개 설정 × 20개 입력 = 총 160개 모두 PASS**, 여덟 실행 모두 exit code 0입니다.
모든 설정의 입력/출력 fingerprints가 baseline과 일치했습니다.
기존 단일 코어·blocked·K-streamed 188개 suite도 다시 통과했습니다.
지원하지 않는 core 수 0/3은 장치 생성 전에 exit code 1로 거부하는 것을 확인했습니다.

각 실행은 CPU FP64 reference와 원소별 비교하고, 공식 tilizer 일치·NaN sentinel 덮어쓰기·padding 0을
검사합니다. 모드/코어 수/CB 깊이 사이의 입력·출력 FNV fingerprints도 일치해야 합니다.
정수 fixture는 exact, 소수 fixture는 기존 `0.03+0.03×abs(reference)` 기준을 사용합니다.

추가 평가는 다음을 검사합니다.

- 요청한 코어 수가 출력 타일 수보다 클 때 활성 코어 축소
- 균등하지 않은 나눗셈과 코어별 타일 수 차이 <=1
- 전역 C 타일 구간의 빈틈·중복 없음, 논리 코어 좌표 중복 없음
- 담당 구간이 출력 행 경계를 넘는 사례
- 각 코어에서 output CB 깊이보다 많은 타일을 처리하는 CB wrap 사례
- 4개 활성 코어와 K=512 사례
- 전체 DRAM 읽기/쓰기 모델이 baseline과 같고 CB 합계가 활성 코어 수에 비례함

## 실행과 증거

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_multicore_matmul.sh
```

실행 파일의 인자는 `matmul report.json multicore depth [requested_cores]`입니다.
depth는 1/2, requested_cores는 1/2/4이며 생략 시 4입니다.
환경 변수·descriptor와 실제 8개 실행은 wrapper가 준비합니다.
기존 `bash scripts/run_matmul.sh`는 기존 10개 설정/188개 suite를 유지합니다.
이 단계에서 `--benchmark` 경로에 multicore를 연결하지는 않았습니다.

- [메타데이터·버전·소스 SHA-256](../results/matmul_multicore/20261019T104604881253Z/metadata.json)
- [분배·동일 입력/출력·모델 평가](../results/matmul_multicore/20261019T104604881253Z/design-assessment.json)
- [4코어/depth2 실행 결과](../results/matmul_multicore/20261019T104604881253Z/multicore_cores4_depth2.json)
- [4코어/depth2 전체 로그](../results/matmul_multicore/20261019T104604881253Z/multicore_cores4_depth2.log)
- [단일 코어 baseline](../results/matmul_multicore/20261019T104604881253Z/baseline_cores1_depth2.json)
- [기존 188개 회귀 검증](../results/matmul/20261019T104610065773Z/design-assessment.json)

참고 API는 고정된 [공식 v0.79.0 multicore 예제](https://github.com/tenstorrent/tt-metal/tree/v0.79.0/tt_metal/programming_examples/matmul/matmul_multi_core)입니다.
그 예제의 core-grid API를 확인했고 출력 구간 분배와 검증 로직은 프로젝트에서 구현했습니다.
TT-Metal 전체 재빌드, source revision 교체, 물리 장치용 component 설치는 하지 않았습니다.
빌드 전 C: 여유 약 16.9GiB, WSL 가용 메모리 약 3.3GiB를 확인하고 기존 라이브러리를 재사용했습니다.

## 다음 단계

이제 같은 출력 분할을 기준으로 sender가 A/B를 읽고 receiver로 전달하는 방식을 비교할 수 있습니다.
그 단계에는 코어 간 NoC 전송, 준비/소비 semaphore, local CB와 multicast 목적지 동기화가 필요합니다.
현재 independent multicore의 전체 입력 읽기량을 기준선으로 두고 감소량과 추가 동기화 비용을
함께 평가합니다. 아직 sender/receiver 구현이나 실물 성능 검증이 완료된 것은 아닙니다.

후속 측정 완료: [16. Multicore TTSim 시간 비교](16-multicore-simulator-performance.md)에서 동일 커널의 1/2/4코어와 CB depth1/2를 반복 측정했습니다. 이 문서의 미측정 설명은 최초 구현 단계의 기록이며, 현재 `--benchmark` multicore 경로도 구성되어 있습니다. 실물 성능은 계속 미측정입니다.

후속 구현: [17. A multicast](17-a-multicast-matmul.md)에서 sender가 A를 읽어 receiver에 전달하는 버전을 구현했습니다. B는 local read이고 2차원 A/B multicast는 아직 미구현입니다.
