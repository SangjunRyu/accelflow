# 29. Metalium 단계 최종 결과와 TT-NN 인계

상태: **저수준 커널 학습 범위 구현·검증 완료, TT-NN은 후속 단계**. 날짜: 2026-11-10.
[진행 순서](README.md) · 이전: [28. 열 분할 RMSNorm](28-column-sharded-rmsnorm.md)

## 여기까지 완료한 범위

AccelFlow는 CUDA에서 익힌 메모리 이동·타일링·동기화를 가상 Wormhole의 TT-Metalium 커널로
옮겨 보는 프로젝트다. 환경 설치에서 시작해 reader/compute/writer, CB 수명, 데이터 재사용,
코어 간 통신과 정규화 fusion까지 구현했다. 같은 입력으로 CPU·PyTorch 기준과 대조하고
설계를 바꿀 때 전송량·L1·동기화 비용이 어떻게 달라지는지 기록했다.

이 문서는 지금까지 저장한 실험의 정리다. 이전 단계 전체를 한 번에 재실행한 통합 인증 결과는 아니다.
TT-NN 연산, 학습·backward, 모델 전체 실행, 실물 Tenstorrent 칩 성능 측정은 완료 범위에 포함하지 않는다.

| 주제 | 구현·확인한 내용 | 대표 근거 |
| --- | --- | --- |
| 환경·재현 | 고정 TT-Metal/TTSim, 공식 예제 결과 21, 자체 실행 스크립트 | [04](04-ttsim-installation.md), [06](06-tt-metal-build-validation.md) |
| Vector Add | scalar/tiled, reader·compute·writer와 CB recycling | [08](08-vector-add.md), [09](09-vector-add-design-assessment.md) |
| MatMul 재사용 | 타일/face 배치, A 재사용, 2×2 blocked, K-streamed FP32 누적 | [10](10-matmul.md)–[12](12-k-streamed-matmul.md) |
| MatMul 통신 | 독립 multicore, A multicast, 2D A/B multicast, K 배치 전송 | [15](15-independent-multicore-matmul.md), [17](17-a-multicast-matmul.md)–[20](20-batched-2d-multicast-matmul.md) |
| CUDA 비교 | naive/tiled MatMul, cudaArray 입력 경로, reduction 3종, RMSNorm fusion | [21](21-cuda-matmul-comparison.md), [23](23-texture-matmul-and-profiler.md)–[25](25-rmsnorm.md) |
| Reduction·정밀도 | SFPU scan/carry, padding, FP32 저장과 operand 경로의 정밀도 차이 | [22](22-row-wise-reduction.md), [25](25-rmsnorm.md) |
| RMSNorm fusion | 제곱·통계·정규화를 한 프로그램으로 연결, 중간 DRAM 제거 | [26](26-npu-rmsnorm-fusion.md) |
| RMSNorm 병렬화 | 독립 행 분할, 열 분할 부분합 gather·대표 코어 집계·전체 합 전달 | [27](27-multicore-rmsnorm.md), [28](28-column-sharded-rmsnorm.md) |

## 설계를 바꾸며 확인한 것

MatMul에서는 데이터를 덜 읽는 대신 보관 공간이 늘어나는 관계를 확인했다. 128×128×128의
blocked 입력 payload 모델은 baseline의 256 KiB에서 128 KiB로 줄지만 depth2 CB는
12 KiB에서 68 KiB로 늘었다. K streaming에서는 중간 BF16 pack/spill을 피하고 FP32 DST에
부분합을 유지했다. 이 수치는 소스에서 계산한 작업량·저장량이며 실물 bandwidth 측정값은 아니다.

통신에서는 DRAM 읽기를 줄여도 빠르다는 보장이 없었다. 2D multicast가 모델상 DRAM 읽기를
줄인 조건에서도 독립 multicore보다 TTSim 시간이 길었다. K 배치 4·8은 semaphore 교환 모델을
70~80% 줄였고 320/640 실험에서 기존 2D보다 host/TTSim 비용이 약 36% 줄었다.
독립 multicore보다는 여전히 길었다. 느린 결과와 변동은 [19](19-multicast-simulator-performance.md)·[20](20-batched-2d-multicast-matmul.md)에 남겼다.

RMSNorm에서는 3-stage의 제곱·prefix 중간 DRAM을 fused 경로의 L1 cache로 옮겼다.
행 분할은 코어 간 통신 없이 처리할 행을 나누며 열 분할은 긴 한 행에도 여러 코어를 사용한다.
열 분할로 코어당 입력 cache를 줄일 수 있지만 partial/global 통계 전달과 대표 코어의 집계가 추가된다.
두 분할 방식은 입력 shape와 자원·통신 비용을 보고 선택해야 한다.

정확성 검사도 설계의 일부였다. BF16 입력을 공통 binary fixture로 고정하고 fingerprint와
SHA-256을 보존했다. RMSNorm은 PyTorch FP32·FP64와 대조했다. FP32로 buffer를 저장해도
SrcA/B를 거치며 operand 정밀도가 줄어드는 사례를 발견해 hi/low 또는 hi/middle/low 조각으로
전달·재구성했다. 재구성과 통계 덧셈은 compute engine에서 수행하고 host는 검증만 맡았다.

## 측정 결과를 설명하는 기준

| 근거 종류 | 설명할 수 있는 범위 | 설명할 수 없는 범위 |
| --- | --- | --- |
| MX150 CUDA event·host 측정 | 해당 GPU·driver·toolkit·입력·측정 구간의 실행 비용 | 최신 GPU 일반 성능이나 Tenstorrent와의 칩 속도 순위 |
| TTSim 실행·전체 출력 검증 | 저장한 입력의 기능, 통신·동기화 경로, simulator/host 실행 비용 | 실물 NPU latency·TOPS·전력·확정적 코어 가속률 |
| source payload·CB 모델 | 요청 bytes, 호출 횟수, 설정한 CB payload의 변화 | cache hit·NoC 혼잡·전체 L1 사용량·hardware counter |
| PyTorch CPU 대조 | 고정 dtype·epsilon·허용 오차의 수치 결과 | PyTorch GPU 성능, 임의 입력·학습 전체의 정확성 |

nvprof/CUPTI counter 수집은 실패했고 Nsight Systems trace로 shared-memory bank conflict를
정량 검증하지 못했다. cudaArray 경로도 비교했지만 bank conflict를 해소했다고 주장하지 않는다.
구체적인 실패와 관측 범위는 [23번](23-texture-matmul-and-profiler.md)에 있다.

## 포트폴리오에 사용할 설명

> CUDA와 가상 Wormhole에서 MatMul·Reduction·RMSNorm을 구현했다. 타일 재사용,
> multicast batching, fusion, 행·열 분할을 단계적으로 비교하고 공통 BF16 입력과
> CPU/PyTorch 기준으로 결과를 검증했다. 실제 GPU 측정값, simulator 실행 비용,
> 소스 기반 메모리·통신 모델을 구분해 최적화의 효과와 비용을 기록했다.

소개할 대표 사례는 MatMul K 배치 통신, RMSNorm 중간 DRAM 제거, RMSNorm 행/열 분할이다.
각 사례는 원래 문제 → 바꾼 데이터 흐름 → 정확성 근거 → 자원 비용 → 측정 한계 순서로 설명하면 된다.
개별 단계의 검증 수는 조건과 반복 단위가 달라 전부 더해 하나의 테스트 수로 소개하지 않는다.

## TT-NN으로 이어갈 범위

Metalium의 교육용 구현은 여기서 한 단계 마무리한다. 남은 최적화 후보인 blocked+multicast 결합,
트리 통계 reduction, gamma caching과 load/compute 중첩은 선택 실험이다.
TT-NN을 먼저 익힌 뒤 공식 program configuration·sharding과 비교해 우선순위를 정한다.

다음 세션에서는 현재 TT-Metal v0.79.0/TTSim v1.11.1 조합에 맞는 Python 바인딩과
TT-NN 연산 실행 가능 여부부터 확인한다. 최신 wheel이나 source를 기존 환경에 바로 덮어쓰지 않는다.
새 빌드·의존성이 필요하면 C: 실제 여유와 WSL 메모리를 다시 확인하고 버전별 build tree를 분리한다.
이번 열 분할 작업 시작 시 C: 여유는 약 10.0 GiB였으며 현재 공간은 다음 설치 전에 다시 측정해야 한다.

먼저 작은 MatMul·RMSNorm을 실행하고 기존 fixture로 PyTorch·Metalium·TT-NN을 대조한다.
dtype·출력 형식·합산 순서가 다를 수 있어 조건별 오차를 기록한다. 이후 row-major/tile 변환,
DRAM/L1, interleaved/sharded 배치와 선택되는 커널을 살펴보고 작은 RMSNorm/MLP/residual 블록으로 확장한다.
현재 환경의 TT-NN 호환성과 실행 성공은 아직 확인하지 않았다.

다음 세션에 전달할 내용:

```text
프로젝트 /home/ysjun5656/dev/accelflow에서 TT-NN 단계를 진행해줘.
AGENTS.md, docs/README.md, docs/28-column-sharded-rmsnorm.md,
docs/29-metalium-results.md를 먼저 읽어줘. 후속 문서는 33번부터 작성해줘.
현재 TT-Metal v0.79.0 / TTSim v1.11.1을 보존하고 호환성·디스크·메모리부터 확인해줘.
목표는 작은 TT-NN MatMul/RMSNorm 실행, 기존 입력을 이용한 정확성 대조,
layout·메모리·sharding과 구현 경로 학습, 작은 연산 블록으로의 확장이야.
TTSim 시간을 실제 NPU 성능으로 해석하지 말고 실행하지 않은 단계는 구분해줘.
```

공개용 대표 실험표와 단일 재현 진입점은 [30번](30-portfolio.md)에 정리했다. TT-NN 후속 문서는 33번부터 이어간다.

FP32-to-Dest의 native 실패와 로컬 호환 경로의 비트 분할 없는 계산은 [31번](31-fp32-to-dest-rmsnorm.md)에 별도 기록했다. TT-NN 후속은 33번부터 이어간다.

같은 3-stage 조건의 split/FP32-to-Dest 실행 비용 비교는 [32번](32-rmsnorm-fp32-dest-timing.md)에 기록했다.
