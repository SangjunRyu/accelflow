# 30. 공개 포트폴리오와 대표 실험

상태: **저장 근거 대조·대표 simulator 재현 경로 검증 완료**. 날짜: 2026-11-10.
[README](../README.md) · [진행 순서](README.md) · [Metalium 결과와 TT-NN 인계](29-metalium-results.md)

## 프로젝트에서 맡은 범위

자체 host, data-movement/compute kernel, 실행·검증 스크립트를 작성했다.
상위 기반은 공식 TT-Metal v0.79.0, TTSim v1.11.1과 CUDA 12.9다. TT-Metal runtime이나
TTSim 자체를 구현한 프로젝트는 아니다. PyTorch는 RMSNorm의 독립 CPU 기준으로 사용했다.
TT-NN과 모델 실행은 아직 하지 않았으며 다음 세션에서 이어갈 범위다.

타일 배치를 이해하기 위한 host `pack`은 공식 `tilize_nfaces` 결과와 대조한다.
이 함수가 공식 `convert_layout`에 없는 기능을 제공하거나 더 빠르다고 주장하지 않는다.
Metalium API로 데이터 흐름·CB 수명·동기화를 직접 구성하고 그 결과를 검증한 것이 구현 범위다.

## MatMul 설계 변경과 근거

비교 기준은 BF16 입출력, HiFi4, FP32 destination 누적, CB depth2의 128×128×128 정수 입력이다.
입력 A/B와 출력 fingerprint가 7개 구성에서 모두 같고 CPU FP64 기준 오차는 0이다.
단위는 KiB이며 DRAM read와 CB 값은 source model이다.

| 구현 | 변경 이유와 확인한 내용 | 활성 코어 | DRAM read 모델 | CB 코어당 / 전체 |
| --- | --- | --- | --- | --- |
| [tile baseline](10-matmul.md) | 출력 타일마다 A/B를 읽는 기준 경로 | 1 | 256 | 12 / 12 |
| [A reuse](10-matmul.md) | 같은 A 타일을 출력 열마다 다시 읽는 요청 제거 | 1 | 160 | 16 / 16 |
| [2×2 blocked](11-blocked-matmul.md) | 출력 4타일에서 A/B를 재사용하되 보관 공간 증가 | 1 | 128 | 68 / 68 |
| [독립 multicore](15-independent-multicore-matmul.md) | 출력 타일을 나눠 여러 코어에 배정; 총 읽기량 유지 | 4 | 256 | 12 / 48 |
| [A multicast](17-a-multicast-matmul.md) | 중복 A 읽기를 sender/receiver 전달로 대체 | 4 | 160 | 12 / 48 |
| [2D A/B multicast](18-2d-multicast-matmul.md) | A는 행 방향, B는 열 방향으로 공유 | 4 | 128 | 12 / 48 |
| [2D batch4](20-batched-2d-multicast-matmul.md) | K 타일 4개를 묶어 ready/data/valid 교환 감소 | 4 | 128 | 36 / 144 |

baseline에서 reuse는 같은 A를 반복 읽는 비용을 줄인다. blocked는 출력 4타일에 걸쳐 A/B를
재사용해 읽기 모델을 더 줄이지만 저장 공간과 destination 슬롯을 늘린다. K streaming은
K 전체 cache 대신 chunk를 사용하면서 부분합을 FP32 DST에 유지한다. [10](10-matmul.md)–[12](12-k-streamed-matmul.md).

multicore는 출력 소유권을 나누며 총 입력 읽기량은 baseline과 같다. multicast는 중복 DRAM
읽기를 줄이는 대신 NoC 전달·ready/valid·sender 역할을 추가한다. reuse/blocked와 multicore는
별도 분기이며 앞 단계 전체를 합친 단일 커널의 누적 개선율을 뜻하지 않는다. reader NoC와 출력
분할도 달라 독립 multicore와 multicast의 시간을 순수 transport 차이 하나로 해석하지 않는다.

128³에서 2D batch4의 A+B payload multicast 호출 모델은 64에서 16으로 줄고
DRAM read는 128 KiB로 유지된다. 코어당 CB는 12에서 36 KiB로 늘어난다.
이 shape의 Kt=4와 다음 320 입력의 Kt=10을 구분한다. Kt=10은 batch4가 4+4+2,
batch8이 8+2로 마지막 짧은 배치를 처리한다.

## 세 종류의 숫자를 구분하는 방법

| 종류 | 출처와 범위 | 해석의 한계 |
| --- | --- | --- |
| CUDA 실측 ms | Windows native MX150에서 GPU event 측정 | 해당 GPU·입력·event 구간의 비용; 칩 일반 성능으로 확대하지 않음 |
| TTSim 관측 ms | WSL host clock의 enqueue→Finish 경과 시간 | runtime·slow dispatch·simulator 대기 포함; 실물 NPU latency 아님 |
| source model bytes/calls | reader·CB·semaphore 루프에서 계산 | 요청량·설정 용량; 실제 DRAM transaction·cache·NoC stall counter 아님 |

CUDA는 BF16으로 양자화한 입력을 FP32로 확장해 저장·연산·출력한다. NPU MatMul은 BF16
타일로 저장하고 FP32로 누적한 뒤 BF16으로 출력한다. 같은 입력값이어도 저장 bytes와 출력
형식까지 같은 것은 아니다. 소수 출력은 각 조건의 오차 기준으로 대조하며 bitwise 일치를 일반화하지 않는다.

### CUDA 실측

20회 warmup 뒤 sample 15개를 수집했다. sample 하나는 kernel20회의 event batch 시간/20이다.
process별 sample 중앙값을 구한 뒤 process3개의 중앙값을 사용했다. H2D/D2H·초기화·검증은
event 구간에서 제외했다. GPU clock·다른 GPU 작업을 고정하지 않아 변동을 보존한다.

| M,K,N | naive ms | tiled ms | naive/tiled |
| --- | --- | --- | --- |
| 128,128,128 | 0.08369 | 0.08890 | 0.94 |
| 320,320,320 | 1.68745 | 0.74112 | 2.28 |
| 640,320,640 | 7.62204 | 2.31803 | 3.29 |

128³ tiled의 process 중앙값 범위는 0.04980–0.11623 ms, 320³은 0.61809–2.36498 ms다.
128³은 대표값에서 tiled가 느리다. 640×320×640은 같은 MX150 조건에서 tiled가 짧았지만
global 접근·재사용·barrier가 함께 바뀌므로 shared memory 하나의 효과로 분리하지 않는다.
cuBLAS·Tensor Core·bank conflict counter를 사용한 최적화 검증은 하지 않았다.
[21번의 방법·입력 대조](21-cuda-matmul-comparison.md), [23번 profiler 진단](23-texture-matmul-and-profiler.md).

### TTSim 관측과 소스 모델

4코어/depth2, process3회, warmup2/sample5다. 각 process sample 중앙값을 다시 process 사이에서
중앙값으로 요약했다. 순서를 seed로 섞고 순차 실행했으며 JIT cache는 지우지 않았다.
입력 전송·packing·출력 sentinel 초기화·readback·CPU 검증은 시간에서 제외했다.
DRAM read와 CB 열은 시간과 함께 수집한 source model이며 별도의 단위로 표시한다.

| M,K,N | 구현 | host/TTSim ms | DRAM read 모델 KiB | 코어당 CB 모델 KiB |
| --- | --- | --- | --- | --- |
| 320,320,320 | multicore | 512.976 | 4000 | 12 |
| 320,320,320 | mcast | 882.779 | 2600 | 12 |
| 320,320,320 | mcast2d | 1162.351 | 2000 | 12 |
| 320,320,320 | mcast2d_batch1 | 1393.987 | 2000 | 12 |
| 320,320,320 | mcast2d_batch4 | 746.244 | 2000 | 36 |
| 320,320,320 | mcast2d_batch8 | 780.125 | 2000 | 68 |
| 640,320,640 | multicore | 2329.654 | 16000 | 12 |
| 640,320,640 | mcast | 2806.735 | 10000 | 12 |
| 640,320,640 | mcast2d | 4629.212 | 8000 | 12 |
| 640,320,640 | mcast2d_batch1 | 6026.675 | 8000 | 12 |
| 640,320,640 | mcast2d_batch4 | 3241.892 | 8000 | 36 |
| 640,320,640 | mcast2d_batch8 | 2975.694 | 8000 | 68 |

320³의 batch4는 기존 2D보다 35.8%, 640×320×640의 batch8은 35.7% 짧았다.
ready atomic·payload multicast 호출 모델은 Kt=10에서 batch4/8이 70%/80% 감소한다.
DRAM read bytes는 변하지 않는다. batch1은 기존 2D보다 느렸고 선택한 batch4/8도
독립 multicore보다 길었다. 신호 횟수 감소와 CB 확대·read 발행·새 루프의 효과가 섞여 있으므로
semaphore만이 병목이라고 확정하지 않는다. [20번 전체 조건](20-batched-2d-multicast-matmul.md).

CUDA와 TTSim은 실행 대상·정밀도·시간 범위가 다르다. 두 표 사이의 GPU/NPU 속도 비율은 계산하지 않는다.
서로 다른 날짜의 simulator 측정도 동일 성능 기준으로 합치지 않는다.

## RMSNorm으로 확장한 내용

3-stage 제곱→합산→정규화를 한 프로그램의 L1 CB 흐름으로 연결해 중간 DRAM buffer를 제거했다.
저장된 BF16 fixture·epsilon을 고정하고 CPU/PyTorch와 대조했다. 행 분할은 통신 없이 다른 행을
처리하고 열 분할은 같은 행의 부분합을 leader로 모아 compute에서 더한 뒤 전체 합을 돌려준다.

1×1025의 열 분할 4코어는 행 분할 1코어와 비교해 최대 코어당 CB가 316→132 KiB지만
전체 CB는 316→504 KiB다. peer 통계 payload 모델은 768 bytes가 추가된다.
행·열 49개 구성에서 PyTorch FP32·FP64 대조를 통과했고 보조 검증 포함 76회 RMSNorm 연산을
실행했다. PyTorch FP32 대비 최대 절대 오차는 약 2.38e-7이다. 합산 순서가 달라 열 분할의
bitwise 일치를 필수 조건으로 삼지 않았다. [26](26-npu-rmsnorm-fusion.md)–[28](28-column-sharded-rmsnorm.md).

## 한 진입점으로 확인하기

Linux/WSL에서 Python3.10 이상과 Git으로 저장 기록부터 검사한다. GPU·TTSim·PyTorch 설치는 필요 없다.

```bash
git clone https://github.com/SangjunRyu/accelflow.git
cd accelflow
bash scripts/reproduce_portfolio.sh --evidence
```

`EVIDENCE_PASS sources=68 models=7 cuda_rows=6 ttsim_rows=12`를 확인한다.
원본 68개 SHA-256을 확인하고 CUDA18개/TTSim36개 trial JSON의 raw sample 중앙값을 다시
계산해 저장 summary와 비교한다. 대표 정수 입력의 A/B·출력 fingerprint와 통과 상태도 확인한다.
검사 대상은 [대표 manifest](../results/portfolio/20261110-public-summary/evidence.json)의 source 목록에 있다.
이 단계는 저장 기록 감사이며 커널 재실행이나 원본 진위의 외부 인증을 뜻하지 않는다.

가상 장치에서 실행하려면 [04번 TTSim](04-ttsim-installation.md)과 [06번 빌드](06-tt-metal-build-validation.md)의
고정 환경이 선행돼야 한다. 기본 위치는 아래 두 경로이며 다른 위치는 환경 변수로 지정한다.

```bash
export TT_METAL_HOME="$HOME/dev/tenstorrent/tt-metal-v0.79.0"
export TT_METAL_SIMULATOR="$HOME/dev/tenstorrent/sim/libttsim_wh.so"
bash scripts/reproduce_portfolio.sh --simulate
```

기존 `build_Release`의 TT-Metal 라이브러리와 Clang20/CMake/Ninja를 사용해 자체 host만 빌드한다.
64×64×96의 한 입력에서 6개 구현을 각각 warmup2/sample5로 실행한다. 매번 NaN sentinel을
다시 쓰며 반복 출력, CPU 기준값, padding, 공식 tilizer와의 일치를 검사한다. mode 사이 입력·출력
fingerprint도 같아야 통과한다. 각 mode는 process 한 번이므로 과거 성능 표의 통계 재현은 아니다.

이번 경로는 6개 mode·42회 제출·CPU 기준 오차 0으로 통과했다.
[실제 명령·source/executable SHA-256·exit code](../results/portfolio/20261110T090405563804Z/metadata.json),
[baseline 출력·검증](../results/portfolio/20261110T090405563804Z/baseline.json),
[batch4 로그](../results/portfolio/20261110T090405563804Z/mcast2d_batch4.log).
전체 188개 회귀나 큰 입력 시간 재측정은 각 실험 문서의 스크립트로 별도 실행한다.

새로운 TT-Metal 빌드가 필요하면 WSL VHDX가 있는 물리 volume의 공간과 메모리를 먼저 확인한다.
이 진입점은 upstream 전체 빌드·패키지 설치·driver 변경을 하지 않는다. CUDA 재실행은
Windows native [21번 경로](21-cuda-matmul-comparison.md)를 사용하며 이 smoke test에 포함하지 않는다.

## 지원서에서 설명할 내용

> MatMul의 타일 재사용, multicore 데이터 공유와 batching을 구현하고 읽기 요청량·L1 비용·
> 동기화 횟수의 변화를 비교했다. CUDA는 MX150에서 측정하고 NPU는 TTSim에서 기능을 검증했다.
> RMSNorm에서는 중간 DRAM 제거와 행/열 분할을 구현했으며 공통 BF16 입력과 CPU/PyTorch
> 기준으로 결과를 대조했다. 느린 결과와 측정 한계도 원본 로그와 함께 보존했다.

현재 완료 범위는 저수준 커널과 위의 실험이다. TT-NN 기준 연산·sharding·작은 연산 블록은
다음 단계다. 후속 실행 문서는 33번부터 작성하면 된다. [29번 인계](29-metalium-results.md).

FP32-to-Dest의 native 실패와 로컬 호환 경로의 비트 분할 없는 계산은 [31번](31-fp32-to-dest-rmsnorm.md)에 별도 기록했다. TT-NN 후속은 33번부터 이어간다.

같은 3-stage 조건의 split/FP32-to-Dest 실행 비용 비교는 [32번](32-rmsnorm-fp32-dest-timing.md)에 기록했다.
