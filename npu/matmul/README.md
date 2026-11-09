# AccelFlow MatMul

BF16 입력/출력, 단일 Tensix core, 자체 reader/compute/writer 커널의 타일 행렬곱입니다.

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_matmul.sh
```

`build/matmul/matmul`을 빌드하고 baseline/reuse_a/blocked/streamed의 10개 설정을 실행합니다.
같은 seed와 공통 14개 입력, K 확장 입력 6개를 사용합니다. 기존 blocked는 공통 입력만 지원합니다.
현재는 모든 비교 모드의 destination을 FP32로 누적하고 최종 결과를 BF16으로 pack합니다. CPU 참조는 양자화된 BF16 입력의
FP64 곱·누적이며, layout은 공식 `tilize_nfaces`와 비교합니다.

- **baseline**: 출력 tile (m,n)마다 A(m,k), B(k,n)를 읽어 K tile 축을 누적
- **reuse_a**: 한 A tile row를 CB에 유지해 모든 n 출력 열에서 재사용
- **blocked**: 출력 2×2 블록에서 A/B를 함께 재사용, Kt<=4
- **CB depth**: baseline 타일 수, reuse_a B/output 타일 수, blocked 입력 블록 수/output 타일 수
- **reuse A 용량**: streaming depth와 별개로 Kt tiles가 필요

M/K/N의 32배수 padding과 네 16×16 face packing/unpacking을 host가 수행합니다.
device/session/buffer 헬퍼는 기존 `../vector_add/common.hpp`를 공유합니다.
고정 TT-Metal v0.79.0의 이미 빌드된 라이브러리를 재사용하며 upstream은 수정하지 않습니다.

숫자 오차·source hash·실행 로그·구조 모델은 `results/matmul/<UTC>/`에 저장합니다.
[10. 구현·평가 보고서](../../docs/10-matmul.md).

API 기준은 [공식 v0.79.0 단일 코어 MatMul](https://github.com/tenstorrent/tt-metal/tree/v0.79.0/tt_metal/programming_examples/matmul/matmul_single_core)입니다.
원본 예제는 © Tenstorrent USA, Inc., Apache-2.0입니다. 타일 누적 API 흐름을 참고하고
새 reader 재사용 로직·CPU 참조·입력·검증·평가를 구현했습니다.

## 후속 blocked 모드

`blocked`는 최대 2×2 출력 타일에 필요한 A 두 타일 행/B 두 타일 열을 보관하고,
네 destination tile에 K축 부분 곱을 누적합니다. Kt<=4로 범위를 제한합니다.
`reader_blocked.cpp → compute_blocked.cpp → writer_blocked.cpp` 순서로 읽으면 됩니다.

현재 `run_matmul.sh`는 baseline/reuse_a/blocked × depth 1/2, 각 14개 입력으로
총 84개 사례를 검증합니다. 첫 구현의 네 실험/12개 입력 기록은 10번 문서에 보존했습니다.
blocked의 입력 depth는 블록 수, 출력 depth는 타일 수이며 CB 용량이 기존 모드와 다릅니다.
정확성과 크기별 전송량/L1 비용은 [11. Blocked MatMul](../../docs/11-blocked-matmul.md)을 참고하세요.

## 현재 K-streamed 모드

`streamed`는 K 타일을 chunk1/2개씩 공급하며 네 destination tile의 누적값을 유지합니다.
입력 CB 용량은 K가 아닌 chunk에 비례하며 chunk2/depth2는 36KiB입니다.
`reader_streamed.cpp → compute_streamed.cpp → writer_blocked.cpp` 순서로 읽으면 됩니다.
현재 runner는 10개 설정, 총 188개 사례를 실행합니다. K<=512까지 검증했습니다.
기존 누적 설정과의 수치 차이·범위는 [12. K streaming](../../docs/12-k-streamed-matmul.md)에 기록했습니다.

## 실행 비용 측정

`bash scripts/benchmark_matmul.sh`는 4개 shape, 36개 구성을 각각 2회 warmup 후
7회 측정합니다. 반복마다 NaN sentinel을 재설정하고 출력 전체를 비교합니다.
CPU 기준값 검사도 통과해야 보고서를 생성합니다. `--samples 15`로 표본 수를 바꿀 수 있습니다.
`results/matmul_perf/<UTC>/`의 raw samples와 median/p95는 host/simulator 제출·완료 시간이며
실물 NPU latency가 아닙니다. [13. 측정 범위·결과](../../docs/13-matmul-simulator-performance.md).

## 컴파일 산출물 학습

`bash scripts/inspect_compilation.sh`로 현재 host IR, streamed reader/unpack/math/pack/writer
ELF와 어셈블리를 확인합니다. TT-Explorer는 TT-MLIR 그래프용의 별도 학습 단계입니다.
[14. 도구와 읽는 순서](../../docs/14-compiler-inspection-and-tt-explorer.md).

## 독립 multicore

`bash scripts/run_multicore_matmul.sh`로 baseline과 요청 코어 1/2/4, CB depth 1/2를
같은 20개 입력에서 비교합니다. 160개 검증이 통과했습니다.
코어별 독립 reader/compute/writer와 CB가 있고 각 코어는 자기 출력 타일의 K 전체를 처리합니다.
`multicore.hpp`의 출력 분배·`KernelHandle`·runtime args부터 읽으면 됩니다.
전체 입력 DRAM payload는 baseline과 같고 sender/receiver는 아직 미구현입니다.
[15. 구현·분배 예·검증](../../docs/15-independent-multicore-matmul.md).

## Multicore TTSim 실행 비용 비교

`bash scripts/benchmark_multicore_matmul.sh`로 32개 구성 × 독립 process 3회,
각 process warmup2/timing7을 측정합니다. 요청 1/2/4코어와 실제 활성 코어 수를 구분하고
동일 CB depth에서 시간/변동/전체 CB/코어별 최대 작업량을 비교합니다.
`--trials 1 --samples 7`로 간단히 재실행할 수 있습니다.
[16. 독립 재측정과 해석](../../docs/16-multicore-simulator-performance.md).

## A multicast sender/receiver

`bash scripts/run_mcast_matmul.sh`로 A multicast와 baseline/독립 multicore를 비교합니다.
rank0이 A를 읽어 실제 NoC multicast로 receiver에 전파하고 B는 각 코어가 직접 읽습니다.
요청 1/2/4코어, depth1/2와 기존 20개 fixture를 사용해 200개 검증 및 42회 반복 실행을 통과했습니다.
활성 코어는 min(requested,Nt)이며 마지막 열 group의 idle receiver도 A CB를 소비합니다.
`mcast.hpp → reader_mcast.cpp → compute_mcast.cpp → writer_mcast.cpp` 순서로 읽으면 됩니다.
[17. 동기화·tail 처리·전송량](../../docs/17-a-multicast-matmul.md). B 열 multicast 구현은 아래 2D 절을 참고하세요.

## 2D A/B multicast

`bash scripts/run_mcast2d_matmul.sh`로 A 행 공유+B 열 공유를 baseline/독립/A-only와 비교합니다.
4코어는 기본 2×2, 한 축이 한 타일이면 최대 1×4/4×1 배치입니다.
A/B는 독립 semaphore 쌍을 사용하고 모든 코어가 각 단계에서 A→B 순서로 진행합니다.
240개 공통 검증과 98회 반복/fallback 실행, 기존 188개 회귀 검증이 통과했습니다.
`mcast2d.hpp → reader_mcast2d.cpp → compute_mcast2d.cpp → writer_mcast2d.cpp` 순서로 읽으면 됩니다.
[18. 경계·dummy·모델·증거](../../docs/18-2d-multicast-matmul.md).

## 독립/A-only/2D 실행 비용 비교

`bash scripts/benchmark_multicast_matmul.sh`는 실제 활성 4코어를 맞춘 세 전략과
독립1코어 대조군을 depth1/2에서 비교합니다. 4개 shape × 8개 구성 × process3회,
각 warmup2/timing7로 672개 표본을 저장하며 출력/CPU/분배/모델도 검증합니다.
느린 2D 결과와 변동을 포함한 [19. 비교 결과](../../docs/19-multicast-simulator-performance.md)를 참고하세요.

## K 배치 2D multicast

`mcast2d_batch1/4/8`은 K 방향 타일을 연속 L1 배치로 모아 배치마다 한 번 semaphore를 교환합니다.
`bash scripts/benchmark_batched_matmul.sh`로 정확성 200개와 320/640 크기 반복 비교를 실행합니다.
[20. 데이터 흐름·마지막 배치·시간·CB 비용](../../docs/20-batched-2d-multicast-matmul.md)을 참고하세요.
