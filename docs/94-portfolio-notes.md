# 94. Portfolio evidence — 참고 문서

[진행 순서와 문서 안내](README.md)

Phase 0: retain environment inspection and upstream installation-plan references.
Future case studies should explain problem, evidence, hypothesis, change, expected
effect, correctness validation, and limits of available hardware measurements.
Only claim execution for cases backed by saved logs and reports; keep future experiments separate.

Phase 1 evidence: [Vector Add](08-vector-add.md), uint32 scalar-loop baseline and
BF16 tiled reader/compute/writer pipeline, 21 validated cases, padding/CB recycling,
CPU numerical reference and explicit synchronization. No hardware speedup claims.

Design evidence: [09](09-vector-add-design-assessment.md) separates observed buffer
alignment and correctness from code-derived transfer/barrier/padding models.
CB depth doubles configured memory but does not reduce the DRAM payload.

Phase 2 evidence: [MatMul](10-matmul.md), BF16 single-core tile baseline and
A tile-row reuse, four configurations and 48 passing cases. Same input/output
fingerprints and CPU numerical comparison support functional equivalence.
For M=64,K=64,N=96 the source-derived read payload falls from 49,152 to 32,768
bytes (33.3%). This is a transfer model, not observed bandwidth or hardware speedup.
For K=128/depth2, configured CB capacity grows from 12 to 16 KiB: record the local
memory tradeoff alongside the reuse benefit. CUDA MatMul comparison was pending at this stage; see [21](21-cuda-matmul-comparison.md) for the completed MX150 work.

Blocked evidence: [11](11-blocked-matmul.md), 84 passing cases including odd output
blocks and repeated input CB wrap. At 128×128×128, modeled input payload falls
256→128 KiB against baseline, with depth2 CB capacity 12→68 KiB and maximum
destination tile slots 1→4. At 32×128×96 it is worse than reuse_a (40 vs 32 KiB),
so document shape-dependent tradeoffs rather than universal optimization claims.
Hardware speedup and bandwidth remain unmeasured.

K-streaming evidence: [12](12-k-streamed-matmul.md), 188 passing cases / 10 configurations.
At chunk2/depth2 the configured CB capacity remains 36KiB through K=512. Four
destination tiles retain partial sums across chunks without BF16 partial-result
pack/spill. Initial K=257 fractional baseline exceeded the existing tolerance; all
controlled configurations then switched to FP32 destination accumulation and passed
with unchanged tolerances. Read payload matches blocked, while read barriers increase.
This is a storage/precision design result, not measured hardware speedup.

Timing evidence: [13](13-matmul-simulator-performance.md), 36 configurations, two
warmups plus seven timed dispatches each (324 total). Timing scopes exclude input
transfers/readback/CPU verification; all repetitions match and final CPU comparisons
pass. Host/simulator median/p95/ranges are measured, with large scheduling-related
variation possible. Payload reductions do not establish physical hardware speedup.

Independent multicore evidence: [15](15-independent-multicore-matmul.md), 160 passing
cases and 188 existing regression cases. Contiguous global C tile ranges are balanced
over up to four logical cores, including uneven splits and fewer tiles than requested
cores. Each core uses independent L1 CBs and directly reads A/B from DRAM. Total
input payload matches baseline; configured CB capacity scales with active cores.
Sender/receiver sharing and physical parallel speedup remain unimplemented/unmeasured.

Multicore timing evidence: [16](16-multicore-simulator-performance.md), repeated
process measurements and an independent repeat of the complete experiment.
Compare core1/2/4 at matching CB depth, report trial medians and a one-output-tile
control, and separate observed host/simulator time from the T/ceil(T/P) workload
distribution factor. CB capacity scales with active cores; global input payload
remains unchanged. Preserve variability and slower configurations rather than
claiming monotonic scaling or physical-chip speedup.

A multicast evidence: [17](17-a-multicast-matmul.md), 200 fixture cases and 42 repeated
submissions plus 188 existing regressions pass. A sender uses actual multicast data
and valid signaling after receiver-ready atomics. Tail receivers consume A to preserve
matching CB ring offsets. At 128×128×128 on four active cores the DRAM read payload
model falls 256→160KiB (37.5%); receiver L1 delivery and handshake traffic are added.
B remains locally read. No stable multicast timing/hardware-speedup claim is made.

2D multicast evidence: [18](18-2d-multicast-matmul.md), 240 fixture cases and
98 repeated/fallback dispatches plus 188 regressions pass. Four semaphore channels
(two ready/valid pairs) separate row-A and column-B sharing. Invalid edge sources
push dummy CB slots without DMA; idle outputs consume both operands. At 128³,
DRAM read payload model is 256/160/128KiB for independent/A-only/2D while
receiver payload grows from 96 to 128KiB versus A-only. Distinguish DRAM savings
from additional NoC/handshake traffic and unmeasured hardware speedup.

Multicast timing evidence: [19](19-multicast-simulator-performance.md), 32 configurations
and 96 independent process trials, 672 timed samples / 864 submissions. Match four
active cores and depth, retain independent1 controls and all slow results. At 128³,
2D halves the modeled DRAM reads but was slower in measured simulator/host time.
The busiest sender still reads 32 tiles while multicast responsibilities increase.
Reader NoC and output ownership differ, so this is not an isolated transport ablation.
Use negative results, sender load models and trial variation to motivate further work.


## GitHub 공개 준비 — 2026-11-09

공개 저장소는 [SangjunRyu/accelflow](https://github.com/SangjunRyu/accelflow)입니다. 기존 프로젝트명과 문서·스크립트 경로를 유지하면서,
설명에는 `NPU kernel and runtime optimization with TT-Metal/TTSim`을 사용하면 됩니다.
CUDA MatMul naive/tiled 구현과 MX150 실측은 [21번](21-cuda-matmul-comparison.md)에서 완료했습니다. 실제 GPU 시간과 NPU simulator 비용을 구분해 소개합니다.

버전 관리 기준:

| 포함 | 제외 |
| --- | --- |
| 자체 host/device 소스, CMake, 실행·검증 스크립트 | build, Python/의존성 cache, generated runtime 상태 |
| 번호 문서, 실험 metadata·summary·측정 JSON·검증 로그 | 외부 TT-Metal clone, TTSim 공유 라이브러리, 다운로드·데이터셋 |
| 컴파일 관찰용 텍스트 IR·assembly·symbol 보고서 | 재생성 가능한 ELF·object·실행 파일·공유 라이브러리 |

`results/` 전체 또는 `*.log`를 무시하지 않습니다. 현재 결과는 약 32MiB이며 문서에서
참조하는 정확성·재현성의 근거입니다. 초기 커밋에는 텍스트 결과를 보존하고, 향후 반복 실험이
커지면 문서가 참조하는 대표 실행을 남기고 추가 raw 결과는 별도 보관하는 방식을 검토합니다.
ELF 등 바이너리는 제외하므로 metadata에 기록된 모든 원본 파일이 GitHub에 올라가는 것은 아닙니다.
생성 시점의 SHA256과 원본 경로는 출처 기록이며 `scripts/inspect_compilation.sh`로 다시 관찰합니다.
환경 보고서에는 로컬 사용자명·호스트명·경로가 포함될 수 있으므로 공개 전 노출 범위를 확인합니다.
현재 LICENSE는 소유자의 선택을 기다리는 placeholder입니다. 공개 저장소와 오픈소스 사용 허가는
구분하며, 프로젝트 라이선스 선택과 외부에서 복사한 관찰 자료의 고지는 공개 전에 확인합니다.

지원서에 사용할 수 있는 최신 사례:

> TT-Metal/TTSim 가상 Wormhole에서 타일 기반 MatMul을 구현하고, 데이터 재사용·K streaming·
> multicore·2D multicast로 확장했습니다. K 배치 전송으로 신호 교환 횟수 모델을 70~80% 줄였고,
> 320/640 크기의 host/simulator 실행 비용을 기존 2D 대비 약 36% 줄였습니다. 배치 구현의
> 정확성 200건과 기존 회귀 188건을 검증했으며, 실제 NPU 칩 성능과 시뮬레이터 결과를 구분했습니다.

수치의 조건과 제한은 [20번 결과](20-batched-2d-multicast-matmul.md)를 함께 제시합니다.


## CUDA MatMul — 2026-11-09

NPU가 사용한 BF16 입력을 그대로 내보내 CUDA FP32 입력값으로 사용했습니다.
MX150에서 naive/shared-memory tiled 커널을 실행했고, 30개 process의 출력 1,380건과
기존 NPU 회귀 188개를 검증했습니다. 640×320×640의 event batch 평균 대표값은
7.62204→2.31803ms로 약 3.29배 차이가 났습니다. 128³에서는 tiled가 더 짧지 않았습니다.
실험 조건과 작은 입력의 결과, trial 변동을 [21번](21-cuda-matmul-comparison.md)에 함께 남겼습니다.
이 비율은 같은 실제 GPU의 두 CUDA 구현 사이 값이며 GPU/NPU 속도 비율이 아닙니다.


## NPU Reduction — 2026-11-09

행별 합을 SFPU scan과 타일 간 carry로 구현했습니다.
최초 native reduce 경로는 4096+1-4096 상쇄에서 실패했고, dtype/fidelity 변경만으로 해결되지 않았습니다.
실패 로그를 남기고 경로를 바꾼 뒤 같은 상쇄를 포함한 50개 구성과 152회 제출을 통과했습니다.
정수는 exact, 소수 최대 절대 오차는 4.768371582e-07입니다.
CB는 depth에 따라 6/12KiB이며 폭과 무관합니다. 출력에는 마지막 prefix 타일 전체를 저장하므로
compact vector보다 전송량이 큽니다. 이것은 정확성 기준 구현이며 물리 칩 성능 개선을 주장하지 않습니다.
자세한 조건과 실패·성공 기록은 [22번](22-row-wise-reduction.md)을 참고합니다.


## cudaArray와 profiler 진단 — 2026-11-09

같은 16×16 shared-memory/FMA 흐름에서 입력을 선형 pointer와 cudaArray point texture로 바꿔 비교했습니다.
45개 process·출력 2,070건을 검증했습니다. 큰 입력에서 texture kernel 대표 시간은 약 4.6% 짧았지만,
작은 입력은 개선되지 않았고 자원 생성·업로드 비용도 있었습니다. 내부 swizzle만의 효과라고 해석하지 않습니다.
nvprof trace는 성공했으며 legacy Event API는 CUPTI_ERROR_NOT_SUPPORTED(27)로 실패했습니다.
같은 오류의 580 driver 호환성 보고를 찾았지만 Windows MX150에서 driver 변경 검증은 하지 않았습니다.
driver 변경 없이 확보한 근거와 한계는 [23번](23-texture-matmul-and-profiler.md)에 정리했습니다.
