# AccelFlow

CUDA에서 익힌 메모리 이동, 타일링, 버퍼링, 동기화 개념을 TT-Metalium / TT-NN과
TTSim으로 탐구하는 가속기 시스템 포트폴리오 프로젝트입니다.
전체 명세는 [AccelFlow_PROJECT.md](AccelFlow_PROJECT.md)를 참고하세요.

타일 기반 MatMul에서 시작해 L1 데이터 재사용, K streaming, multicore, A/B 2D multicast를 구현했습니다.
마지막 실험에서는 K 방향 타일을 묶어 semaphore 교환 횟수 모델을 70~80% 줄였습니다.
320×320×320과 640×320×640 입력에서 host/TTSim 실행 비용은 기존 2D보다 약 36% 줄었지만,
독립 multicore보다는 여전히 길었습니다. 배치 구현의 정확성 200건과 기존 회귀 188건이 통과했습니다.
[실험 조건·수치·한계](docs/20-batched-2d-multicast-matmul.md)를 함께 기록했습니다.

## 구조

`CPU reference → CUDA / NVIDIA MX150` 또는 `TT-Metalium / TT-NN → TTSim / Wormhole`.
[구조도](docs/90-architecture.md), [측정 정책](docs/92-measurement-policy.md).
Vector Add의 자체 host/장치 커널 구현은 `npu/vector_add/`에 있습니다.
MatMul의 자체 host/장치 커널 구현은 `npu/matmul/`에 있습니다.
CUDA MatMul의 naive/shared-memory tiled 구현은 `cuda/matmul/`에 있습니다.
Reduction과 RMSNorm은 `npu/`, `cuda/` 아래에 구현했으며 모델 디렉터리는 후속 단계용 골격입니다.

## 시작하기

PowerShell에서 Ubuntu로 진입합니다.

```powershell
wsl.exe -d Ubuntu-24.04
```

Ubuntu에서 읽기 전용 점검을 실행합니다. 실행 권한 설정 없이 Bash로 실행할 수 있습니다.

```bash
cd /home/ysjun5656/dev/accelflow
set -o pipefail
bash scripts/check_env.sh | tee "results/setup/environment-$(date -u +%Y%m%dT%H%M%SZ).txt"
```

## 현재 상태

프로젝트 작업 경로는 WSL `/home/ysjun5656/dev/accelflow`입니다.
Windows에서는 `\\wsl.localhost\Ubuntu-24.04\home\ysjun5656\dev\accelflow`로 열 수 있습니다.
[07. WSL 작업 공간 이동](docs/07-wsl-workspace-migration.md).

**문서 읽는 순서와 실제 작업 현황은 [00. 진행 순서 안내](docs/README.md)에서 확인하세요.**
`01 → 02 → 03 → 04 → 05(계획) → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18 → 19(multicast 시간 비교) → 20(K 배치 2D) → 21(CUDA 실측 비교) → 22(Reduction) → 23(cudaArray·profiler)` 순서로 정리했습니다.

- [x] 문서 31번 전체 확인 및 저장소 구조 생성
- [x] WSL2 Ubuntu 환경 점검, Bash 문법 검사, 실제 실행 및 보고서 저장
- [x] 공식 문서 조사 및 [설치 계획](docs/02-installation-plan.md) 작성
- [x] VHDX 실제 저장 위치·호스트 자원 확인 및 최소 빌드 의존성 설치
- [x] TTSim v1.11.1 Wormhole 바이너리 설치 및 독립 ABI/lifecycle 검증
- [x] TT-Metal v0.79.0 외부 소스와 고정 submodules 준비
- [x] TT-Metal 빌드 및 TTSim 공식 정수 덧셈 예제 검증 — 결과 21, exit code 0

[환경 요약](docs/01-environment.md), [문제 해결](docs/93-troubleshooting.md).
현재 **Phase 3 단일 코어 NPU row-wise sum 검증까지 완료**했습니다.
[06. 실제 빌드·검증 보고서](docs/06-tt-metal-build-validation.md)에 재현 방법과 전체 로그를 기록했습니다.
[08. 자체 Vector Add 보고서](docs/08-vector-add.md): 최초 uint32 9개 / BF16 pipeline 12개 케이스 통과.
[09. 설계·성능 구조 평가](docs/09-vector-add-design-assessment.md): CB depth 1/2·65타일·버퍼 정렬을 포함한 총 38개 후속 검증 통과.
[10. MatMul 구현·평가](docs/10-matmul.md): baseline/A 행 재사용 × CB depth 1/2, 총 48개 케이스 통과.
[11. Blocked MatMul](docs/11-blocked-matmul.md): 세 버전 × depth 1/2, 총 84개 검증 통과.
[12. K-streamed MatMul](docs/12-k-streamed-matmul.md): FP32 누적·chunk1/2·K<=512, 총 188개 검증 통과.
[13. TTSim 실행 비용 측정](docs/13-matmul-simulator-performance.md): 36개 구성, 324회 dispatch, raw samples/통계 비교 완료.
[15. 독립 multicore MatMul](docs/15-independent-multicore-matmul.md): baseline 비교·1/2/4코어·160개 검증 통과.
[16. Multicore TTSim 측정](docs/16-multicore-simulator-performance.md): 32개 구성·3 process trials·재측정, 시간/변동/CB 비교 완료.
[17. A multicast MatMul](docs/17-a-multicast-matmul.md): 200개 검증·42회 재실행·기존 188개 회귀 통과.
[18. 2D A/B multicast](docs/18-2d-multicast-matmul.md): 240개 검증·98회 반복/fallback·기존 188개 회귀 통과.
[19. Multicast 시간 비교](docs/19-multicast-simulator-performance.md): 32개 구성·96개 process trials·672개 timing/864회 dispatch 완료.
CUDA MatMul 두 구현을 같은 입력으로 실행하고 NPU 결과와 대조했습니다.
[21. CUDA 실측 비교](docs/21-cuda-matmul-comparison.md). NPU row-wise sum은 [22번](docs/22-row-wise-reduction.md)에서 구현했습니다. CUDA Reduction 3개 구현과 정확성·반복 측정은 [24번](docs/24-cuda-reduction.md)에서 완료했습니다. RMSNorm의 NPU 기준 구현·CUDA 기본/fused 비교는 [25번](docs/25-rmsnorm.md)에서 완료했습니다. TT-NN 전체는 아직 검증하지 않았습니다.

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_vector_add.sh
bash scripts/run_matmul.sh
```
라이선스는 소유자 선택 전까지 미정입니다. `LICENSE`의 현재 안내를 참고하세요.

## 측정 제한과 로드맵

TTSim 실행 시간은 실제 NPU 성능이나 BOS Eagle-N 지연 시간을 나타내지 않습니다.
CUDA 실측과 시뮬레이터 기능 검증을 구분합니다.

환경 재현 → vector add → tiled MatMul → reduction → RMSNorm → 소형 모델 분석.
PyTorch CPU 기준 대조까지 완료했습니다. NPU 단일 코어 fusion까지 검증했습니다. 행 분할 multicore RMSNorm도 검증했습니다. TT-NN 기준 대조와 모델 실행은 후속 단계입니다. CUDA는 MatMul·row-wise reduction·RMSNorm 구현 및 MX150 실측을 완료했습니다.

컴파일 학습: [14. 관찰 환경과 TT-Explorer](docs/14-compiler-inspection-and-tt-explorer.md).
`bash scripts/inspect_compilation.sh`로 host IR·장치 ELF/assembly를 확인합니다.

K 배치 2D multicast와 320/640 크기 비교 결과는 [20번 문서](docs/20-batched-2d-multicast-matmul.md)를 참고하세요.

CUDA 실행 방법과 측정 조건은 [21번 문서](docs/21-cuda-matmul-comparison.md)에 정리했습니다.

NPU Reduction은 [22번 문서](docs/22-row-wise-reduction.md)를 참고하고 `bash scripts/run_reduction.sh`로 검증합니다.

CUDA의 선형 입력과 cudaArray/texture 입력 비교, nvprof 카운터 진단은 [23번](docs/23-texture-matmul-and-profiler.md)에 정리했습니다.

CUDA Reduction의 naive/tree/shuffle 구현, NPU 입력 대조와 큰 입력 재측정은 [24번](docs/24-cuda-reduction.md)에 정리했습니다.

RMSNorm 공통 입력 20종 대조, NPU 정밀도 전달과 CUDA fusion 비교는 [25번](docs/25-rmsnorm.md)을 참고하세요.

NPU RMSNorm의 중간 DRAM 제거, L1 보관과 폭 제한은 [26번](docs/26-npu-rmsnorm-fusion.md)에 정리했습니다.

RMSNorm의 1·2·4코어 행 분할, 꼬리/유휴 코어 처리와 TTSim 실행 기록은 [27번](docs/27-multicore-rmsnorm.md)에 정리했습니다.
