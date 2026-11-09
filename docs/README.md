# 00. 진행 순서와 문서 안내

마지막 갱신: 2026-10-19. **Phase 2 K 배치 2D multicast 구현·320/640 크기 비교 완료**입니다.
가상 Wormhole에서 결과 21 / exit code 0을 확인했습니다.

## 읽는 순서

파일 번호는 문서를 읽는 순서입니다. 실제 작업 단계는 `0-A`부터 `0-E`로 표시합니다.
각 문서의 상단에서 완료 상태와 이전/다음 문서를 확인할 수 있습니다.

| 순서 | 문서 | 작업 단계와 상태 |
| --- | --- | --- |
| 01 | [환경 점검](01-environment.md) | 0-A 완료; 후속 설치 결과도 누적 |
| 02 | [초기 설치 계획](02-installation-plan.md) | 최초 계획 작성 완료; 현재 명령은 05 참조 |
| 03 | [호스트 자원·위험 보고서](03-resource-risk.md) | 0-B 자원 검사·최소 의존성 설치 완료; 0-C 자원 재확인 |
| 04 | [TTSim 설치·검증](04-ttsim-installation.md) | 0-C 완료 |
| 05 | [TT-Metal 후속 계획](05-tt-metal-stage-plan.md) | 실행 전 조사·계획 기록 |
| 06 | [TT-Metal 실제 빌드·검증](06-tt-metal-build-validation.md) | 0-D/0-E 완료 |
| 07 | [WSL 작업 공간 이동](07-wsl-workspace-migration.md) | 프로젝트 경로를 Linux 파일시스템으로 이전 |
| 08 | [자체 Vector Add](08-vector-add.md) | Phase 1 구현·정확성 완료 |
| 09 | [설계·성능 구조 평가](09-vector-add-design-assessment.md) | 동일 입력 CB 실험·정렬·구조 평가 완료 |
| 10 | [MatMul 구현·재사용 평가](10-matmul.md) | 첫 48개 검증·A 타일 행 재사용 평가 완료 |
| 11 | [Blocked MatMul](11-blocked-matmul.md) | 첫 84개 검증·A/B 재사용·L1 비용 비교 완료 |
| 12 | [K-streamed MatMul](12-k-streamed-matmul.md) | FP32 누적·K<=512·188개 검증 완료 |
| 13 | [TTSim 실행 비용 측정](13-matmul-simulator-performance.md) | 36개 구성·324회 dispatch 비교 완료 |
| 14 | [컴파일 관찰·TT-Explorer](14-compiler-inspection-and-tt-explorer.md) | Host IR·장치 ELF 5개 확인; Explorer 별도 단계 |
| 15 | [독립 multicore MatMul](15-independent-multicore-matmul.md) | 출력 분할·1/2/4코어·160개 검증 완료 |
| 16 | [Multicore TTSim 측정](16-multicore-simulator-performance.md) | 코어 수/depth 비교·3 trials·독립 재측정 완료 |
| 17 | [A multicast MatMul](17-a-multicast-matmul.md) | sender/receiver·200개 검증·42회 재실행 완료 |
| 18 | [2D A/B multicast](18-2d-multicast-matmul.md) | 행/열 공유·240개 검증·98회 반복/fallback 완료 |
| 19 | [Multicast 시간 비교](19-multicast-simulator-performance.md) | **32개 구성·96개 trials·느린 결과/변동 포함 완료** |
| 20 | [K 배치 2D multicast](20-batched-2d-multicast-matmul.md) | 배치1/4/8·200개 검증·320/640 반복 시간 비교 완료 |

## 실제 진행 현황

| 작업 순서 | 작업 | 상태 |
| --- | --- | --- |
| 1 | 프로젝트 구조 생성·WSL 환경 점검 | 완료 |
| 2 | 설치 계획 작성·VHDX 위치·물리 RAM·실제 여유 공간 확인 | 완료 |
| 3 | 최소 Ubuntu 빌드 의존성 설치 — CMake/Ninja 등 | 완료 |
| 4 | 공식 TTSim release 조회·Wormhole 바이너리 다운로드 | 완료 |
| 5 | SHA-256·ELF·동적 의존성·독립 API/lifecycle 검증 | 완료 |
| 6 | 재현 스크립트 저장·재사용·checksum 거부 확인 | 완료 |
| 7 | TT-Metal revision 후보·해당 revision 의존성·공간 예산 조사 | 완료 |
| 8 | 호스트/WSL 자원 재확인·예제 target으로 범위 제한·디스크 감시 | 완료 |
| 9 | 선택한 TT-Metal revision의 추가 toolchain 설치 | 완료 |
| 10 | TT-Metal clone/submodules/configure/target build | 완료 |
| 11 | SOC descriptor 복사·slow dispatch 공식 예제 실행 | 완료 |
| 12 | `Success: Result is 21`·exit code 0·전체 로그 확인 | 완료 |
| 13 | 프로젝트를 WSL Linux 파일시스템으로 이동 | 완료 |
| 14 | 자체 Vector Add·CPU 비교·CB 깊이 실험 | 완료 |
| 15 | BF16 MatMul baseline·A 행 재사용·48개 검증 | 완료 |
| 16 | Blocked MatMul·세 버전 비교·84개 검증 | 완료 |
| 17 | K streaming·FP32 누적·188개 검증 | 완료 |
| 18 | MatMul host/simulator 반복 실행 비용 측정 | 완료 |
| 19 | 현재 C++ 컴파일 산출물 관찰 환경 구성 | 완료 |
| 20 | 독립 multicore MatMul·출력 분할·기존 188개 회귀 검사 | 완료 |
| 21 | Multicore TTSim 시간·작업량·CB 비교 및 재현성 측정 | 완료 |
| 22 | A multicast sender/receiver·tail 동기화·기존 회귀 검사 | 완료 |
| 23 | 2D A/B multicast·행/열/모서리 경계·기존 회귀 검사 | 완료 |
| 24 | 독립/A-only/2D의 동일 4코어 TTSim 실행 비용 비교 | 완료 |
| 25 | K 배치 2D multicast·320/640 크기·회귀 및 시간 비교 | **완료** |

## TTSim 자체 설치 완료 결과

아래는 이미 저장된 설치·검증 증거의 요약입니다.

```yaml
TTSim version: v1.11.1
libttsim_wh.so: OK
install_path: /home/ysjun5656/dev/tenstorrent/sim/libttsim_wh.so
file_size_bytes: 227544
SHA256: 196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a
ELF 64-bit x86-64: OK
dynamic_loading: OK
standalone_init_PCI_exit: OK
idempotent_reuse: OK
TT_Metal_kernel_smoke_test: PASS
```

상세 증거와 재실행 방법은 [04번](04-ttsim-installation.md)을 참고하세요.
설치 완료 상태를 유지하며 재검증하려면 WSL의 프로젝트 루트에서
`bash scripts/setup_ttsim.sh`를 실행합니다. 정상 파일은 다시 다운로드하지 않습니다.

## 다음에 할 일

[10번](10-matmul.md)에서 BF16 MatMul baseline과 A 행 재사용을 구현하고 48개 케이스를 검증했습니다.
[11번](11-blocked-matmul.md)에서 2×2 blocked MatMul과 B 재사용까지 검증했습니다.
[12번](12-k-streamed-matmul.md)에서 K streaming과 FP32 누적, K<=512 검증을 완료했습니다.
[15번](15-independent-multicore-matmul.md)에서 독립 multicore의 1/2/4코어 출력 분할을 검증했습니다.
[17번](17-a-multicast-matmul.md)에서 A sender/receiver 공유를 구현했습니다.
[18번](18-2d-multicast-matmul.md)에서 A/B 2차원 multicast를 구현했습니다.
[19번](19-multicast-simulator-performance.md)에서 동일 4코어의 시간/변동/전송 모델을 비교했습니다.
[20번](20-batched-2d-multicast-matmul.md)에서 K 배치와 320/640 크기 비교를 완료했습니다.
다음 후보는 동일 매핑/NoC 대조, 출력 block 재사용과 multicast 결합, CUDA 실측 비교입니다.
현재 재사용 효과는 소스 기반 payload 모델이며 실물 성능 측정은 아닙니다.

준비·빌드·실행의 정확한 최종 설정과 로그는 [06번](06-tt-metal-build-validation.md)을 참고하세요.
TT-Metal v0.79.0 / TTSim v1.11.1 / Clang 20.1.2 / GCC-G++ 14.2가 준비됐으며,
06번 당시 빌드 후 C: 여유는 약 30.99GiB였으며 현재 여유 공간은 다음 빌드 전에 다시 확인해야 합니다. 시뮬레이터 실행 시간은 실제 NPU 성능으로 보고하지 않습니다.

## 참고 문서

90번 이후는 실행 순서가 아니라 참고 문서 번호입니다.

| 번호 | 문서 | 용도 |
| --- | --- | --- |
| 90 | [구조도](90-architecture.md) | 프로젝트 전체 구조 |
| 91 | [CUDA와 NPU 비교](91-cuda-vs-npu.md) | 프로그래밍 개념 비교 |
| 92 | [측정 정책](92-measurement-policy.md) | 실측과 simulator 검증 구분 |
| 93 | [문제 해결](93-troubleshooting.md) | 설정·빌드 문제 기록 |
| 94 | [포트폴리오 노트](94-portfolio-notes.md) | 후속 결과 정리 기준 |

원래 [프로젝트 명세](../AccelFlow_PROJECT.md)의 문서명에는 번호가 없지만,
현재 구현 문서는 이 목록의 번호가 붙은 파일을 사용합니다.
