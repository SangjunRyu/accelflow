# 04. TTSim Wormhole 설치·독립 검증 — Phase 0-C

상태: **완료**. 후속 TT-Metal 공식 커널 실행도 [06번](06-tt-metal-build-validation.md)에서 통과했습니다.
[진행 순서](README.md) · 이전: [03. 자원 보고서](03-resource-risk.md) · 다음: [05. TT-Metal 계획](05-tt-metal-stage-plan.md)

2026-10-06 공식 GitHub API와 고정 태그 문서를 확인하고 설치했습니다.

| 항목 | 결과 |
| --- | --- |
| 공식 릴리스 | [v1.11.1](https://github.com/tenstorrent/ttsim/releases/tag/v1.11.1), 조회 당시 latest |
| 태그 commit | `18faa8c28c3b1849b833545d41d6b80c23d39c08` |
| 다운로드 URL | `https://github.com/tenstorrent/ttsim/releases/download/v1.11.1/libttsim_wh.so` |
| 설치 위치 | `/home/ysjun5656/dev/tenstorrent/sim/libttsim_wh.so` |
| 파일 크기 | 227,544 bytes |
| SHA-256 | `196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a` |
| file | ELF 64-bit LSB shared object, x86-64, dynamically linked, stripped |
| ELF BuildID | `b62504ba40b55b9a1735eda0efe2a53f1621e3c9` |
| DT_NEEDED | `libc.so.6` 하나; libstdc++ / libgcc_s 의존성 없음 |
| ldd | libc와 `/lib64/ld-linux-x86-64.so.2` 해결됨, not found 없음 |
| 참조 GLIBC symbol versions | 2.2.5, 2.3.4, 2.4, 2.14; 로컬 glibc 2.39 |

GLIBC 심볼 버전은 이 바이너리의 관측값이며 모든 배포판 지원에 대한 보장이 아닙니다.
플랫폼 선택 근거는 [고정 태그의 API/ABI 문서](https://github.com/tenstorrent/ttsim/blob/v1.11.1/docs/libttsim_api.md)의
Ubuntu 22.04/24.04 및 x86_64 지원입니다.

## 실행한 검증

- 공식 release asset digest와 다운로드 SHA-256 일치
- ELF64 / little endian / x86_64 확인, dynamic section / program headers / symbol versions 기록
- `ldd` 의존성 확인 및 `RTLD_NOW`로 실제 즉시 심볼 바인딩 성공
- 공식 API의 `libttsim_*` export 10개 확인
- 독립 Python 프로세스에서 `libttsim_init()` 한 번, `libttsim_exit()` 한 번 성공
- PCI vendor/device 값 `0x401e1e52`로 Wormhole 확인
- 존재하지 않는 PCI function/device에서 `0xffffffff` 확인
- BAR 설정 읽기 및 cache-line-size 설정 레지스터 쓰기/읽기/원래 값 복원
- 펌웨어를 올리지 않은 idle 상태에서 `libttsim_clock(1)` 호출 성공
- 설치 스크립트 재실행 시 다운로드 없이 동일 라이브러리 재사용
- 잘못된 SHA-256을 주입한 실패 경로에서 exit code 1로 거부, 기존 파일 보존 확인

호출 규약과 PCI 기대값은 [공식 API](https://github.com/tenstorrent/ttsim/blob/v1.11.1/docs/libttsim_api.md)와
[libttsim 구현](https://github.com/tenstorrent/ttsim/blob/v1.11.1/src/libttsim.cpp)을 확인했습니다.
초기화의 DRAM은 [sim.cpp](https://github.com/tenstorrent/ttsim/blob/v1.11.1/src/sim.cpp)의
lazy anonymous mmap입니다. 전체 DRAM을 읽거나 쓰지 않았습니다.
검증 프로세스 peak RSS는 146,188 KiB였습니다. 라이브러리가 출력한 시간/클럭 수치는
진단 로그일 뿐 실제 NPU 성능으로 사용하지 않습니다.

라이브러리는 CLI 실행 파일이 아니므로 `.so`를 직접 실행하지 않았습니다.
ABI 위반은 프로세스 전체를 종료할 수 있어 별도 프로세스와 30초 timeout을 사용했습니다.
DMA 작업, tile-relative access(Wormhole에서 비지원), 실제 커널 연산은 수행하지 않았습니다.

## 증거

- [최초 다운로드·검증 로그](../results/setup/ttsim-20261006T074119Z-Qvf87U/validation.log)
- [재실행 및 lifecycle 검증 로그](../results/setup/ttsim-20261006T074331Z-VzDFn2/validation.log)
- [최종 메타데이터](../results/setup/ttsim-20261006T074331Z-VzDFn2/metadata.json)
- [공식 릴리스 API 원본](../results/setup/upstream-2026-10-06/ttsim-release.json)
- [checksum 거부 테스트](../results/setup/ttsim-checksum-rejection-2026-10-06.log)
- [거부 후 실제 checksum](../results/setup/ttsim-checksum-after-rejection-2026-10-06.txt)
- [최종 환경 보고서](../results/setup/environment-after-ttsim-2026-10-06.txt)

설치 디렉터리에도 `accelflow-install.json`을 저장했습니다.

## 재현

WSL Ubuntu에서 프로젝트 루트로 이동한 뒤:

```bash
bash scripts/setup_ttsim.sh
```

기본값은 고정된 `v1.11.1`, `TTSIM_CHIP=wormhole`, `$HOME/dev/tenstorrent/sim`입니다.
선택 변수는 프로젝트 스크립트 변수이며 TT-Metal 공식 변수와 구분합니다.
다른 태그를 선택할 때는 공식 release의 `TTSIM_SHA256`을 함께 지정해야 합니다.
기존 바이너리 checksum이 다르면 덮어쓰지 않고 실패합니다. 다운로드는 임시 파일에서
검증한 후 이동하며, 매번 `results/setup/ttsim-<UTC>-<suffix>/`에 증거를 남깁니다.

TTSim 소스 빌드, 드라이버/firmware/tt-smi 설치, hugepages 설정, Windows GPU 설정 변경은
수행하지 않았습니다. SOC descriptor는 향후 선택한 TT-Metal checkout에서 복사합니다.

## 후속 검증

이 문서의 0-C 시점에는 TT-Metal을 clone/build하지 않았습니다.
이후 [06. 실제 빌드·검증](06-tt-metal-build-validation.md)에서 v0.79.0을 빌드하고,
공식 정수 덧셈 예제의 `Success: Result is 21` 및 종료 코드 0을 확인했습니다.
다른 연산자와 사용자 정의 커널의 검증은 다음 단계입니다.
