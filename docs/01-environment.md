# 01. 환경 점검 — Phase 0-A 및 후속 기록

최신 결과: [06. TT-Metal 실제 빌드·검증](06-tt-metal-build-validation.md) 완료.
Clang 20.1.2 / GCC-G++ 14.2 개발 의존성을 준비하고, TTSim v1.11.1과 TT-Metal v0.79.0의
공식 정수 덧셈 예제에서 결과 21 / exit code 0을 확인했습니다.
[최종 환경 보고서](../results/setup/environment-after-metal-build.txt).
이하 0-C/0-B 기록은 당시 상태를 보존합니다.

상태: 환경 점검 완료; 0-B/0-C 결과 누적.
[진행 순서](README.md) · 다음: [02. 초기 설치 계획](02-installation-plan.md)

## 0-C 설치 후 업데이트

TTSim v1.11.1 공식 prebuilt Wormhole을
`/home/ysjun5656/dev/tenstorrent/sim/libttsim_wh.so`에 설치했습니다.
227,544 bytes, 공식 SHA-256 일치. [설치·검증 보고서](04-ttsim-installation.md).
ELF/의존성/즉시 동적 로딩/API 심볼/초기화/PCI 식별/종료 검사와 재실행 재사용이 통과했습니다.
TT-Metal clone/build 및 정수 덧셈 예제는 아직 실행하지 않았습니다.

새로운 C: 가용 공간은 약 33.26GiB, WSL 가용 메모리는 약 3.23GiB입니다.
최신 TT-Metal 후보는 v0.79.0으로 갱신했습니다. [후속 설치 계획](05-tt-metal-stage-plan.md).
추가 toolchain 패키지는 용량 조회만 했으며 설치하지 않았습니다.
최초 보고서와 0-B 기록은 아래에 유지합니다.

## 0-B 설치 후 업데이트

최소 의존성 설치 및 검증 완료. 설치 종료 코드 0, `dpkg --audit` 문제 없음.
[설치 후 환경 보고서](../results/setup/environment-after-deps-2026-10-06.txt),
[설치 로그](../results/setup/dependency-install-2026-10-06.log),
[자원 위험 보고서](03-resource-risk.md).

- Linux CMake: `/usr/bin/cmake`, 3.28.3
- Linux Ninja: `/usr/bin/ninja`, 1.11.1
- pkg-config로 OpenSSL 3.0.13, hwloc 2.10.0, NUMA 2.0.18,
  TBB 2021.11.0, Capstone 4.0.2 확인
- GCC/G++ 13.3.0, Git 2.43.0, Python 3.12.3 유지
- 23개 신규 패키지와 6개 curl/OpenSSL 관련 업데이트 적용; 패키지 제거 없음
- Clang 20 및 추가 LLVM/GCC 저장소는 아직 설치하지 않음
- VHDX 실제 위치는 C:; 호스트 RAM 약 8GB, 설치 전 볼륨 여유 12.23GiB
- TTSim 바이너리, TT-Metal clone/build, 물리 장치 구성요소는 미진행

아래는 변경 전 최초 점검 기록입니다.

실행: `scripts/check_env.sh`, Ubuntu 내 종료 코드 0. `bash -n` 문법 검사 통과.
원본: [환경 보고서](../results/setup/environment-2026-10-06.txt).

| 항목 | 실제 확인 결과 |
| --- | --- |
| 배포판 | Ubuntu-24.04, Ubuntu 24.04.5 LTS |
| WSL | Windows 배포판 목록에서 VERSION 2 확인 |
| 커널 | `5.15.90.1sangjunryu20195834+` |
| CPU | x86_64, Intel Core i7-8565U, 논리 CPU 8개 |
| RAM / swap | 총 3.8GiB / 1.0GiB, 측정 당시 가용 RAM 3.3GiB |
| C: 디스크 | 953G 중 약 13G 가용, 사용률 99% |
| Linux 루트 | ext4 `/dev/sdc`, 논리 용량 1007G, 가용 931G |
| Python | 3.12.3; python3-dev와 python3-venv 패키지 존재 |
| GCC / G++ | 13.3.0 |
| Clang / Clang 20 | 발견되지 않음 |
| CMake / Ninja | 발견되지 않음 |
| Git / glibc | 2.43.0 / 2.39 |
| Docker | 클라이언트 29.0.1 발견, 서버 연결 실패 |
| GPU | NVIDIA MX150 2GiB, WSL nvidia-smi에서 확인 |
| CUDA | nvcc 미발견; nvidia-smi의 CUDA 13.0 표시는 툴킷 설치 증거가 아님 |
| TT-Metal / TTSim | 환경변수 미설정; 설치 여부를 전체 홈 검색으로 단정하지 않음 |

## 설치 후 채울 항목

- TTSim tag / 전체 commit / 라이브러리 SHA-256: 0-C에서 검증, 위 보고서 참조
- TT-Metal tag / 전체 commit / submodule revisions: 미검증
- 실제 사용 compiler / CMake / Ninja: 미정
- 빌드 옵션 및 로그: 미실행
- Wormhole 공식 예제 종료 코드 / 출력 검증: 미실행

## 알려진 문제와 적용한 조치

WSL 실행은 샌드박스 내에서 `E_ACCESSDENIED`로 실패했으며 허용된 외부 실행으로
읽기 전용 점검을 완료했습니다. 첫 시작 시 NAT 설정 실패 후 `VirtioProxy`로
전환하는 경고가 출력되었습니다. 인터넷 연결 성공을 검증한 것은 아닙니다.
이 경고는 WSL 시작 stderr에 있었으므로 Bash 보고서에는 포함되지 않았습니다.

Linux 가상 디스크의 가용 용량은 Windows 호스트의 실제 여유 공간과 다릅니다.
최초 점검에서는 VHDX 저장 볼륨을 미확인했습니다. 0-B에서 C:임을 확인했습니다.
최초 점검 이후 최소 패키지만 설치했으며, 드라이버 설치·WSL 설정 변경·CUDA 설정 변경은
수행하지 않았습니다.
