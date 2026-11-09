# 02. 초기 설치 계획 — Phase 0

상태: 최초 계획 작성 완료; 현재 TT-Metal 실행 전 계획은 05번 참조.
[진행 순서](README.md) · 이전: [01. 환경 점검](01-environment.md) · 다음: [03. 자원 보고서](03-resource-risk.md)

0-C 완료: [TTSim 설치·독립 검증](04-ttsim-installation.md).
현재 공식 TT-Metal 최신 후보는 **v0.79.0**이며 기존 v0.78.0 후보를 대체합니다.
고정 revision, 최신 의존성 조사, 추가 공간 예산, 정확한 실행 전 명령은
[TT-Metal 후속 계획](05-tt-metal-stage-plan.md)에 정리했습니다. 아래는 최초 계획 기록이며
이전 경로/버전의 명령을 바로 실행하지 않습니다.

2026-10-06 업데이트: 사용자 승인 범위에 따라 호스트 자원 검사와 0-B 최소 의존성 설치를
완료했습니다. [자원 보고서](03-resource-risk.md), [설치 로그](../results/setup/dependency-install-2026-10-06.log).
Ubuntu CMake 3.28.3 / Ninja 1.11.1을 확인했습니다. 아래의 외부 clone/download/build 명령은
아직 실행하지 않았습니다. 자원 여유 확보와 재검토 전에는 TT-Metal을 빌드하지 않습니다.

## 목표와 후보 버전

WSL2 Ubuntu 24.04 / x86_64에서 단일 가상 Wormhole을 실행합니다.
외부 소스와 바이너리는 Linux 파일시스템의 `$HOME/dev/tenstorrent/` 아래에 둡니다.
프로젝트 경로는 현재 Windows 작업 폴더를 유지합니다.

- TTSim 후보: `v1.11.1`, x86_64 `libttsim_wh.so`.
  [공식 릴리스](https://github.com/tenstorrent/ttsim/releases/tag/v1.11.1)의 SHA-256:
  `196fb781bfc0bc37af2ec15f7f57a27199ba24cf7440643db84796bcbce46b1a`.
- TT-Metal 후보: `v0.78.0`, [공식 릴리스](https://github.com/tenstorrent/tt-metal/releases/tag/v0.78.0).
  해당 태그의 [빌드 스크립트](https://github.com/tenstorrent/tt-metal/blob/v0.78.0/build_metal.sh)를 확인했습니다.

두 후보의 조합을 실제 검증한 것은 아닙니다. 클론 후 태그의 전체 commit, submodule,
설치 문서, simulator 구현, SOC descriptor 및 예제 존재를 확인하고 조합을 고정합니다.
태그가 맞지 않으면 임의로 main으로 전환하지 않고 확인된 조합으로 계획을 갱신합니다.

## 먼저 해결할 자원 및 호환성 위험

| 관측 | 설치 전 대응 |
| --- | --- |
| WSL RAM 3.8GiB / swap 1GiB | 초기 빌드 병렬도를 1로 제한; 부족하면 할당 조정 계획을 별도로 검토 |
| C: 여유 약 13GiB | VHDX 실제 저장 볼륨과 여유 공간 확인; 현재 상태에서 대형 클론/빌드 시작 보류 |
| Linux df의 가용 931G | 가상 디스크 논리 용량이므로 호스트 여유 공간 대신 사용하지 않음 |
| WSL NAT 실패 경고 | Ubuntu에서 GitHub 및 패키지 저장소 DNS/HTTPS 접근을 먼저 점검 |
| Clang 없음; CMake 3.28.3 / Ninja 1.11.1 설치 완료 | 전체 태그·submodule 툴체인 요구를 향후 빌드 전에 확인 |
| Docker 서버 접근 실패 | 이번 계획은 native WSL 빌드; Docker 필수 조건 없음 |
| 사용자 지정 커널 문자열 | WSL 설정은 유지; 실제 최소 예제에서 기능 확인 |

## 설치 순서와 변경 범위

1. 물리 저장 공간, WSL 네트워크, apt 후보 버전을 읽기 전용으로 점검합니다.
2. 공식 저장소 `https://github.com/tenstorrent/tt-metal.git`의 지정 태그를
   외부 디렉터리에 클론하고 recursive submodules를 초기화합니다.
   [공식 source 설치 안내](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)는
   submodule 포함 클론과 `build_metal.sh` 사용을 설명합니다.
3. 고정 태그의 의존성 스크립트를 다시 읽고, 실제 패키지 목록과 추가 저장소를 표시합니다.
   현재 main의 [공식 의존성 소스](https://github.com/tenstorrent/tt-metal/blob/main/install_dependencies.sh)는
   Ubuntu 24.04용 g++-14, LLVM 20, CMake 4.0.2를 준비하며 LLVM 저장소와 GCC PPA를 추가합니다.
   따라서 일반 apt 패키지만으로 충분하다고 가정하지 않습니다. 이 스크립트 전체를
   검토 없이 실행하지 않습니다.
4. 일반 의존성 후보는 build-essential, git, ninja-build, pkg-config, curl, wget,
   python3-dev, python3-venv, libssl-dev, libhwloc-dev, libnuma-dev, libtbb-dev,
   libcapstone-dev입니다. 태그 기준으로 정확한 버전·추가 패키지를 확정합니다.
   Python 도구는 venv 안에 설치하고 시스템 Python을 교체하지 않습니다.
   CMake 사용자 설치 경로와 명시적인 compiler 경로를 우선 검토합니다.
5. TTSim 바이너리를 위 릴리스에서 내려받아 SHA-256을 대조합니다.
   기본 선택은 `TTSIM_CHIP=wormhole`입니다. 이는 앞으로 만들 프로젝트 스크립트의
   선택 변수이며 Tenstorrent 공식 환경변수라고 주장하지 않습니다.
6. 최소 C++ 예제를 포함해 TT-Metal을 빌드하고 로그를 남깁니다.
   후보 태그에서 확인한 옵션은 다음과 같습니다.

```bash
# TT-Metal checkout 안에서; 설치 승인 및 의존성 준비 후 실행
CMAKE_BUILD_PARALLEL_LEVEL=1 ./build_metal.sh \
  --build-programming-examples --without-distributed --without-python-bindings
```

Python bindings와 TT-NN 모델 의존성은 최소 예제 성공 후 추가합니다.
물리 Tenstorrent 장치용 TT-Installer, TT-KMD, 펌웨어 플래시, hugepages,
`/dev/tenstorrent` passthrough는 이 시뮬레이터 설치 경로에 포함하지 않습니다.

## 공식 최소 예제 검증

[공식 TTSim README](https://github.com/tenstorrent/ttsim#running-with-tt-metalium)에
명시된 환경변수·descriptor·예제 경로를 사용합니다.

```bash
# 예시 외부 경로; 해당 파일들이 실제 존재하는지 먼저 확인
export TT_METAL_HOME="$HOME/dev/tenstorrent/tt-metal"
export TT_METAL_SIMULATOR="$HOME/dev/tenstorrent/ttsim-v1.11.1/libttsim_wh.so"
cp "$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" \
  "$(dirname "$TT_METAL_SIMULATOR")/soc_descriptor.yaml"
cd "$TT_METAL_HOME"
TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build/programming_examples/metal_example_add_2_integers_in_riscv
```

검증 스크립트는 라이브러리 의존성, SOC descriptor, 실행 파일을 검사하고
종료 코드와 전체 출력을 `results/setup/`에 기록해야 합니다.
예제 소스의 실제 검증 방식도 확인합니다. 종료 코드 0만으로 아직 존재하지 않는
사용자 정의 vector-add의 정확성까지 주장하지 않습니다.

## 완료 조건

고정 버전/해시 기록, 재실행 가능한 설치 스크립트, 빌드 로그, 최소 예제의 검증된 출력,
환경 문서 갱신까지 완료되어야 Phase 0 bootstrap 완료입니다.
시뮬레이터 시간은 실제 NPU 성능으로 보고하지 않습니다.
