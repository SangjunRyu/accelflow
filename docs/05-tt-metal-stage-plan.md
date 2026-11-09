# 05. TT-Metal 조사·실행 전 계획 — Phase 0-D / 0-E

상태: 실행 전 계획 기록. **실제 준비·빌드·예제 검증은 [06번](06-tt-metal-build-validation.md)에서 완료**했습니다.
[진행 순서](README.md) · 이전: [04. TTSim 설치](04-ttsim-installation.md)
아래 예산과 명령은 당시 계획이며, 최종 실행 설정과 실제 사용량은 06번을 참고하세요.

2026-10-06의 실행 전 조사 기록입니다. 당시 미실행 상태의 계획을 아래에 보존합니다.

## 고정 버전과 호환성 근거

GitHub API의 현재 최신 릴리스는 [TT-Metal v0.79.0](https://github.com/tenstorrent/tt-metal/releases/tag/v0.79.0)입니다.
기존 v0.78.0 후보를 다음 후보로 갱신합니다.

| 구성요소 | 고정 revision |
| --- | --- |
| TTSim v1.11.1 | `18faa8c28c3b1849b833545d41d6b80c23d39c08` |
| TT-Metal v0.79.0 | `de546d3b146758714d900f11b218c8f9c805f410` |
| 고정 UMD submodule | `f9173df20bec9bb93e5c92407e701e604ddef33d` |
| Tracy submodule | `9c3ca7e5fe48cecf90aa518c82552aae706cf110` |
| cluster descriptors submodule | `ac8a3b5ba3a54ad4e93cb75da3692942e5926c50` |

고정 UMD의 [TTSimCommunicator](https://github.com/tenstorrent/tt-umd/blob/f9173df20bec9bb93e5c92407e701e604ddef33d/device/simulation/tt_sim_communicator.cpp)는
단일 칩 경로에서 init/exit, PCI config read, PCI memory read/write, tile read/write,
clock, DMA callbacks의 9개 심볼을 필수로 로딩합니다. 설치된 TTSim은 모두 export하며
호출 규약도 공식 ABI와 일치합니다. 다중 칩 전용 API는 이 경로의 요구조건이 아닙니다.
고정 태그에 Wormhole SOC descriptor와 정수 덧셈 target도 존재합니다.

이 조합은 **단일 Wormhole의 ABI 및 소스 요구사항이 일치하는 후보**입니다.
공식 문서에 이 두 release를 짝지은 보증표는 확인하지 못했습니다. UMD CI는 최신 Metal,
UMD, simulator 조합을 사용하므로 고정 release 조합의 성공 증거로 간주하지 않습니다.
최종 호환성 판정은 아래 예제 실행 후에만 합니다.

## Ubuntu 24.04 툴체인

[v0.79.0 build_metal.sh](https://github.com/tenstorrent/tt-metal/blob/v0.79.0/build_metal.sh)의
기본 host 툴체인은 **clang-20 / clang++-20 + libstdc++**입니다.
공식 [libstdcpp toolchain 파일](https://github.com/tenstorrent/tt-metal/blob/v0.79.0/cmake/x86_64-linux-clang-20-libstdcpp-toolchain.cmake)은
`ENABLE_LIBCXX=FALSE`이며, 가능한 경우 lld-20 또는 mold를 선택합니다.
Ubuntu용 [install_dependencies.sh](https://github.com/tenstorrent/tt-metal/blob/v0.79.0/install_dependencies.sh)는
g++-14와 libc++-20-dev / libc++abi-20-dev도 설치 목록에 포함합니다.

따라서 현재 GCC 13.3만으로 공식 기본 환경을 갖췄다고 보지 않습니다. 향후 Clang 20과
GCC 14/libstdc++ 개발 헤더를 준비합니다. libc++ 두 패키지는 전체 공식 의존성 목록에는
있지만 선택한 libstdc++ 툴체인의 C++ runtime으로 직접 사용되는 것은 아닙니다.
이번 조사에서는 모두 포함한 비용을 계산했으며 설치하지 않았습니다.

로컬 apt에서 g++-14, clang-20, libc++-20-dev, libc++abi-20-dev 후보가 Ubuntu
noble-updates에 존재합니다. 따라서 현재는 이를 위해 새로운 LLVM 저장소/GCC PPA를
추가할 필요가 없습니다. upstream 전체 설치 스크립트는 저장소·compiler 기본값·CMake
등을 변경하므로 그대로 실행하지 않는 계획입니다.

최상위 CMake minimum은 3.24이며 범위 끝 4.2는 policy compatibility 선언입니다.
현재 Linux CMake 3.28.3은 이 최소 요구를 충족합니다. upstream bootstrap의 4.0.2를
무조건 덮어 설치하지 않습니다. 하위 의존성의 configure 결과는 아직 검증되지 않았습니다.
Ninja 1.11.1은 설치되어 있습니다. kernel/JIT용 **SFPI 7.78.0**도 고정 태그가 요구합니다.
이는 host Clang과 별도이며, 기본 source configure 경로의 prebuilt 다운로드를 사용합니다.
`--use-system-sfpi`는 설치된 SFPI가 없어 사용하지 않습니다.

## 실제 자원과 추가 공간 계산

- C: 실제 가용 **35,717,005,312 bytes = 33.26GiB**. VHDX는 계속 C:에 있습니다.
- WSL RAM 4,057,583,616 bytes, 가용 약 **3.23GiB**, swap **1GiB**.
- Windows 물리 RAM **7.89GiB**, 조회 순간 가용 **약 0.83GiB**.
- [Windows 조회 원본](../results/setup/host-resources-phase0c-2026-10-06.json).

고정 Git tree에서 blob 크기를 합산했습니다. TT-Metal 405,201,675 bytes,
UMD 4,416,834 bytes, Tracy 24,578,427 bytes, cluster descriptors 4,292,448 bytes입니다.
합계 **438,489,384 bytes (약 418MiB)**는 압축 해제된 추적 파일의 논리 크기입니다.
git pack/history, filesystem blocks, CPM, SFPI, 빌드 산출물은 포함하지 않습니다.

apt `--assume-no --no-install-recommends`로 추가 도구들을 계산한 결과:
**112MB 다운로드 / 548MB 추가 설치**, 28개 신규 패키지, 제거 없음.
이 조회는 root로 실행했지만 마지막 설치 질문에 N을 선택해 설치하지 않았습니다.
[apt 계산 로그](../results/setup/tt-metal-toolchain-proposal-2026-10-06.log).

| 공간 항목 | 계획상 추가 예산 |
| --- | --- |
| shallow source/submodules + .git/filesystem 여유 | 1–2GiB |
| host toolchain + 다운로드 cache/향후 보완 | 2–3GiB (실제 apt 계산은 위의 548MB) |
| SFPI + CMake fetched dependencies | 2–4GiB |
| object/PCH/library + kernel/runtime cache | 10–25GiB |
| 합계 | **15–34GiB** |
| Windows 잔여 공간 reserve | **10GiB** |
| 보수적 시작 기준 | **44GiB free** |

blob/apt 수치는 조사값이고 나머지는 **사전 계획 여유분**입니다. 아직 빌드하지 않아
정확한 산출물 크기를 알 수 없습니다. 44GiB는 upstream 최소 요구가 아니라 이 작업의
보수적 기준이며, targeted build가 실제로 더 적게 사용할 가능성도 있습니다.
현재 33.26GiB는 이 기준에 못 미치므로 **clone/configure/build 모두 이번에는 보류**합니다.
Windows 가용 메모리도 적어 향후 사용 중인 앱을 정리하고 재확인해야 합니다.
메모리/swap/.wslconfig는 변경하지 않았습니다.
[계산 JSON](../results/setup/tt-metal-resource-budget-2026-10-06.json).

## 자원 확인 통과 후 제안하는 정확한 명령

먼저 C: free와 Windows/WSL memory를 다시 측정합니다. 아래는 실행 전 계획입니다.
패키지 후보는 기존 공식 Ubuntu 저장소에서 확인했으며, WSL 내에서 실행합니다.

```bash
sudo apt-get update
sudo apt-get --assume-no --no-install-recommends install \
  clang-20 lld-20 g++-14 libc++-20-dev libc++abi-20-dev \
  libatomic1 libstdc++6 xz-utils pandoc python3-pip python3-pkg-resources xxd
# 위 출력과 호스트 여유 공간을 검토한 후에만:
sudo apt-get --no-install-recommends install \
  clang-20 lld-20 g++-14 libc++-20-dev libc++abi-20-dev \
  libatomic1 libstdc++6 xz-utils pandoc python3-pip python3-pkg-resources xxd

mkdir -p "$HOME/dev/tenstorrent"
# 새 경로에만 clone; 기존 checkout은 덮어쓰지 않음
git clone --branch v0.79.0 --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/tenstorrent/tt-metal.git "$HOME/dev/tenstorrent/tt-metal-v0.79.0"
cd "$HOME/dev/tenstorrent/tt-metal-v0.79.0"
test "$(git rev-parse HEAD)" = de546d3b146758714d900f11b218c8f9c805f410
git submodule status --recursive
du -sh .
df -h /mnt/c
free -h
```

다시 공간을 확인한 뒤 configure와 target 빌드를 분리합니다. 이 configure도 SFPI/CPM
다운로드와 runtime 파일 생성을 수행할 수 있으므로 자원 확인 전에 실행하지 않습니다.

```bash
./build_metal.sh --configure-only --build-programming-examples \
  --without-distributed --without-python-bindings --disable-profiler
# configure 후 실제 공간을 다시 확인하고, 예제와 필요한 라이브러리만 한 job으로 빌드:
cmake --build build --target metal_example_add_2_integers_in_riscv --parallel 1
```

공식 wrapper의 전체 install target을 즉시 빌드하는 대신, 소스에서 확인한 target을
선택합니다. 이는 CMake의 target 선택 방식이며 별도 simulator compile flag가 아닙니다.
TT-NN 모델/전체 tests/다중 칩/MPI/Python bindings는 초기 smoke 범위에서 제외합니다.
configure가 추가 의존성을 요구하면 실제 오류를 바탕으로 계획을 갱신합니다.

## 빌드 성공 후의 공식 smoke test

[TTSim 실행 문서](https://github.com/tenstorrent/ttsim/blob/v1.11.1/README.md#running-with-tt-metalium)의
환경변수와 SOC descriptor 경로를 사용합니다. 로그에는 command/exit code/output을 남깁니다.

```bash
set -euo pipefail
export TT_METAL_HOME="$HOME/dev/tenstorrent/tt-metal-v0.79.0"
export TT_METAL_SIMULATOR="$HOME/dev/tenstorrent/sim/libttsim_wh.so"
cp "$TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml" \
  "$HOME/dev/tenstorrent/sim/soc_descriptor.yaml"
log_dir="$HOME/dev/accelflow/results/setup"
log_file="$log_dir/metal-smoke-$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$log_dir"
cd "$TT_METAL_HOME"
printf '%s\n' 'TT_METAL_SLOW_DISPATCH_MODE=1 ./build/programming_examples/metal_example_add_2_integers_in_riscv' > "$log_file"
set +e
TT_METAL_SLOW_DISPATCH_MODE=1 \
  ./build/programming_examples/metal_example_add_2_integers_in_riscv 2>&1 | tee -a "$log_file"
status=${PIPESTATUS[0]}
set -e
printf 'example_exit_code=%s\n' "$status" | tee -a "$log_file"
test "$status" -eq 0
grep -F 'Success: Result is 21' "$log_file"
```

고정 예제 소스는 결과 길이 1과 정수 값 21을 직접 검사하며, 실패 시 -1을 반환합니다.
따라서 종료 코드 0과 성공 문자열을 모두 확인합니다. 현재 이 smoke test는 미실행입니다.
TTSim은 functional/structure 검증에 사용하며 simulator runtime은 실제 NPU latency가 아닙니다.
