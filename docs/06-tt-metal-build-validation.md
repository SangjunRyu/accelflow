# 06. TT-Metal 빌드·공식 예제 검증 — Phase 0-D / 0-E

상태: **완료**. 2026-10-06 17:57 KST, 가상 Wormhole에서 공식 예제 실행과 결과 검증에 성공했습니다.
[진행 순서](README.md) · 이전: [05. 실행 전 계획](05-tt-metal-stage-plan.md)

## 이번 실행 범위

```yaml
TT_Metal: v0.79.0
TTSim: v1.11.1
device: virtual Wormhole
example: metal_example_add_2_integers_in_riscv
output: "Success: Result is 21"
example_exit_code: 0
log_exit_code: 0
result: PASS
```

- TT-Metal v0.79.0, commit `de546d3b146758714d900f11b218c8f9c805f410`
- 외부 checkout: `/home/ysjun5656/dev/tenstorrent/tt-metal-v0.79.0`
- 공식 Clang 20.1.2 / GCC-G++ 14.2 및 개발 의존성 설치 완료
- 고정 submodules 포함 shallow checkout 완료, 초기 약 699MiB
- TTSim v1.11.1 공식 Wormhole 바이너리 재사용
- 정수 덧셈 target만 빌드; PCH/profiler/Python bindings/MPI 비활성화
- 최종 설정은 공식 Unity build ON / 두 job: 묶음별 실측 메모리 여유 확인 후 병렬도 확정

초기 C: 여유는 약 34.59GiB, WSL 가용 메모리는 약 3.2GiB였습니다.
05번의 44GiB는 전체 빌드용 보수적 사전 예산이며 실측 최소 요구는 아닙니다.
이번에는 범위를 예제 target으로 제한하고 실제 공간을 감시해 진행합니다.
`guarded_command.py`는 C: 여유가 10GiB 아래로 내려가면 configure/build 프로세스 그룹을 중단합니다.
WSL 메모리/swap 설정과 Windows CUDA 설정은 변경하지 않습니다.

## 재현 스크립트

WSL 프로젝트 루트에서 순서대로 실행합니다.

```bash
bash scripts/setup_tt_metal.sh
bash scripts/build_tt_metal.sh
bash scripts/verify_ttsim.sh
```

toolchain은 먼저 `05` 문서의 추가 패키지 목록으로 준비합니다. 선택한 executable target은
`metal_example_add_2_integers_in_riscv`입니다. 최종 성공 조건은 종료 코드 0과
`Success: Result is 21` 문자열입니다.

## 실행 증거

- [빌드 전 호스트 자원](../results/setup/host-resources-before-build.json)
- [툴체인 설치 로그](../results/setup/tt-metal-toolchain-install.log)
- [clone/submodule 로그](../results/setup/tt-metal-clone.log)
- [configure 로그](../results/setup/tt-metal-configure.log)
- [빌드 로그](../results/setup/tt-metal-build.log)
- [공식 예제 전체 로그](../results/setup/metal-smoke-20261006T085752Z.log)
- [검증 메타데이터](../results/setup/tt-metal-bootstrap-metadata.json)
- [빌드 후 환경](../results/setup/environment-after-metal-build.txt)
- [빌드 후 Windows 자원](../results/setup/host-resources-after-build.json)

## 실제 결과와 자원

configure와 최종 target build가 종료 코드 0으로 완료됐습니다. 원본 commit과 고정
submodule을 사용했고, 예제 소스나 simulator를 수정하지 않았습니다.
소스·의존성·빌드 산출물·첫 JIT cache를 포함한 checkout은 약 **3.3GiB**입니다.
빌드 후 C: free는 33,276,309,504 bytes, 약 **30.99GiB**입니다.
전체 workflow 전후의 host free 차이는 약 3.6GiB이며, Windows의 다른 사용량 변화도
포함될 수 있어 TT-Metal 단독 정확한 사용량으로 해석하지 않습니다.

최종 설정: Release / Clang 20.1.2 / libstdc++ / CMake 3.28.3 / Ninja 1.11.1 /
SFPI 7.78.0 / Unity ON / PCH OFF / 두 job.
초기 보수적 설정에서 실측 후 빌드 묶음·병렬도를 조정했습니다. 로그의
`interrupted by user`는 이 설정 전환을 위한 의도적 중단이며 컴파일 오류가 아닙니다.
최종 빌드와 예제는 모두 성공했습니다. 반복 실행을 위한 현재 스크립트는 최종 설정을 사용합니다.

실행에서는 공식 SOC descriptor를 `.so` 옆의 `soc_descriptor.yaml`로 복사하고,
`TT_METAL_SIMULATOR`와 `TT_METAL_SLOW_DISPATCH_MODE=1`을 사용했습니다.
로그에서 Simulation device 생성 및 PCI `vendor_id=0x1e52 device_id=0x401e`를 확인했습니다.
host의 실제 결과 검사는 길이 1과 정수 21을 비교하고, 검증 스크립트는 종료 코드 0과
성공 문자열을 다시 확인했습니다. 따라서 이 고정 조합의 **최소 공식 예제 연동**은 검증됐습니다.

## 비치명적 안내와 검증 범위

- Python Inspector 자동완성용 pycapnp가 없어 stub 생성 안내가 나왔습니다.
  이번 C++ 예제에는 필요하지 않아 추가 설치하지 않았습니다.
- 실행 시 기본 n150 board 정보의 harvesting 기대값과 virtual SOC mask가 다르다는
  warning이 출력됐습니다. 공식 descriptor를 유지했으며 최소 예제는 정상 통과했습니다.
- DPRINT core 설정 안내가 출력됐습니다. 커널 디버그 출력은 이번 성공 조건에 포함하지 않았습니다.

이 결과는 모든 TT-NN 연산자·다중 칩·최적화·실물 성능을 검증한 것은 아닙니다.
simulator가 로그에 출력한 시간/클럭은 진단 정보이며 실제 NPU 성능으로 보고하지 않습니다.
KMD/tt-smi/실물 firmware 설치·hugepages 설정·WSL 설정 변경은 수행하지 않았습니다.
kernel/JIT용 SFPI와 virtual device 실행에 필요한 runtime은 공식 빌드가 준비했습니다.

다음 단계는 **Phase 1: Vector Add**입니다. 이번 작업에서는 사용자 정의 커널로 확장하지 않았습니다.
