# 93. 문제 해결 — 참고 문서

[진행 순서와 문서 안내](README.md)

## 실제 예제 빌드·실행에서 확인한 항목

공식 Unity build ON / PCH OFF / 두 job으로 정수 덧셈 target을 성공적으로 빌드했습니다.
현재 Clang 20.1.2 / GCC-G++ 14.2가 설치되어 있으며, 아래 미설치 언급은 이전 기록입니다.
로그의 `interrupted by user`는 설정 전환을 위한 의도적 중단입니다.

실행에서 n150 harvesting 정보와 virtual SOC mask의 불일치 warning 및 DPRINT core 설정
안내가 나왔지만, 공식 descriptor를 사용한 결과 21 / exit code 0 검증은 통과했습니다.
Inspector Python stub용 pycapnp는 C++ smoke test에 필요하지 않았습니다.
상세 로그와 검증 범위는 [06번](06-tt-metal-build-validation.md)을 참고하세요.

## WSL E_ACCESSDENIED

에이전트 샌드박스 내 WSL 서비스 접근이 차단되었습니다. 허용된 외부 실행에서는
배포판 열람 및 환경 점검이 성공했습니다. 배포판 손상으로 해석하지 않습니다.

## 네트워크 시작 경고

실행 시 NAT 설정 실패 후 VirtioProxy 전환 경고를 관측했습니다.
설치 전에 Ubuntu 내에서 DNS/HTTPS 접근을 점검하고 실제 오류를 기록합니다.
현재 WSL 네트워크 설정을 변경한 것은 아닙니다.

## 빌드 OOM 또는 저장 공간 부족

현재 RAM 3.8GiB와 C: 여유 약 13GiB가 제약입니다. Linux VHDX의 논리 여유 공간은
호스트 물리 여유 공간을 보장하지 않습니다. 설치 전에 VHDX 위치를 확인하고
필요한 공간을 확보합니다. 빌드는 병렬도 1부터 시작합니다.

## Docker 연결 실패

클라이언트가 있어도 daemon 연결이 실패할 수 있습니다. 현재 native WSL 계획에는
Docker가 필요하지 않습니다. 이 결과는 TTSim 실행 실패를 의미하지 않습니다.

## simulator 또는 SOC descriptor 누락

설치 후 `TT_METAL_SIMULATOR`가 실제 .so를 가리키는지, 같은 디렉터리에
`soc_descriptor.yaml`이 있는지 확인합니다. 선택한 TT-Metal 태그의 실제 파일 경로를
사용합니다. 근거: [공식 TTSim 실행 문서](https://github.com/tenstorrent/ttsim#running-with-tt-metalium).

## 툴체인 및 버전 충돌

0-B에서 Linux CMake 3.28.3, Ninja 1.11.1을 설치했습니다. Clang은 아직 미설치입니다.
고정 태그와 submodule 전체의 요구 버전을 빌드 전에 확인합니다.
현재 설치 계획의 두 버전은 호환성 후보이며 검증 완료 조합이 아닙니다.
기존 CUDA나 시스템 Python을 변경해서 해결하지 않습니다.

## 외부 Vector Add 프로젝트

`TT_METAL_RUNTIME_ROOT`가 없으면 root directory 오류가 발생합니다.
고정 `rtoptions.cpp`에서 확인한 이 변수를 runner가 설정합니다.
64-bit DeviceAddr는 범위 검사 후 32-bit runtime arg로 변환합니다.
최초 빌드 시 범위 검사 추가 전의 narrowing 오류와 runtime root 미설정 로그를 보존했습니다.
최종 빌드와 두 variant의 모든 검증은 통과했습니다. [08번](08-vector-add.md).
