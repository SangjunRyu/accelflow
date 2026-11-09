# AccelFlow Vector Add — Phase 1

두 host executable과 자체 장치 커널을 구현했습니다.

| 구현 | 데이터 | 연산 위치 | 장치 커널 |
| --- | --- | --- | --- |
| `vector_add_riscv` | uint32, 모든 원소 정확한 비교 | data-movement RISC-V | 1 |
| `vector_add_tiled` | BF16, 32×32 타일·tail zero padding | Tensix compute | reader / compute / writer 3 |

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_vector_add.sh
```

실행 파일은 `build/vector_add/`, 검증 로그·JSON은 `results/vector_add/<UTC>/`에 생성됩니다.
TT-Metal v0.79.0의 이미 빌드된 shared libraries와 host interface를 사용합니다.
설치되지 않은 build-tree에서는 공식 CMake package export가 완성되어 있지 않아,
`build_vector_add.py`가 고정 예제의 Ninja 컴파일 명령에서 include/define을 추출합니다.
이 adapter는 고정 revision만 지원하며 upstream source/configuration을 수정하지 않습니다.

필요한 simulator/runtime 환경은 `run_vector_add.py`가 설정합니다. 외부 프로젝트에서의
`TT_METAL_RUNTIME_ROOT`는 고정 소스 `tt_metal/llrt/rtoptions.cpp`에서 확인했습니다.

자료형의 의미가 다르므로 두 구현을 동일한 정밀도나 성능으로 직접 비교하지 않습니다.
uint32 덧셈은 modulo 2^32이며, BF16 기준값은 양자화된 입력의 FP32 합입니다.
작은 signed integer 입력은 BF16에서도 정확한 일치를 요구합니다.
타일 tail은 0으로 채우고 출력 padding도 0인지 확인합니다.

상세 흐름·구조 지표·허용 오차는 [08. Phase 1 보고서](../../docs/08-vector-add.md)를 참고하세요.

API 패턴의 근거는 Tenstorrent의 고정 Apache-2.0 예제입니다:

- [정수 RISC-V 덧셈](https://github.com/tenstorrent/tt-metal/tree/v0.79.0/tt_metal/programming_examples/add_2_integers_in_riscv)
- [타일 덧셈](https://github.com/tenstorrent/tt-metal/tree/v0.79.0/tt_metal/programming_examples/eltwise_binary)

원본 예제의 권리와 라이선스는 Tenstorrent에 있습니다. 전체 upstream 저장소는 외부 checkout으로 유지합니다.

설계 평가 보강: 현재 10개 uint32와 14개 BF16 케이스를 depth 1/2로 실행합니다.
최대 65타일, 동일 입력 fingerprint, buffer 정렬 API 값과 구조 모델을 기록합니다.
[09. 구조 평가](../../docs/09-vector-add-design-assessment.md).
