# 31. FP32-to-Dest RMSNorm 정밀도 실험

상태: **별도 디렉터리 구현·로컬 호환 경로 검증 완료. native API 경로는 현재 조합에서 실패**.
날짜: 2026-11-12. [진행 순서](README.md) · [기존 RMSNorm](25-rmsnorm.md) · [포트폴리오](30-portfolio.md)

## 확인하려던 질문

기존 `npu/rmsnorm/`은 FP32 operand가 SrcA/B를 거치며 TF32로 잘리는 문제를
`residual_bits`의 hi/low·hi/middle/low 분할로 피했다. 이번에는 `npu/rmsnorm_fp32_dest/`를
따로 만들고 `fp32_dest_acc_en=true`와 입력 CB별 `UnpackToDestFp32`를 사용했다.
제곱·합산·정규화 경로에는 데이터의 mantissa를 나누거나 재구성하는 함수가 없다.

결과는 두 가지로 나뉜다. 순정 `copy_tile`·`transpose_tile`에 해당 옵션만 켠 `direct`는
실패했다. 실험 내부에서 LLK의 후행 보조 unpack을 조정한 `compat`는 copy·transpose의
비트 보존과 RMSNorm 계산을 통과했다. 따라서 **옵션만 켜면 기존 조합에서 바로 동작한다는 결과는 아니다**.
기존 RMSNorm 소스·TT-Metal checkout·기본 simulator는 수정하지 않았다.

## 모드와 데이터 경로

모든 모드는 Dest를 FP32로 사용한다. 입력 전달 방식과 보조 unpack 처리만 구분한다.

| 모드 | 입력 CB 설정 | 로컬 보조 unpack 처리 | 확인 결과 |
| --- | --- | --- | --- |
| default | Default → SrcA/B TF32 | 없음 | transport 하위 비트 손실, 실제 FP32 RMSNorm 대조군 실패 |
| direct | UnpackToDestFp32 | 없음 | native LLK 경로가 simulator 오류로 종료 |
| compat | UnpackToDestFp32 | 실험 내부 TEN-3868 호환 처리 | transport 비트 보존·RMSNorm 검증 통과 |

`UnpackToDestFp32`를 표시한 CB는 `copy_tile`·FP32 `transpose_tile`만 소비한다.
곱셈·덧셈·scan·rsqrt는 Dest의 SFPU 연산이다. `add_tiles`·`mul_tiles`·`reduce_tile` 같은
SrcA/B FPU 연산으로 marked CB를 읽지 않는다. `mul_binary_tile`·`add_binary_tile`은
이름이 비슷해도 `eltwise_binary_sfpu.h`의 Dest 연산이다. 이 경로에는 dual-consumer alias가 필요 없다.

```text
DRAM FP32 X → copy-to-Dest → SFPU square → DRAM FP32 squared
             → transpose-to-Dest → SFPU cumsum 1 pass → DRAM FP32 prefix
             → copy-to-Dest(sum / X / gamma) → SFPU normalize → DRAM FP32 Y
```

정밀도 전달을 분리해 확인하려고 단일 코어·3-stage로 구성했다. 기존 fused/multicore를
통째로 바꾸지 않았다. reader는 FP32 words의 이동과 행별 통계 복제만 하고 산술은 compute에서 한다.
최종 평균의 분모는 padding 폭이 아닌 논리 W다. 중간 DRAM을 제거한 fusion 실험이나 성능 개선 결과는 아니다.

## Native 실패를 좁힌 과정

FPU 계산이 전혀 없는 copy·transpose만으로도 기존 오류가 재현됐다.

```text
UndefinedBehavior: tensix_unpacr:
unpack_to_dst=0 in_data_format=0 out_data_format=0
```

고정 LLK의 `unpack_to_dest_tile_done`은 실제 Dest 전송이 끝난 뒤 TEN-3868 workaround로
SrcA unpack을 한 번 더 발행한다. 이 후행 명령은 `unpack_to_dst=0`이며 FP32 output 형식이
남아 있으면 simulator가 거부한다. 해당 helper에는 ZeroWrite2와 context 선택도 포함돼 있다.
입력·출력 형식만 조정한 시도는 이어서 `UntestedFunctionality: zero_write2=1`,
실제 discarded write로 바꾼 시도는 `src_a row=64`, 여러 타일에서는 `cfg_context_id=1`에 막혔다.

최종 [fp32_unpack_compat.hpp](../npu/rmsnorm_fp32_dest/kernels/fp32_unpack_compat.hpp)는
이 후행 보조 unpack만 바꾼다.

- 실제 FP32-to-Dest 전송이 끝나기를 기다리고 보조 unpack의 출력 형식만 잠시 TF32로 설정한다.
- TEN-3868의 후행 SrcA unpack 자체는 유지하되 ZeroWrite2 대신 버리는 실제 write를 수행한다.
- 보조 write의 Z/W counter를 0으로 맞추고 지원되는 config context 0을 사용한다.
- 보조 unpack 완료 후 실제 데이터용 FP32 형식을 복원한다.

보조 SrcA의 값은 산술에 사용하지 않는다. 실제 입력은 이미 FP32 Dest에 전달돼 있다.
simulator의 검사나 소스는 바꾸지 않았고 검사를 끄는 설정도 사용하지 않았다.
반면 이 코드는 **공식 upstream 수정이나 실제 칩에서 검증된 workaround가 아니다**.
현재 고정 revision의 SFPU 전용 실험에 한정한다. 다른 LLK·FPU 혼용·실물 장치에 일반화하지 않는다.

기존 TTSim v1.11.1 외에 공식 v1.16.0 prebuilt를 별도 경로에 받아 native 최소 경로도 확인했다.
그 조합도 같은 형식 오류로 종료했다. 최종 전체 검증은 원래 v1.11.1을 사용했다.
새 simulator를 기본 경로에 덮어쓰거나 TT-Metal 전체를 재빌드하지 않았다.

공식 근거:
[고정 버전의 unpack-to-Dest 계약](https://github.com/tenstorrent/tt-metal/blob/de546d3b146758714d900f11b218c8f9c805f410/tt_metal/api/tt-metalium/base_types.hpp),
[TEN-3868 helper](https://github.com/tenstorrent/tt-metal/blob/de546d3b146758714d900f11b218c8f9c805f410/tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/cunpack_common.h),
[FPU와의 혼용 제한 #61208](https://github.com/tenstorrent/tt-metal/issues/61208),
[TTSim v1.16.0](https://github.com/tenstorrent/ttsim/releases/tag/v1.16.0).
#61208의 잘못된 FPU operand 사용과 이번 SFPU-only 경로의 후행 unpack 문제는 구분한다.

## 검증 결과

transport는 양수·음수의 실제 FP32 값 4096개를 4개 타일에 넣어 CB wrap과 번갈아 쓰는 context까지 검사했다.
값은 BF16이나 TF32로 정확히 표현되지 않는 하위 mantissa 비트를 갖는다. 출력은 NaN sentinel로 초기화했다.

| 검사 | default | direct | compat |
| --- | --- | --- | --- |
| copy, 4096개 | 비트 불일치 4096 | simulator 오류 | 비트 불일치 0 |
| transpose, 4096개 | 비트 불일치 4096 | simulator 오류 | 좌표 변환 후 비트 불일치 0 |

RMSNorm은 기존 BF16 fixture 20종과 새 FP32 fixture 9종을 사용했다. 후자는 low bits,
fractional, 1×1025, 65×129 tail, tiny, zero, FP32 gamma, epsilon, 큰 offset 입력을 포함한다.
기존 fixture는 `AFRM0001`의 BF16 비트를 확장하고 새 `AFRF0001`은 FP32 words를 직접 읽는다.
FP32 gamma도 BF16으로 내리지 않는다.

- CB depth1·2, 총 **58개 구성**을 각각 2회 실행했다. pilot 4회를 포함해 **120회 RMSNorm 계산**, 3-stage 프로그램 **360회 제출**이다.
- 최종 transport 6회와 FP32 기본 경로의 RMSNorm negative control 2회는 별도다. native transport 2회는 의도한 실패 대조다.
- PyTorch FP32·FP64와 저장된 **58개 출력**을 대조했다. `atol=1e-4`, `rtol=1e-4`를 유지했다.
- 제곱 결과는 CPU FP32 곱셈과 모든 검사에서 비트 단위로 일치했다. 제곱합·최종 출력은 FP64 기준과 대조했다.
- 출력 전체와 padding을 검사했고 반복 결과와 depth 사이 출력은 비트 단위로 일치했다.
- BF16 공통 입력 20종은 기존 split/fused 출력과도 비교했다. 입력·gamma fingerprint가 일치했고 출력 최대 차이는 4.76837158203e-07였다.

| 지표 | 결과 |
| --- | --- |
| CPU FP64 대비 RMSNorm 최대 절대 오차 | 9.83273234834e-07 |
| PyTorch FP32 대비 최대 절대 오차 | 9.53674316406e-07 |
| PyTorch FP64 대비 최대 절대 오차 | 9.8327323661e-07 |
| 제곱합 FP64 대비 최대 절대 오차 | 0.479377686977 |
| 제곱합 최대 상대 오차 | 7.40096805698e-07 |
| FP64 제곱합을 FP32로 반올림한 값과의 최대 ULP 거리 | 9 |
| default RMSNorm 대조군의 최대 절대 오차 | 0.00161911197131 |
| default RMSNorm 대조군의 허용 오차 초과 출력 | 793개 |

제곱합의 절대 오차는 입력 크기에 따라 커진다. 최대 0.479377686977도 포함해 기록했으며
상대 오차·ULP와 함께 읽어야 한다. 입력 전달이 비트 단위로 정확하더라도 SFPU 합산 순서와
FP32 반올림 때문에 최종 합이 correctly rounded FP64 sum과 같지는 않다. bitwise transport와
수치 오차 검증을 통과한 RMSNorm을 같은 의미로 사용하지 않는다.

기본 경로 대조군은 실제 FP32 lowbits·fractional 2종 모두에서 허용 오차를 넘었다.
새 경로가 정확히 전달한 하위 비트가 최종 정규화에도 영향을 준다는 근거다.
검증 범위는 저장한 유한 정상값과 zero corpus다. 임의 FP32 동적 범위, NaN/Inf/subnormal,
overflow·underflow, backward와 모든 합산 순서의 오차 한계를 보장하지 않는다.

## 재현과 코드 읽는 순서

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_rmsnorm_fp32_dest.sh
# 위 명령이 출력한 새 RESULTS 경로
.venv/bin/python tools/report/audit_rmsnorm_fp32_dest.py --results <RESULTS>
```

`run_rmsnorm_fp32_dest.sh`는 native transport의 exit 1을 예상 실패로 확인하고 compat가 통과해야
RMSNorm으로 진행한다. 원래 TT-Metal v0.79.0/TTSim v1.11.1과 기존 `.venv`의 CPU PyTorch를 사용한다.
추가 패키지·driver 설치나 전체 upstream 빌드는 하지 않는다. 약 2.5 GiB의 C: 여유를 확인한 뒤
자체 host·작은 kernel만 빌드했으며 자원 기록은 아래 결과 폴더에 남겼다.

host CLI:

```text
rmsnorm_fp32_dest probe report.json default|direct|compat copy|transpose
rmsnorm_fp32_dest run fixtures.txt report.json depth repeats default|direct|compat
```

[host·CB 설정](../npu/rmsnorm_fp32_dest/rmsnorm_fp32_dest.cpp) →
[copy/transpose probe](../npu/rmsnorm_fp32_dest/kernels/probe.cpp) →
[로컬 LLK 호환 처리](../npu/rmsnorm_fp32_dest/kernels/fp32_unpack_compat.hpp) →
[square](../npu/rmsnorm_fp32_dest/kernels/square.cpp) →
[sum](../npu/rmsnorm_fp32_dest/kernels/sum.cpp) →
[normalize](../npu/rmsnorm_fp32_dest/kernels/normalize.cpp) 순서로 읽으면 된다.

- [전체 summary](../results/rmsnorm_fp32_dest/20261112T121407505012Z/summary.json), [명령·버전·source/executable SHA-256](../results/rmsnorm_fp32_dest/20261112T121407505012Z/metadata.json)
- [PyTorch 대조](../results/rmsnorm_fp32_dest/20261112T121407505012Z/pytorch-audit.json), [제곱합 정밀도·negative control](../results/rmsnorm_fp32_dest/20261112T121407505012Z/precision-controls.json)
- [native copy 실패 로그](../results/rmsnorm_fp32_dest/20261112T121407505012Z/probe-direct-copy.log), [compat copy 비트 검사](../results/rmsnorm_fp32_dest/20261112T121407505012Z/probe-compat-copy.json)
- [compat depth2 실행 로그](../results/rmsnorm_fp32_dest/20261112T121407505012Z/compat-depth2.log)

이 실험은 비트 분할을 없애도 현재 corpus의 FP32 RMSNorm이 동작하는지 확인한 것이다.
fused/multicore 이식은 하지 않았다. 다음 TT-NN 실행 문서는 33번부터 이어간다.

같은 3-stage 조건의 split/FP32-to-Dest 실행 비용 비교는 [32번](32-rmsnorm-fp32-dest-timing.md)에 기록했다.
