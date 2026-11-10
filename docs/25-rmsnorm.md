# 25. RMSNorm 구현과 Fusion 비교

상태: **CPU 기준값·CUDA 기본/fused·NPU 단일 코어 검증 완료**. 날짜: 2026-11-09.
[진행 순서](README.md) · 이전: [24. CUDA Reduction](24-cuda-reduction.md)

## 계산과 입력

각 행에서 `y[h,w] = x[h,w] * rsqrt(sum(x[h,:]^2)/W + epsilon) * gamma[w]`를 계산했다.
행 평균의 분모는 padding 폭이 아닌 논리 폭 W다. gamma는 열별 가중치이고 epsilon은 양수다.
입력과 gamma는 같은 BF16 binary fixture에서 읽어 FP32로 정확히 확장했다.
누적·중간 buffer·출력은 FP32이며 CPU 기준값은 실제 양자화 입력과 저장된 FP32 epsilon을 사용한다.

Python의 `math.fsum`·`math.sqrt` 기준값을 별도로 계산했다. 각 C++ host의 FP64 검증 외에
Python 기준값과 NPU·CUDA 출력을 직접 대조하므로 두 host가 같은 실수를 해도 놓치지 않도록 했다.
20종 공통 입력은 zero, ones, signed, fractional, tiny, mixed magnitude, gamma=0,
음수·비균일 gamma, epsilon=1e-6·1e-5·1e-3을 포함한다.
H=1·2·3·17·31·32·33·65, W=1·3·31·32·33·35·65·129·257·1025·4097을 검사했다.

## CUDA 기본 구현과 Fused 구현

| 구분 | 기본 unfused | fused |
| --- | --- | --- |
| kernel 호출 | square → inverse_rms → normalize, 총 3회 | 총 1회 |
| reduction | 256 threads/행, warp shuffle + shared 부분합 | 같은 작업 분배와 집계 방식 |
| 입력 보관 | 정규화 단계에서 global 입력을 다시 읽음 | 행 전체를 shared memory에 보관 |
| global 중간 buffer | squared matrix + 행별 inverse RMS | 없음 |
| block shared memory | reduction에서 32 bytes | `4*(W+9)` bytes |

기본 구현도 행별 역 RMS는 한 번만 계산한다. 두 구현 모두 FP32로 제곱을 반올림한 뒤
합산하므로 fused만 제곱과 덧셈의 FMA 결합으로 다른 계산을 하도록 두지 않았다.
fusion 비교에는 호출 수, 중간 buffer, 입력 재사용과 자원 사용량 변화가 함께 들어간다.
shared memory 하나의 효과나 bank conflict 감소를 분리 측정한 결과는 아니다.

1024×8193에서 padded width는 8224다. 기본 구현의 global 중간 buffer는 33689600 bytes,
fused는 0 bytes다. fused block은 shared memory 32808 bytes를 사용한다.
레지스터·공통 입출력 buffer는 이 중간 buffer 수치에 포함하지 않았다.
shared memory가 커지면 block 상주 수에 영향을 줄 수 있으나 occupancy counter는 수집하지 않았다.

## NPU 단일 코어 데이터 흐름

가상 Wormhole에서는 정확성 기준이 되는 3-stage 구현을 만들었다. NPU fusion은 아직 하지 않았다.

```text
DRAM input → SFPU square → DRAM squared tiles
           → 두 pass의 transpose/cumsum → DRAM sum-square prefix
           → SFPU mean + epsilon + rsqrt + x*gamma → DRAM output
```

각 stage는 reader/compute/writer와 CB를 사용한다. stage 사이에서 host로 값을 가져와
합이나 정규화를 계산하지 않는다. 검증용 중간 readback은 세 stage가 끝난 뒤 수행한다.
RISC-V reader는 타일 배치와 비트 분할을 맡고 산술 연산은 compute engine에서 처리한다.

제곱합은 기존 reduction의 SFPU scan을 재사용했다. 행을 전치해 열 방향으로 누적하고
마지막 prefix를 행별 제곱합으로 사용한다. 두 pass의 마지막 prefix는 DST 슬롯 2개에 유지한다.
scan carry가 살아 있는 동안 다른 SFPU 연산을 끼우지 않고 두 scan이 끝난 뒤 결과를 합친다.
row group이 바뀌면 carry를 초기화한다.

## FP32 저장과 입력 경로의 정밀도

초기에는 squared buffer와 통계 buffer를 FP32로 저장하면 다음 stage에서도 그대로 읽힐 것으로
생각했지만 검증이 실패했다. 65×129 입력에서 제곱합 3085는 DRAM에서 정확했으나
정규화 출력은 기준 1.635902 부근에서 1.636167로 벗어났다.
중간 mean을 출력한 probe에서는 3105/129+epsilon 대신 약 3104/129+epsilon에 해당하는 값이 나왔다.
역제곱근 Newton 보정만 추가해도 바뀌지 않아 최종 구현에서는 그 보정을 제거했다.

공식 API는 SrcA/B를 거치는 경로에서 operand 정밀도가 줄어든다고 설명한다.
고정된 Wormhole/TTSim 조합에서 FP32-to-Dest 설정을 시험했을 때는 simulator가
`UndefinedBehavior: ... unpack_to_dst=0 in_data_format=0 out_data_format=0`으로 종료했다.
최종 구현은 기본 입력 모드를 유지하고 TF32 경로에서 정확히 표현되는 조각을 전달한다.

- BF16 값의 정확한 FP32 제곱은 최대 16개 유효 비트를 갖는다. 제곱합 입력을 hi/low 두 조각으로
  나눠 각각 scan하고 SFPU에서 합친다. squared DRAM 입력은 두 번 읽는다.
- 행별 FP32 통계는 hi/middle/low 세 조각으로 보내 SFPU에서 재구성한 뒤 평균과 rsqrt를 계산한다.
  데이터 이동 커널은 정수 비트 연산만 사용한다.

실제 corpus의 제곱 30008개와 행별 통계 274개에 대해, 각 조각이 TF32로 표현 가능하고
FP32 재구성이 원래 값과 일치하는지 별도 검사했다. 이 방식이 임의의 NaN/Inf/subnormal과
전체 BF16 동적 범위를 지원한다고 주장하지 않는다. 실패 로그와 최종 source를 함께 보존했다.

공식 근거: [입력 경로와 정밀도](https://github.com/tenstorrent/tt-metal/blob/de546d3b146758714d900f11b218c8f9c805f410/tt_metal/api/tt-metalium/base_types.hpp),
[Wormhole FP32/TF32 경로를 구분한 테스트](https://github.com/tenstorrent/tt-metal/blob/de546d3b146758714d900f11b218c8f9c805f410/tests/tt_metal/tt_metal/llk/test_sfpu_binary_bcast.cpp).

## 검증 결과

허용오차는 구현 전 정한 `atol=1e-4`, `rtol=1e-4`를 유지했다.
출력을 NaN sentinel로 초기화하고 모든 유효 값과 padding을 검사했다.

- NPU: 입력 20종 × CB depth1/2 = 40개 구성, 구성당 2회로 80회 operator 제출을 통과했다.
  pilot 2회는 별도다. operator마다 stage 3개를 실행한다. depth 사이 입력·gamma·출력 fingerprint도 일치했다.
- CUDA: 공통 정확성 입력과 큰 입력 4종을 합쳐 64개 process, 1184회 출력 검증을 통과했다.
- NPU의 square는 원소별 FP32 제곱과 exact 비교했고 제곱합과 최종 출력은 FP64 기준과 대조했다.
- 공통 입력 20종에서 NPU와 CUDA 기본/fused 출력을 Python 기준값으로 다시 검증했다.

CUDA 최대 절대 오차는 1.54650299677e-07, NPU는 2.33068715882e-07였다.
세 backend 출력의 최대 절대 차이는 2.38418607745e-07였고
Python FP64 기준 대비 최대 절대 오차는 2.33068715882e-07였다.

## CUDA 실측

MX150, CUDA 12.9, Windows driver 582.66에서 실행했다. 20회 warmup 뒤
15개 sample을 수집하고 sample당 operator 20회를 CUDA Event로 감쌌다.
표는 batch 평균의 sample 중앙값, process 3개의 중앙값을 차례로 구한 값이다.
기본 구현의 event 구간에는 kernel 3개가 모두 들어간다.
GPU scheduling과 host 제출 사이 빈 구간이 포함될 수 있어 순수 instruction latency로 보지 않는다.

pipeline은 pinned input/gamma H2D → operator 1회 → output D2H → stream 완료 대기다.
allocation, 입력 parsing, sentinel 초기화, CPU 검증은 측정에서 제외했다.
큰 입력 4종은 CUDA 성능용이며 TTSim에서 같은 큰 입력을 실행한 것은 아니다.

| H×W | 기본 연산 구간 ms | fused 연산 구간 ms | 관찰 배율 | pipeline ms |
| --- | --- | --- | --- | --- |
| 128×320 | 0.037210 | 0.016486 | 2.26배 | 0.2155 → 0.2381 |
| 640×320 | 0.127083 | 0.066304 | 1.92배 | 0.7961 → 0.7968 |
| 320×4097 | 0.639248 | 0.267621 | 2.39배 | 4.1077 → 3.8202 |
| 1024×8193 | 3.990170 | 1.974477 | 2.02배 | 25.2811 → 23.5033 |

process 중앙값 3개의 최소–최대도 남긴다. 아래 범위는 개별 sample 범위가 아니다.

| H×W | 구현 | process 중앙값 범위 ms |
| --- | --- | --- |
| 128×320 | unfused | 0.036776–0.039528 |
| 128×320 | fused | 0.014234–0.075774 |
| 640×320 | unfused | 0.126989–0.131910 |
| 640×320 | fused | 0.066099–0.072085 |
| 320×4097 | unfused | 0.639109–0.641634 |
| 320×4097 | fused | 0.266955–0.277965 |
| 1024×8193 | unfused | 3.989739–3.990401 |
| 1024×8193 | fused | 1.848064–1.998336 |

128×320 fused는 한 process에서 지연이 커서 실행마다 일관된 우위를 보장하지 않는다.
나머지 세 크기는 이번 process 범위에서 연산 구간이 분리됐다.
1024×8193에서는 약 2.02배였지만 pipeline 개선은 약 7.0%다.
전송·동기화 비용까지 포함하면 kernel fusion의 이점이 작아진다.
bank-conflict hardware counter, 실제 DRAM traffic, 원인별 stall은 측정하지 않았다.

## NPU 메모리·전송 모델

`T=ceil(H/32)*ceil(W/32)`, `R=ceil(H/32)`로 두면 중간 DRAM capacity는 `(T+R)*4096` bytes다.
CB payload 최대값은 depth1에서 28KiB, depth2에서 52KiB이며 code·stack·전체 L1 크기는 제외한다.
source에서 계산한 DRAM payload는 `(7*T+2*R)*4096` bytes다.
이는 squared 입력 두 pass, tile별 gamma 읽기, row group별 통계 읽기를 포함한다.
하드웨어 카운터 측정값은 아니다. TTSim wall time도 실제 NPU 지연 시간으로 보고하지 않는다.

## 실행과 자료

WSL에서 NPU 검증만 실행한다.

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_rmsnorm.sh
```

Windows PowerShell에서 공통 입력으로 NPU 검증과 CUDA 비교를 실행한다.
UNC script가 RemoteSigned에 막히면 script만 로컬에 복사한다.

```powershell
$project = '\\wsl.localhost\Ubuntu-24.04\home\ysjun5656\dev\accelflow'
$runner = Join-Path $env:TEMP 'accelflow-run-cuda-rmsnorm.ps1'
Copy-Item (Join-Path $project 'scripts\run_cuda_rmsnorm.ps1') $runner
& $runner -ProjectRoot $project
```

- [CUDA 코드](../cuda/rmsnorm/rmsnorm.cu) · [읽기 안내](../cuda/rmsnorm/README.md)
- [NPU host](../npu/rmsnorm/rmsnorm.cpp) · [stage·정밀도 전달 설명](../npu/rmsnorm/README.md)
- [공통 입력 생성](../tools/report/rmsnorm_fixtures.py) · [독립 대조](../tools/report/audit_rmsnorm.py)
- [NPU 검증 요약](../results/rmsnorm/20261109-rmsnorm/npu-summary.json)
- [CUDA 실측 요약](../results/rmsnorm/20261109-rmsnorm/cuda-summary.json)
- [직접 대조·Python 기준값 검증](../results/rmsnorm/20261109-rmsnorm/cross-backend-audit.json)
- [정밀도 전달 검사](../results/rmsnorm/20261109-rmsnorm/precision-transport-audit.json)
- [NPU 명령·환경·source hash](../results/rmsnorm/20261109-rmsnorm/npu-metadata.json)
- [CUDA 명령·원시 표본](../results/rmsnorm/20261109-rmsnorm/cuda-measurements.json)
- [최종 자료 감사](../results/rmsnorm/20261109-rmsnorm/audit.json)
- [공식 rsqrt API](https://github.com/tenstorrent/tt-metal/blob/de546d3b146758714d900f11b218c8f9c805f410/tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h)

실제 NPU와 GPU의 속도를 비교한 결과는 아니다. NPU에서는 기능과 데이터 흐름을,
MX150에서는 CUDA fusion의 실측 효과를 확인했다. NPU fusion·multicore RMSNorm과
TT-NN·모델 실행은 후속 단계다.


## PyTorch 기준 대조 추가

프로젝트 `.venv`에 CPU 전용 PyTorch 2.15.0+cpu를 설치하고,
공식 `torch.nn.functional.rms_norm`의 FP32·FP64 결과와 공통 입력 20종을 대조했다.
기존 NPU depth1/2와 CUDA unfused/fused의 저장된 출력 80개가 모두 통과했다.
커널 재실행이나 GPU 성능 측정은 하지 않았다.

원래 BF16 입력·gamma 비트를 FP32로 확장하고, binary에 저장된 epsilon을 명시적으로 전달했다.
BF16 출력과 비교하는 검사는 아니며 기존 `atol=1e-4`, `rtol=1e-4`를 유지했다.

| 저장된 출력 | PyTorch FP32 대비 최대 절대 오차 | PyTorch FP64 대비 최대 절대 오차 |
| --- | --- | --- |
| NPU depth1/2 | 2.384185791015625e-07 | 2.3306871588246736e-07 |
| CUDA unfused | 1.1920933906672815e-07 | 1.5465025549055156e-07 |
| CUDA fused | 1.1920933906672815e-07 | 1.5465025549055156e-07 |

PyTorch FP64와 기존 Python `math.fsum` 기준값의 최대 차이는 0이었다.
이번 20종 입력에서 프레임워크 기준과 일치함을 추가 확인했다.
backward, 임의 입력 범위, PyTorch GPU 속도 비교는 검증 범위에 포함하지 않았다.

```bash
cd /home/ysjun5656/dev/accelflow
# 새 환경을 준비할 때만 실행
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-pytorch-reference.txt

.venv/bin/python tools/report/compare_rmsnorm_pytorch.py \
  --results results/rmsnorm/20261109-rmsnorm
```

- [대조 결과·버전·입력/보고서 SHA-256](../results/rmsnorm/20261109-rmsnorm/pytorch/comparison.json)
- [실행 로그](../results/rmsnorm/20261109-rmsnorm/pytorch/comparison.log)
- [설치 패키지 고정 목록](../results/rmsnorm/20261109-rmsnorm/pytorch/requirements-lock.txt)
- [대조 스크립트](../tools/report/compare_rmsnorm_pytorch.py)
- [공식 PyTorch API](https://docs.pytorch.org/docs/2.15/generated/torch.nn.functional.rms_norm.html)

후속 NPU 단일 코어 fusion과 L1/DRAM 비교는 [26번](26-npu-rmsnorm-fusion.md)에서 완료했습니다.
