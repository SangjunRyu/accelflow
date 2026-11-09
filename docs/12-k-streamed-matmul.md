# 12. K-streamed MatMul — 일정한 CB 용량으로 K 확장

상태: **K streaming 구현·K<=512 검증 완료**. 작업 날짜: 2026-10-07.
[진행 순서](README.md) · 이전: [11. Blocked MatMul](11-blocked-matmul.md)

## 해결한 문제

기존 blocked는 출력 2×2 타일을 계산하기 위해 A/B의 K 전체를 L1에 보관했습니다.
CB 용량이 Kt에 비례하므로 첫 버전은 padded K<=128로 제한했습니다.

새 `streamed` 모드는 K 타일을 **chunk 1개 또는 2개씩** 공급합니다.
입력만 교체하며 네 출력 타일의 누적값은 destination registers에 유지합니다.
K chunk마다 중간 출력을 BF16으로 pack하거나 DRAM에 쓰지 않습니다.
따라서 중간 저장에 따른 추가 양자화나 partial-result spill이 없습니다.
최종 비교는 **모든 모드에 FP32 destination 누적**을 적용했습니다.
입력과 최종 출력은 BF16이고 math fidelity는 HiFi4입니다.
중간 결과를 저장하지 않아도 입력 양자화·FP32 누적·BF16 최종 출력 반올림 오차는 남습니다.

## 읽는 순서와 동작

1. [matmul.cpp](../npu/matmul/matmul.cpp): `streamed`, chunk, CB 용량, CPU 검증 선택
2. [reader_streamed.cpp](../npu/matmul/kernels/reader_streamed.cpp): 출력 블록마다 K축 chunk를 순서대로 읽음
3. [compute_streamed.cpp](../npu/matmul/kernels/compute_streamed.cpp): destination 유지, 입력 CB 교체, 전체 K 누적
4. [writer_blocked.cpp](../npu/matmul/kernels/writer_blocked.cpp): 기존 출력 주소 매핑 그대로 재사용

```text
출력 블록 시작 → destination acquire (한 번)
    K chunk 준비 → A/B CB wait
        matmul_tiles를 K 오름차순으로 누적
    A/B 입력 슬롯 pop → 다음 chunk 공급
K 전체 완료 → destination commit/wait → 유효 출력 pack/push → release
```

최대 4개 출력 destination tile은 전체 K 처리 동안 유지됩니다.
각 출력의 K 누적 순서는 baseline/reuse_a/blocked와 같습니다.
출력 쓰기는 K 전체가 끝난 뒤에 수행하며 출력 CB depth1도 지원합니다.
FP32 누적 설정은 `ComputeConfig.fp32_dest_acc_en = true`이며 원본 upstream API를 사용합니다.

### 마지막 chunk와 홀수 출력 경계

`count=min(chunk,Kt-start)`개만 읽고 계산합니다.
A의 CB 인덱스는 `r*chunk+k`, B는 `k*2+c`, destination은 `r*2+c`입니다.
마지막 chunk가 작거나 출력 행/열이 하나만 남으면 미사용 슬롯은 접근하지 않습니다.
reserve/push/pop은 각 입력에서 항상 `2*chunk` 슬롯을 사용합니다.
고정 크기 슬롯과 그 배수의 CB 용량으로 ring wrap 시 연속된 chunk 공간을 유지합니다.

## 메모리와 전송량 모델

`d=depth`, `q=K chunk tile 수`, `Mb=ceil(Mt/2)`, `Nb=ceil(Nt/2)`입니다.

| 지표 | 전체 K 보관 blocked | K-streamed |
| --- | --- | --- |
| A/B CB 용량 (각각 tiles) | 2×Kt×d | 2×q×d |
| output CB 용량 (tiles) | d | d |
| 구성 CB payload bytes | 2048×(4Kt+1)×d | 2048×(4q+1)×d |
| 최대 destination tile slots | 4 | 4 |
| 입력 read payload | 2048×Kt×(Mt×Nb+Nt×Mb) | 동일 |
| C tile writes | Mt×Nt | 동일 |
| matmul_tiles 호출 | Mt×Kt×Nt | 동일 |
| read barrier 호출 | Mb×Nb | Mb×Nb×ceil(Kt/q) |
| partial-result pack/spill | 없음 | 없음 |

위 값은 **소스 기반 모델**이며 실물 DRAM counter·cycle 측정이 아닙니다.
CB 용량은 구성한 payload capacity이고 전체 L1 사용량이나 destination register 크기는 포함하지 않습니다.

| 설정 | K와 무관한 streamed CB 구성 용량 |
| --- | --- |
| chunk1, depth1 | 10KiB |
| chunk1, depth2 | 20KiB |
| chunk2, depth1 | 18KiB |
| chunk2, depth2 | 36KiB |

K=128, depth2에서 전체 K 보관 blocked는 68KiB였지만 streamed chunk2는 **36KiB**입니다.
K=512(Kt=16)에서도 streamed chunk2는 동일한 **36KiB**를 사용합니다.
Kt=16 전체를 보관한다면 식상 260KiB가 필요하지만, 기존 blocked는 그 크기를 지원/실행하지 않았습니다.

이 변경은 기존 blocked보다 읽기 payload를 더 줄이는 최적화가 아닙니다.
동일한 타일 재사용을 더 작은 입력 저장 공간으로 수행하는 설계입니다.
대신 chunk 준비와 read barrier 횟수가 늘어납니다.
chunk2는 chunk1보다 더 많은 L1을 쓰고 barrier 수를 줄입니다.
실제 overlap, bandwidth, speedup은 미측정입니다.

## 검증 설계와 결과

기존 14개 사례를 유지하며 다음 6개를 추가했습니다.

| 입력 | M,K,N | 목적 |
| --- | --- | --- |
| signed random | 65,129,97 | Kt=5, 마지막 chunk1타일, 출력 경계 |
| signed random | 65,257,97 | Kt=9, 여러 CB wrap, 출력 경계 |
| signed random | 64,512,64 | Kt=16, K 확장 및 반복 chunk |
| fractional | 33,161,65 | Kt=6, 소수 BF16 누적 |
| fractional | 33,257,65 | Kt=9, 짧은 마지막 chunk와 소수 오차 |
| zero | 33,257,65 | K 확장, 전체 출력 덮어쓰기와 padding |

baseline/reuse_a는 depth1/2에서 각각 20개, 기존 blocked는 Kt<=4인 공통 14개만 실행합니다.
streamed는 chunk1/2 × depth1/2, 각각 20개를 실행합니다.
총 **10개 실행 설정, 188개 기능 검증**입니다.
기존 blocked의 K 범위 밖인 6개 사례는 명시적으로 제외하며 성공 사례로 세지 않습니다.

최종 **188개 모두 PASS, 10개 실행 모두 exit code 0**입니다.
지원하는 설정 사이의 동일 입력/출력 FNV fingerprints가 모두 일치했습니다.

| Fractional M,K,N | FP32 destination 적용 후 최대 절대 오차 |
| --- | --- |
| [32, 64, 64] | 0.0073905885 |
| [33, 35, 65] | 0.0038948655 |
| [33, 161, 65] | 0.0082382783 |
| [33, 257, 65] | 0.015813819 |


### 첫 확장 실행에서 발견한 수치 한계

기존 정밀도 설정으로 K=257 fractional 입력을 실행했을 때 baseline부터 실패했습니다.
좌표 (26,1)에서 reference=1.241537, actual=1.312500으로 오차 약 0.070963이
허용 오차 약 0.067246을 초과했습니다.
[첫 실패 실행 메타데이터](../results/matmul/20261007T102400527330Z/metadata.json)와
[baseline 전체 로그](../results/matmul/20261007T102400527330Z/baseline_depth1.log)를 보존했습니다.

이를 streaming 오류나 단순 허용 오차 문제로 넘기지 않고, 모든 최종 비교 모드에 FP32 destination
누적을 적용했습니다. 같은 입력·seed와 같은 허용 오차로 다시 검증했습니다.
10/11번 문서의 이전 정밀도 설정 결과와 최종 출력 fingerprint가 같아야 한다고 주장하지 않습니다.
이번 전송량·메모리 비교는 동일한 FP32 누적 설정 내에서 이루어집니다.

입력 배치는 공식 tilizer와 비교합니다. 출력은 NaN sentinel로 초기화하고
CPU FP64 reference와 유효 영역을 비교하며 모든 padding이 0인지 검사합니다.
정수 fixture는 exact 비교, fractional은 기존 `0.03+0.03*abs(reference)` 기준을 유지합니다.
허용 오차를 K 확장에 맞춰 느슨하게 바꾸지 않았습니다.
이 기준은 현재 fixture의 검증용이며 모든 K/입력 범위에 대한 BF16 오차 보장은 아닙니다.

## 재현

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_matmul.sh
```

실행 파일 인자는 `matmul report.json streamed depth [chunk]`입니다.
chunk 생략 시 2, depth/chunk는 1 또는 2입니다. 기존 세 모드에는 chunk 인자를 전달하지 않습니다.
장치 환경 설정과 descriptor 준비는 `run_matmul.py`에서 기존 고정 simulator 경로로 수행합니다.

전체 실행 로그·exit code·compiler/OS/버전·소스 SHA-256·정확성·구조 모델은 아래에 저장했습니다.

- [최종 메타데이터](../results/matmul/20261007T102645791226Z/metadata.json)
- [chunk1/depth1](../results/matmul/20261007T102645791226Z/streamed_depth1_chunk1.json)
- [chunk1/depth2](../results/matmul/20261007T102645791226Z/streamed_depth2_chunk1.json)
- [chunk2/depth1](../results/matmul/20261007T102645791226Z/streamed_depth1_chunk2.json)
- [chunk2/depth2](../results/matmul/20261007T102645791226Z/streamed_depth2_chunk2.json)
- [전체 streaming 로그](../results/matmul/20261007T102645791226Z/streamed_depth2_chunk2.log)
- [설계 평가 JSON](../results/matmul/20261007T102645791226Z/design-assessment.json)
- [자동 생성 비교 표](../results/matmul/20261007T102645791226Z/design-assessment.md)

평가 도구는 모드별 지원 fixture 집합, 동일 입력/출력 fingerprints, CPU 검사,
독립 계산한 CB/전송량/barrier 모델 및 K와 무관한 streamed CB 용량을 확인합니다.
최신 실행 실패 시 이전 성공 결과로 대체하지 않습니다.

## 범위와 다음 단계

streamed에는 기존 Kt<=4 제한이 없습니다. **이번 실제 검증 범위는 K<=512**입니다.
무한한 K, 임의 크기의 장치 주소/DRAM 용량, NaN/Inf/overflow 또는 모든 입력 범위를 검증한 것은 아닙니다.
현재 CLI는 임의 shape를 입력받는 GEMM 서비스가 아니라 고정 fixture 검증 프로그램입니다.

다음은 shape와 L1 예산에 따라 baseline/reuse_a/blocked/streamed를 선택하는 정책 평가,
CUDA 실측 비교 또는 multi-core 분할입니다. 성능 주장은 실제 장치 측정과 구분합니다.

후속 측정: [13. TTSim 실행 비용 비교](13-matmul-simulator-performance.md)에서 warmup을 제외한 반복 dispatch의 host 시간과 기존 구조 모델을 비교했습니다. 실물 NPU latency는 아닙니다.
