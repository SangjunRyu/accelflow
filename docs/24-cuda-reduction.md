# 24. CUDA Row-wise Reduction 비교

상태: **CUDA 3개 구현·NPU 입력 대조·MX150 실행 검증 완료**. 날짜: 2026-11-09.
[진행 순서](README.md) · 이전: [23. cudaArray·profiler](23-texture-matmul-and-profiler.md)

## 구현 범위

각 행의 합 `y[h] = sum(X[h,w])`을 계산했다. NPU 단일 코어에서 검증한 BF16 입력을
그대로 내보내 CUDA에서 FP32로 확장했다. 누적과 출력도 FP32다.
CUDA 코드는 Windows의 MX150에서 실행하며 WSL에서 GPU kernel을 실행한 것은 아니다.

| 구현 | 작업 분배 | 부분합 집계 | block당 shared memory |
| --- | --- | --- | --- |
| naive | thread 하나가 한 행 | register에서 순서대로 합산 | 0 bytes |
| tree | 256-thread block 하나가 한 행 | shared-memory halving tree | 1024 bytes |
| shuffle | 256-thread block 하나가 한 행 | warp 내 shuffle + warp 간 shared 부분합 | 32 bytes |

tree는 shared memory에 thread별 부분합을 쓴 뒤 8단계로 줄인다. 처음 저장까지 포함하면
block barrier는 9회다. shuffle은 각 warp의 32개 값을 register shuffle로 더한 뒤
8개 warp 부분합만 shared memory에 쓴다. block barrier 한 번 후 첫 warp가 최종 합을 만든다.
폭이 32나 256의 배수가 아니어도 입력이 없는 thread는 0으로 집계에 참여한다.
shuffle의 full mask는 warp 전체가 같은 호출에 참여하는 구조에서 사용했다.

tree와 shuffle은 열 방향 입력 분배가 같다. 인접 thread가 인접 열을 읽는다.
naive와 비교하면 행 내부 병렬화와 global-memory 접근 방식도 함께 바뀐다.
따라서 속도 차이를 shared memory의 효과 하나로 분리해 설명하지 않는다.

## 정확성 검증

NPU 입력 25종을 3개 CUDA 구현에서 검사했다. 내보낸 BF16 값을 TT tile/face 순서로
다시 배치한 fingerprint가 실제 NPU 입력 fingerprint와 25종 모두 일치했다.
CUDA는 FP64 CPU 기준값과 NPU의 논리 row_sums를 함께 대조했다.

- 정수 입력은 오차 0을 요구했다. `4096,1,-4096` 상쇄 사례도 세 구현 모두 통과했다.
- 소수 입력은 `atol=1e-4`, `rtol=1e-5`를 유지했다. 최대 절대 오차는 1.9073486328125e-06이었다.
- H=1·31·32·33·65, W=1·31·32·33·257·1025·4097 등의 경계를 검사했다.
- 측정 batch마다 유효 출력을 NaN sentinel로 초기화했다. padding 행은 0으로 유지하고
  출력 전체를 검사했다. 같은 process에서 반복한 출력도 같아야 통과한다.

큰 입력 4종은 별도 CUDA 성능용 정수 입력이다. 크기는 128×320, 640×320,
320×4097, 1024×8193이며, TTSim에서 이 4종을 실행했다고 주장하지 않는다.
값은 `(row*13 + col*7) % 17 - 8`로 생성했다.

전체 실행 111개 process에서 1941회 출력 검증을 통과했다.
두 큰 입력의 재측정 18개 process에서 558회가 추가로 통과해 총 **129개 process·2499회**다.
입력 내보내기를 추가한 NPU host는 별도로 depth1/2의 50개 구성과 pilot 2회를 다시 통과했다.
NaN/Inf 입력 의미론, 임의의 FP32 동적 범위, 모든 합산 순서의 수치 안정성을 보장하는 검사는 아니다.

## 측정 방법과 첫 결과

MX150, CUDA 12.9, Windows driver 582.66에서 측정했다. kernel은 Pascal `sm_61`로 빌드했다.
20회 warmup 후 15개 sample을 수집했고 sample마다 kernel 20회를 CUDA Event로 감쌌다.
표는 batch 시간을 20으로 나눈 뒤 sample 중앙값, process 3개의 중앙값을 차례로 구한 값이다.
kernel event 시간에는 GPU scheduling과 host 제출 사이의 빈 구간이 포함될 수 있다.
순수 연산 instruction latency나 하드웨어 카운터 값으로 해석하지 않는다.

별도 pipeline 시간은 pinned H2D → kernel 1회 → D2H → stream 완료 대기다.
allocation, CPU 기준값 계산, sentinel 초기화, 결과 검증은 측정에서 제외했다.
CUDA 입력은 BF16 값을 확장한 FP32이며 padding을 포함해 전송한다.

첫 실행의 kernel 시간 단위는 ms다.

| H×W | naive | tree | shuffle |
| --- | --- | --- | --- |
| 128×320 | 0.947331 | 0.605109 | 0.713061 |
| 640×320 | 0.831235 | 3.671066 | 0.218752 |
| 320×4097 | 2.653486 | 1.494376 | 1.781642 |
| 1024×8193 | 9.813685 | 1.459605 | 9.938831 |

첫 실행은 변동이 컸다. 예를 들어 1024×8193의 tree process 중앙값은
0.771029–18.030151 ms였다. 이 표만으로 세 구현의 순위를 정하지 않는다.
시작 전 nvidia-smi 기록은 80°C, GPU utilization 46%였다. 다른 작업과의 경합이나
온도 영향을 분리 측정하지 않았으므로 변동 원인을 확정하지 않는다.

## 큰 입력 재측정

같은 실행 파일과 입력으로 두 크기를 다시 3개 process씩 측정했다.
첫 실행은 그대로 보존했고 재측정 값으로 덮어쓰지 않았다.
아래 범위는 개별 sample 범위가 아니라 process별 중앙값 3개의 최소–최대다.

| H×W | 구현 | kernel 중앙값 ms | process 중앙값 범위 ms | pipeline 중앙값 ms |
| --- | --- | --- | --- | --- |
| 320×4097 | naive | 1.574200 | 1.413781–2.504296 | 4.228600 |
| 320×4097 | tree | 0.189819 | 0.129634–1.004533 | 2.517500 |
| 320×4097 | shuffle | 0.282688 | 0.131243–1.139509 | 3.811200 |
| 1024×8193 | naive | 7.063851 | 6.699920–7.244454 | 23.995600 |
| 1024×8193 | tree | 1.137501 | 1.071406–1.172152 | 16.292300 |
| 1024×8193 | shuffle | 1.107565 | 1.034853–1.401896 | 15.727000 |

1024×8193 재측정에서는 naive 약 7.064 ms, tree 약 1.138 ms, shuffle 약 1.108 ms였다.
naive 대비 약 6.21배·6.38배였고 process 범위도 분리됐다. 이 조건에서 행 내부 병렬화의
이점은 관찰했다. tree와 shuffle의 범위는 겹치므로 shuffle이 항상 빠르다고 결론 내리지 않는다.
320×4097은 재측정에서도 tree/shuffle 변동이 컸다. 이 크기의 정확한 배율은 보류한다.
GPU clock 고정, 다른 앱 종료, driver 변경은 하지 않았다.

bank-conflict hardware counter는 수집하지 않았다. shared-memory 용량과 barrier 횟수는
소스에서 계산한 값이다. 아직 NPU multicore reduction과 RMSNorm은 구현하지 않았다.

## 재현과 코드

Windows PowerShell에서 WSL UNC 프로젝트 경로를 전달한다. UNC script가 RemoteSigned에
막히면 script만 로컬 경로로 복사해 실행한다. execution policy는 변경하지 않는다.

```powershell
$project = '\\wsl.localhost\Ubuntu-24.04\home\ysjun5656\dev\accelflow'
$runner = Join-Path $env:TEMP 'accelflow-run-cuda-reduction.ps1'
Copy-Item (Join-Path $project 'scripts\run_cuda_reduction.ps1') $runner
& $runner -ProjectRoot $project
```

- [CUDA kernel과 host 검증](../cuda/reduction/reduction.cu)
- [구현·측정 조건](../cuda/reduction/README.md)
- [Windows 실행 script](../scripts/run_cuda_reduction.ps1)
- [NPU 입력 생성·검증·내보내기](../tools/report/export_reduction_fixtures.py)
- [CUDA 실행·집계](../tools/report/benchmark_cuda_reduction.py)
- [전체 결과·원시 표본](../results/cuda_reduction/20261109T0950598741989Z/summary.json)
- [큰 입력 재측정](../results/cuda_reduction/20261109T0950598741989Z/recheck/summary.json)
- [입력 동등성 확인](../results/cuda_reduction/20261109T0950598741989Z/input-equality-audit.json)
- [최종 검증·설정 수정 기록](../results/cuda_reduction/20261109T0950598741989Z/audit.json)
- [NPU 회귀 결과](../results/reduction/20261109T094925103459Z/summary.json)
- [CUDA 12.9 shuffle 공식 문서](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#warp-shuffle-functions)

NPU는 TTSim 기능 검증, CUDA는 MX150 실측이다. 둘 사이의 실행 시간을 실제 NPU 대 GPU
성능 비교로 사용하지 않는다. 현재 NPU는 prefix 타일을, CUDA는 compact row-sum 벡터를
출력하므로 물리 전송량도 다르다. 다음 구현 후보는 NPU 코어 간 부분합 집계 또는 RMSNorm이다.

후속 RMSNorm 기준 구현과 CUDA fusion 비교는 [25번](25-rmsnorm.md)에서 완료했습니다.
