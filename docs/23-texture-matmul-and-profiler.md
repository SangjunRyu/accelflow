# 23. cudaArray 입력 경로 비교와 nvprof 진단

상태: cudaArray/texture 타일 커널 구현·MX150 비교 완료. 날짜: 2026-11-09.
[진행 순서](README.md) · 이전: [22. Reduction](22-row-wise-reduction.md)

## 어떤 비교가 가능한가

cudaArray는 행렬곱을 대신 실행하는 자료형이 아닙니다. texture fetch에 적합한 내부 배치를 가진
장치 저장 공간입니다. 현재 타일 커널의 입력을 선형 포인터로 읽을 때와 tex2D로 읽을 때를
비교하면 이 입력 경로가 우리 행렬곱에 유리한지 확인할 수 있습니다.
라이브러리 GEMM과 직접 작성한 GEMM의 비교는 별도 실험이며 cuBLAS는 이번 범위에 넣지 않았습니다.

| 구현 | 입력 저장·읽기 | shared memory / 연산 |
| --- | --- | --- |
| naive | cudaMalloc 선형 FP32 pointer | thread별 직접 FMA |
| tiled | 같은 선형 pointer | 16×16 타일·2KiB shared memory |
| texture_tiled | cudaArray + texture object + tex2D | 같은 16×16 타일·2KiB shared memory·같은 FMA 순서 |

texture에는 float channel, point filtering, element read, unnormalized coordinates를 사용했습니다.
texel 중심 좌표는 column+0.5, row+0.5입니다. 보간이나 정규화로 입력값을 바꾸지 않습니다.
입력 업로드는 cudaMemcpy2DToArrayAsync를 사용하고 출력은 기존 선형 FP32 buffer로 받습니다.
shared-memory 배치는 세 번째 경로에서도 바꾸지 않았습니다.

배열 배치, texture cache와 명령, 주소 계산 비용이 함께 바뀐 비교입니다.
속도 차이를 cudaArray 내부 swizzle 하나의 효과라고 분리해서 해석하지 않습니다.
현재 GEMM은 규칙적인 접근이므로 texture의 2D locality 이점이 항상 클 이유도 없습니다.

## 검증·측정 조건

같은 NPU BF16 fixture를 FP32로 펼친 입력을 사용했습니다. 기존 9개 fixture 중 3개 성능 shape는
각 구현을 process3회씩, 나머지 6개 정확성 fixture는 process1회씩 실행했습니다.
총 45개 process에서 2070개 출력을 검증했습니다.
정수는 FP64 기준과 exact, 소수는 기존 atol=1e-4/rtol=1e-5를 유지했습니다.
입력과 BF16으로 반올림한 출력 fingerprint도 세 구현 사이에 일치했습니다.

각 process는 warmup20회, timing sample15개이며 event sample마다 kernel20회 평균을 기록합니다.
기존과 같이 single-launch host 완료 대기, pinned H2D/D2H, H2D→kernel→D2H pipeline도 나눠 측정했습니다.
실행 순서를 섞었고 GPU clock은 고정하지 않았습니다. 프로파일러를 켠 실행은 성능 표에 섞지 않았습니다.

배열/texture object 또는 선형 buffer의 생성 비용은 resource_creation_host_ms에 따로 기록했습니다.
이는 process 전체 시작 비용이 아니고, pinned host allocation과 입력 업로드도 제외한 host 구간입니다.
pipeline은 이미 만든 자원을 사용하므로 allocation/create 비용을 포함한 단발 실행 시간은 아닙니다.

## 실제 MX150 결과

아래는 ms 단위입니다. process별 sample 중앙값을 구한 뒤 process3회의 중앙값으로 요약했습니다.
양수 시간 감소율은 texture 쪽이 짧았다는 뜻입니다. 이전 21번의 절대 시간과 섞지 않고 이번 control을 기준으로 합니다.

| M,K,N | tiled kernel | texture kernel | texture 시간 감소율 | tiled pipeline | texture pipeline |
| --- | --- | --- | --- | --- | --- |
| 128,128,128 | 0.03105 | 0.03209 | -3.35% | 0.19000 | 0.21300 |
| 320,320,320 | 0.40187 | 0.40540 | -0.88% | 0.96860 | 0.99690 |
| 640,320,640 | 1.78268 | 1.70112 | +4.58% | 3.03080 | 2.99490 |

이번 조건에서는 작은 두 입력에서 texture 경로의 대표 시간이 더 길었고 큰 입력에서는 약 5% 짧았습니다.
큰 입력의 pipeline 차이는 kernel 차이보다 작았습니다. 이미 준비된 입력을 여러 번 재사용하는지,
입력이 바뀔 때마다 업로드하는지에 따라 어떤 구간이 중요한지도 달라집니다.
cudaArray를 사용하면 직접 작성한 타일링보다 항상 빨라진다는 결론은 아닙니다.

| M,K,N | 구현 | kernel trial 중앙값 ms | H2D ms | 자원 생성 host ms |
| --- | --- | --- | --- | --- |
| 128,128,128 | tiled | 0.03089/0.03105/0.03265 | 0.05773 | 0.27170 |
| 128,128,128 | texture_tiled | 0.04049/0.03099/0.03209 | 0.06749 | 0.82250 |
| 320,320,320 | tiled | 0.40182/0.40438/0.40187 | 0.27872 | 0.36350 |
| 320,320,320 | texture_tiled | 0.46418/0.40453/0.40540 | 0.32976 | 2.77450 |
| 640,320,640 | tiled | 1.93152/1.78268/1.78156 | 0.54925 | 0.45510 |
| 640,320,640 | texture_tiled | 1.70112/1.69544/1.72513 | 0.56307 | 0.67440 |

자원 생성은 process당 한 번 측정했으므로 sample15개의 kernel 통계와 반복 수가 다릅니다.
별도로 측정한 중앙값을 더해 정확한 단발 end-to-end 시간이라고 주장하지 않습니다.

## bank conflict와의 관계

현재 shared tile의 scalar FP32 접근을 32개 bank로 모델링하면 conflict-free입니다.
warp가 ty 두 행과 tx 16개씩을 맡아 store는 연속 32개 word, A 읽기는 bank가 다른 두 주소의
broadcast, B 읽기는 16개 주소의 broadcast입니다. 같은 bank의 서로 다른 word와 구분해야 합니다.
8개 warp와 inner index 0..15를 열거한 결과를 diagnostics에 보존했습니다.
compiler가 생성한 실제 명령의 hardware counter 측정 결과를 대신하는 것은 아닙니다.

cudaArray/texture는 입력 읽기 경로를 바꾸며 shared-memory bank 배치를 자동으로 고치지 않습니다.
padding이나 XOR swizzle 실험과도 구분해야 합니다. 특히 이 16×16 block에서 stride를 16→17로
바꾸면 scalar store 모델에는 오히려 한 bank에 서로 다른 word 두 개가 겹칩니다.

## nvprof는 어디에서 실패했는가

현재 도구는 nvprof 12.9.19, GPU는 MX150, 확인한 드라이버 버전은 582.66입니다.
CUDA 12.9 공식 profiler 문서는 Pascal을 nvprof 지원 대상으로 기재합니다.
버전 문자열만 보고 MX150 미지원이라고 판단할 수는 없습니다.

| 확인 | 실제 결과 |
| --- | --- |
| 기존 CUDA executable | 정상 실행·정답 검증 통과 |
| nvprof --print-gpu-trace | kernel·memcpy trace 수집 성공 |
| nvprof --query-metrics | metric 이름 목록 조회 성공 |
| nvprof --query-events | CUDA profiling error |
| cuptiGetVersion / cuptiSubscribe | 성공 |
| cuptiActivityEnable | 성공 |
| cuptiDeviceGetNumEventDomains | CUPTI_ERROR_NOT_SUPPORTED, code 27 |
| cuptiEventGroupCreate | 같은 code 27 |

사용자의 일반 명령 프롬프트에서도 query-events 오류가 재현됐습니다.
즉 kernel 자체의 실행 실패가 아니며, legacy Event API로 카운터를 조회하는 경로에서 실패합니다.
반환값은 권한 부족을 뜻하는 CUPTI_ERROR_INSUFFICIENT_PRIVILEGES(35)가 아니었습니다.
관리자 권한이면 반드시 해결된다고 주장하지 않습니다.

NVIDIA 개발자 포럼에는 CUDA 12.9 + driver580.65.06의 V100에서 같은 API·같은 code27이
발생한 사례가 있습니다. 해당 답변은 580 드라이버 계열과 legacy Event/Metric API의 호환성 문제를
설명하며 575 계열 또는 이전을 언급합니다. 현재 582.66과 관찰 결과가 같은 방향이므로
이 호환성 문제가 유력합니다. 다만 Windows MX150에서 드라이버 변경 전후를 직접 검증한 것은 아닙니다.
metric 목록은 CUPTI 안에 있어 보이지만 domain/event 조회에는 driver 지원이 필요하다는 설명도 증상과 일치합니다.

## 드라이버 변경 판단

현재 CUDA 실행·정확성·event timing·trace는 동작하므로 이번 비교를 마무리하려고 드라이버를
내릴 필요는 없습니다. bank transaction/stall 같은 하드웨어 카운터 수집이 필수 과제가 되면
별도 작업으로 호환 드라이버를 검토할 만합니다.

포럼의 575는 그 사례의 드라이버 번호이며 Windows에 그대로 대입할 설치 지침은 아닙니다.
CUDA 12.9 GA release notes는 대응 Windows driver를 576.02 이상으로 표기하고
별도로 CUDA12.x minor-version compatibility의 더 낮은 기준도 설명합니다.
실제 후보는 MX150 notebook·Windows·CUDA12.9·WSL 조건을 함께 확인해야 합니다.
이번 작업에서 driver, NVIDIA 제어판, GPU clock, counter 접근 권한은 바꾸지 않았습니다.

현행 Nsight Compute는 Pascal을 지원하지 않아 단순한 도구 교체도 해결책이 아닙니다.
현재 문서에는 hardware bank counter를 확보하지 못했다는 제한을 남기고
정적 주소 모델과 실제 CUDA event 시간은 구분해서 보고합니다.

## 재현과 결과

기존 Windows runner에서 variants를 지정합니다. WSL UNC workspace라면 21번과 같이 runner를
로컬 임시 경로에 복사한 뒤 실행합니다. 기본 variants는 이전 naive/tiled 두 개를 유지합니다.

```powershell
$project = '\\wsl.localhost\Ubuntu-24.04\home\ysjun5656\dev\accelflow'
$runner = Join-Path $env:TEMP 'accelflow-run-cuda-matmul.ps1'
Copy-Item -LiteralPath (Join-Path $project 'scripts\run_cuda_matmul.ps1') -Destination $runner
& $runner -ProjectRoot $project -Variants naive,tiled,texture_tiled -ReuseFixturesFrom '<검증된 fixture 결과 경로>'
```

`tools/report/probe_cupti.py output.json`은 Windows에서 DLL을 읽어 진단 결과를 저장합니다.
시스템 설정은 바꾸지 않습니다. CUDA 환경별로 dll 이름·경로가 다르면 해당 경로를 확인해야 합니다.
`tools/report/summarize_texture_matmul.py RESULT_DIRECTORY`는 이번 세 구현의 비교를 생성합니다.

- [최종 성능·준비 비용 비교](../results/cuda_matmul/20261109T0916271819699Z/texture-comparison.json)
- [원시 표본·process별 결과](../results/cuda_matmul/20261109T0916271819699Z/cuda-measurements.json)
- [명령·소스/실행 파일 SHA256](../results/cuda_matmul/20261109T0916271819699Z/cuda-metadata.json)
- [CUPTI 직접 진단](../results/cuda_matmul/20261109T0916271819699Z/diagnostics/cupti-diagnostic.json)
- [nvprof trace](../results/cuda_matmul/20261109T0916271819699Z/diagnostics/nvprof-trace-only.log)
- [카운터 조회 실패](../results/cuda_matmul/20261109T0916271819699Z/diagnostics/nvprof-events.txt)
- [bank 주소 모델](../results/cuda_matmul/20261109T0916271819699Z/diagnostics/bank-model.json)
- [CUDA 12.9 profiler 지원표](https://docs.nvidia.com/cuda/archive/12.9.0/profiler-users-guide/index.html)
- [같은 Event API 오류의 NVIDIA 포럼 답변](https://forums.developer.nvidia.com/t/cupti-event-api-fails-with-error-cupti-error-not-supported-on-v100/364454)
- [CUDA 12.9 driver 대응표](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)
- [cudaArray·texture object 공식 문서](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html)

## Nsight Systems 타임라인 추가 확인

Nsight Systems 2025.1.3으로 MX150에서 `640×320×640` 입력을 실행했다.
`tiled`, `texture_tiled` 모두 GPU kernel 16회가 기록됐고, 출력 검증도 통과했다(`max_abs_error=0`).

- [tiled GUI 보고서](../results/profiling/20261109T094004Z/tiled.nsys-rep)
- [texture_tiled GUI 보고서](../results/profiling/20261109T094004Z/texture_tiled.nsys-rep)
- [열기 안내·CSV·로그·재현 명령](../results/profiling/20261109T094004Z/README.md)

설치된 Nsight Systems에서 `File > Open`으로 `.nsys-rep` 파일을 연다.
CUDA GPU의 kernel·memory 구간과 CUDA API 행에서 복사, 실행, 동기화 순서를 확인할 수 있다.
texture 경로에는 `cudaMemcpy2DToArrayAsync` 등의 호출도 기록됐다.
현재 실행은 한 stream에서 순차적으로 진행하며, 별도 NVTX 구간 표시는 넣지 않았다.

이 보고서에는 shared-memory bank-conflict 카운터가 없다. kernel 시간이 길다는 사실만으로
bank conflict가 원인이라고 판단할 수는 없다. CUDA Event completion tracing을 포함한
프로파일링 오버헤드도 있으므로, 앞의 반복 측정 성능표를 이번 짧은 수집 결과로 대체하지 않는다.
