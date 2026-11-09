"""Summarize CUDA measurements without treating TTSim time as physical NPU latency."""
import argparse,hashlib,json,statistics
from pathlib import Path

r=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('result',type=Path)
args=p.parse_args();out=args.result.resolve()
meta=json.loads((out/'cuda-metadata.json').read_text()); assert meta['passed']
if set(meta.get('variants',['naive','tiled']))!={'naive','tiled'}:
    raise SystemExit('Use summarize_texture_matmul.py for the texture variant comparison')
fixtures=json.loads((out/'fixtures.json').read_text());assert fixtures['passed']
summary=json.loads((out/'cuda-summary.json').read_text())
records=json.loads((out/'cuda-measurements.json').read_text())
for name,digest in meta['source_sha256'].items():assert hashlib.sha256((r/name.replace('\\','/')).read_bytes()).hexdigest()==digest,name
gpu=json.loads((out/(meta['runs'][0]['name']+'.json')).read_text())
tables=[];cross=[];cross_data=[];interpret=[]
for shape in [(128,128,128),(320,320,320),(640,320,640)]:
 rows=[x for x in summary if x['shape']==list(shape)]
 baseline=next(x['medians_ms']['kernel_event_batch_average_ms'] for x in rows if x['variant']=='naive')
 fixture=next(x for x in fixtures['fixtures'] if x['shape']==list(shape))
 npu=json.loads((out/fixture['npu_report']).read_text())['cases'][0]
 for row in rows:
  t=row['medians_ms'];trials='/'.join(f'{x:.4f}' for x in row['trial_kernel_medians_ms'])
  tables.append(f"| {','.join(map(str,shape))} | {row['variant']} | {t['kernel_event_batch_average_ms']:.5f} | {trials} | {t['host_resident_launch_finish_ms']:.5f} | {t['h2d_pinned_event_ms']:.5f} | {t['d2h_pinned_event_ms']:.5f} | {t['host_pipeline_h2d_kernel_d2h_ms']:.5f} | {baseline/t['kernel_event_batch_average_ms']:.2f} |")
 best=next(x for x in rows if x['variant']=='tiled')
 relation='짧았습니다' if best['medians_ms']['kernel_event_batch_average_ms']<baseline else '길었습니다'
 interpret.append(f"{','.join(map(str,shape))}에서는 tiled의 대표 시간이 naive보다 {relation}. tiled process 중앙값 범위는 {min(best['trial_kernel_medians_ms']):.5f}~{max(best['trial_kernel_medians_ms']):.5f}ms입니다.")
 npu_ms=statistics.median(npu['timing']['samples_ms'])
 cross.append(f"| {','.join(map(str,shape))} | {best['medians_ms']['kernel_event_batch_average_ms']:.5f} | {best['medians_ms']['host_resident_launch_finish_ms']:.5f} | {npu_ms:.3f} |")
 cross_data.append(dict(shape=shape,cuda_tiled_medians_ms=best['medians_ms'],npu_input_producer_host_median_ms=npu_ms,
  same_bf16_inputs=True,hardware_speed_ratio=None))
archived=r/'results/matmul_batched_perf/20261019T113209749850Z'
old_rows=json.loads((archived/'measurements.json').read_text())
old_summary=json.loads((archived/'summary.json').read_text())
history=[]
for shape in [(320,320,320),(640,320,640)]:
 new_fixture=next(x for x in fixtures['fixtures'] if x['shape']==list(shape))
 fresh=json.loads((out/new_fixture['npu_report']).read_text())['cases'][0]
 for row in old_rows:
  if row['shape']==list(shape):
   for key in ('input_fingerprint_a','input_fingerprint_b','output_fingerprint'):assert row['case'][key]==fresh[key],key
 for variant in ['multicore','mcast2d','mcast2d_batch4','mcast2d_batch8']:
  row=next(x for x in old_summary if x['shape']==list(shape) and x['variant']==variant)
  history.append(f"| {','.join(map(str,shape))} | {variant} | {row['median_ms']:.3f} | {row['dram_read_bytes']//1024} | {row['cb_bytes_per_core']//1024} |")
relative=out.relative_to(r)
comparison=dict(passed=True,result=str(relative),scope='CUDA hardware measurements and TTSim host costs; no hardware speed ratio',
 matched_input_and_integer_output_fingerprints=True,rows=cross_data,archived_npu_result=str(archived.relative_to(r)),
 cuda_runs=len(meta['runs']),validated_cuda_sample_outputs=meta['validated_outputs'],
 fractional_bf16_fingerprints_equal=[x['bf16_output_fingerprint_matches_npu'] for x in records if x['fixture'].startswith('fractional')])
(out/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
fractional_fixture=next(x for x in fixtures['fixtures'] if x['name']=='fractional')
fractional_npu=json.loads((out/fractional_fixture['npu_report']).read_text())['cases'][0]
fractional_cuda=max(x['max_abs_error'] for x in records if x['fixture'].startswith('fractional'))
text=f'''# 21. MatMul 결과 정리와 CUDA 실측 비교

상태: CUDA naive/shared-memory tiled 구현과 MX150 실행·정확성 검증 완료. 날짜: 2026-11-09.
[진행 순서](README.md) · 이전: [20. K 배치 multicast](20-batched-2d-multicast-matmul.md)

## 이번에 확인한 것

NPU MatMul에서 다뤘던 타일링과 데이터 재사용을 CUDA로 옮겼습니다.
같은 입력에서 단순 구현과 shared-memory 타일 구현을 비교하고, 실제 GPU 시간과
TTSim의 host 실행 비용을 구분해 기록했습니다. GPU/NPU 칩 사이의 속도 순위를 정한 실험은 아닙니다.

NPU 구현은 단일 타일 → A 재사용 → 2×2 출력 블록 → K streaming → 독립 multicore →
A multicast → 2D A/B multicast → K 배치 전송 순서로 확장했습니다.
이 중 출력 블록 재사용과 2D multicast를 하나의 커널로 결합하는 작업은 남아 있습니다.
CB depth 실험은 했지만 실제 NPU 칩의 load/compute 중첩을 측정한 것은 아닙니다.

## CUDA 구현

`cuda/matmul/matmul.cu`에 두 커널을 작성했습니다.

| 구현 | thread/block 역할 | 데이터 재사용 | 동기화 |
| --- | --- | --- | --- |
| naive | thread 하나가 출력 한 원소, block은 16×16 thread | A/B를 K loop에서 직접 읽음 | block 내부 barrier 없음 |
| tiled | 같은 출력 분할, K도 16타일 단위로 처리 | A/B를 shared memory에 모아 각각 16번 사용 | K 타일마다 로드 후·소비 후 __syncthreads |

tiled의 shared memory는 block당 2KiB이며, thread마다 FP32 누적값 한 개를 가집니다.
warp/register tiling, double buffering, Tensor Core, cuBLAS는 이번 구현에 넣지 않았습니다.
교육용 기준 구현과 타일 재사용 구현의 차이를 확인하는 범위입니다.
두 CUDA 커널은 입력을 32의 배수로 0-padding한 영역 전체를 계산합니다.
odd shape 검증은 원래 행렬 바깥의 출력이 0인지 확인하는 것이며, CUDA의 모든 guard 분기를 검증했다는 뜻은 아닙니다.

## 같은 입력을 보장한 방법

기존 NPU executable에 `--export-inputs fixture.bin` 옵션을 추가했습니다.
benchmark selector 뒤에 붙이면 실제로 생성한 padded row-major A/B의 BF16 bits를 저장합니다.
파일 형식은 magic `AFMM001` 8 bytes, M/K/N uint32 각 4 bytes, A/B BF16 배열 순서이며 x86 little-endian입니다.
CUDA 쪽에서는 이 값을 FP32로 펼쳐 읽습니다. MSVC와 libstdc++의 난수 분포 구현 차이에 기대지 않습니다.

CUDA의 input fingerprint는 host에서 NPU의 32×32 tile/16×16 face 순서로 재배열해 계산합니다.
입력 A/B의 fingerprint를 모든 실행에서 대조했습니다. 정수 사례는 CUDA 출력을 BF16으로
반올림한 fingerprint도 NPU와 같아야 통과합니다. 소수는 양쪽 모두 같은 FP64 기준값으로 오차를 검사합니다.

| 항목 | CUDA | NPU |
| --- | --- | --- |
| 입력의 수치 | BF16으로 양자화된 같은 값 | 같은 BF16 값 |
| 장치 저장 형식 | FP32 row-major | BF16 tile/face |
| 누적과 출력 | FP32 FMA, FP32 출력 | HiFi4·FP32 destination 누적, BF16 출력 |
| 출력 비교 | FP64 기준 및 host BF16 반올림 결과 | FP64 기준 |

따라서 같은 수학 문제를 풀지만 저장 bytes와 출력 형식까지 같지는 않습니다.
CUDA 입력/출력 전송은 FP32라 BF16 배열보다 원소당 bytes가 두 배입니다.
FP32→BF16 반올림은 CUDA 측정 구간 밖의 host 검사이며 GPU BF16 처리량을 측정한 것이 아닙니다.

## 환경과 측정 구간

GPU는 {gpu['gpu']}, compute capability {gpu['compute_capability'][0]}.{gpu['compute_capability'][1]},
SM {gpu['sm_count']}개, VRAM {gpu['vram_bytes']/1024**3:.2f}GiB입니다.
Windows native CUDA 12.9 / nvcc 12.9.41 / MSVC x64로 sm_61을 빌드했습니다.
드라이버 표기와 compiler 출력은 결과 폴더에 보존했습니다. 드라이버가 표시하는 CUDA 13.0은 설치된 toolkit 12.9와 구분합니다.
TTSim은 기존 WSL2 Ubuntu의 v1.11.1 / TT-Metal v0.79.0 조합입니다.

| 지표 | 측정 범위 |
| --- | --- |
| CUDA event kernel 평균 | 입력을 GPU에 둔 채 20회 launch를 event로 묶어 측정한 시간 / 20 |
| CUDA host resident | 단일 launch 직전부터 stream synchronize 반환까지 |
| H2D/D2H | pinned host memory, 같은 stream의 CUDA event로 각각 측정 |
| CUDA host pipeline | H2D A/B → 단일 kernel → D2H C → stream synchronize |
| NPU host/TTSim | 기존 workload enqueue 직전부터 Finish 반환까지, 입력/출력 전송 제외 |

CUDA 초기화, allocation, BF16 변환, CPU 기준값 계산과 출력 검증은 시간에서 제외했습니다.
kernel event 시간에는 전송이 없지만 stream의 launch 사이 빈 시간도 들어갈 수 있습니다.
event batch 평균을 단일 launch 지연 분포나 device cycle counter로 해석하지 않습니다.
host pipeline에는 이미 초기화한 버퍼와 pinned memory를 사용하므로 프로그램 시작부터 끝까지의 시간은 아닙니다.
H2D·D2H·kernel을 따로 측정한 중앙값의 합이 pipeline 중앙값과 같을 필요도 없습니다.

## CUDA 실측 결과

성능 입력 3개 × 구현 2개 × 독립 process 3회, 나머지 정확성 입력 6개 × 구현 2개를 실행했습니다.
총 {len(meta['runs'])}개 process이며 각 process는 warmup20회, sample15개입니다.
각 sample의 event 구간에서는 kernel20회, host resident와 pipeline에서는 각각 단일 kernel을 실행합니다.
각 event batch의 마지막 출력과 resident/pipeline 출력을 검사했습니다. batch 안의 모든 중간 출력을 읽었다고 주장하지 않습니다.
총 {meta['validated_outputs']}개 sample/warmup 결과를 검증했고, CUDA kernel launch는 10,500회입니다.
seed 20261109로 실행 순서를 섞었으며 GPU clock을 고정하거나 다른 GPU 작업을 차단하지 않았습니다.

아래 시간은 ms입니다. 각 process의 sample 중앙값을 구한 뒤 세 process 사이의 중앙값을 사용했습니다.
`naive/현재` 비율은 같은 MX150·같은 정밀도·같은 event 범위에서만 계산했습니다.

| M,K,N | 구현 | event kernel 평균 | Trial 중앙값 | host resident | H2D | D2H | host pipeline | naive/현재 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(tables)+'\n\n'+'\n\n'.join(interpret)+'''

이 결과만으로 다른 GPU·shape에서도 같은 비율을 보장하지 않습니다.

## NPU 결과와 함께 읽기

다음 표의 CUDA 두 열은 실제 GPU 실행이며, NPU 열은 이번 입력 내보내기 검증에서 얻은 host/TTSim 시간입니다.
NPU는 독립4코어·depth2의 process 한 번에서 warmup2/sample5를 실행한 값으로,
3개 process를 사용한 CUDA 표와 반복 조건도 다릅니다. GPU/NPU 속도 비율은 계산하지 않았습니다.

| M,K,N | CUDA tiled event ms | CUDA tiled host resident ms | NPU host/TTSim ms |
| --- | --- | --- | --- |
'''+ '\n'.join(cross)+'''

기존 NPU 최적화 비교는 2026-10-19의 동일 환경 내 비교로 보존했습니다.
이번 내보내기 입력과 과거 결과의 A/B/output fingerprint가 일치함을 확인했습니다.
현재 host/TTSim 시간은 과거 기록과 차이가 커서 과거 시간과 이번 CUDA 시간의 차이를 칩 성능으로 해석하지 않습니다.

| M,K,N | NPU 구현 | 당시 host/TTSim ms | DRAM read 모델 KiB | 코어당 CB KiB |
| --- | --- | --- | --- | --- |
'''+ '\n'.join(history)+f'''

2D 배치4/8은 Kt=10에서 신호 교환 횟수 모델을 70%/80% 줄였지만 DRAM payload는 같습니다.
CUDA tiled는 block 안에서 입력을 재사용하고, NPU multicast는 코어 사이에 입력을 전달합니다.
재사용 범위와 동기화 비용이 다르므로 __syncthreads와 semaphore를 같은 비용으로 취급하지 않습니다.
CUDA tiled의 소스상 입력 load 요청 수는 완전한 타일에서 naive의 1/16입니다.
cache·compiler·transaction 결합을 측정하지 않았으므로 실제 DRAM bytes가 1/16이라고 주장하지 않습니다.

## 정확성과 재현

deterministic, zero, identity, signed random, fractional, odd M/N, K=257 경계를 검사했습니다.
정수 입력은 FP64 기준과 오차 0을 요구합니다. 소수 CUDA FP32 출력은 atol=1e-4/rtol=1e-5,
BF16 반올림 결과는 기존 NPU와 같은 atol=0.03/rtol=0.03을 사용했습니다.
finite·padding·반복 출력 일치와 입력 fingerprint도 확인했습니다.
수치 검증을 통과한 경우에만 결과 metadata의 passed를 true로 저장합니다.

소수 33×257×65에서는 CUDA의 BF16 반올림 출력과 NPU 출력의 fingerprint가 달랐습니다.
소수 결과의 bit-exact 일치는 주장하지 않습니다. 같은 FP64 기준에 대한 최대 절대 오차는
CUDA FP32 {fractional_cuda:.8g}, NPU BF16 {fractional_npu['max_abs_error']:.8g}이며 각 허용오차를 통과했습니다.

PowerShell에서 로컬 실행 스크립트를 복사해 실행합니다. UNC 경로의 unsigned script를
RemoteSigned 정책이 차단하는 환경에서도 시스템 실행 정책을 바꾸지 않는 방법입니다.

```powershell
$project = '\\\\wsl.localhost\\Ubuntu-24.04\\home\\ysjun5656\\dev\\accelflow'
$runner = Join-Path $env:TEMP 'accelflow-run-cuda-matmul.ps1'
Copy-Item -LiteralPath (Join-Path $project 'scripts\\run_cuda_matmul.ps1') -Destination $runner
& $runner -ProjectRoot $project
```

기존 fixture를 사용해 CUDA만 다시 측정하려면 `-ReuseFixturesFrom <이전 결과 폴더>`를 지정합니다.
fixture SHA256을 다시 대조하며 새 결과 폴더를 만듭니다. 기본값은 NPU fixture도 새로 검증합니다.
native Windows build는 `%LOCALAPPDATA%\\AccelFlow\\cuda-matmul`에 두며 WSL source와 분리합니다.
기존 NPU 회귀는 WSL에서 `bash scripts/run_matmul.sh`로 실행합니다.

| 결과 | 경로 |
| --- | --- |
| 최종 CUDA 통계 | [cuda-summary.json](../{relative}/cuda-summary.json) |
| 실행 명령·SHA256·환경 | [cuda-metadata.json](../{relative}/cuda-metadata.json) |
| 입력 SHA256·NPU 검증 명령 | [fixtures.json](../{relative}/fixtures.json) |
| 비교 검사 | [comparison.json](../{relative}/comparison.json) |
| 과거 NPU 비교 | [20번](20-batched-2d-multicast-matmul.md) |

raw fixture binary는 결과 폴더의 scratch에 보관하며 Git에서는 제외합니다. JSON·로그는 남깁니다.
초기 CUDA 개발 실행과 최종 실행은 별도 폴더로 구분하고 이 문서는 최종 실행만 사용합니다.
이제 MatMul의 구현·재사용·동기화 사례와 CUDA 실측 근거를 확보했습니다.
다음 연산은 Reduction이며, 이후 RMSNorm으로 연결할 계획입니다.

## 참고한 NVIDIA 공식 문서

- [CUDA 12.9 Programming Guide: shared memory와 동기화](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html)
- [CUDA 12.9 Runtime API: event timing](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__EVENT.html)
- [CUDA 12.9 release notes: Pascal 지원](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html)
'''
(r/'docs/21-cuda-matmul-comparison.md').write_text(text)
print('COMPARISON_PASS '+str(out))
