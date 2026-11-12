# 32. RMSNorm 비트 분할과 FP32-to-Dest 실행 비용 비교

상태: **동일 3-stage·입력·depth의 반복 측정과 정확성 대조 완료**. 날짜: 2026-11-12.
[진행 순서](README.md) · 이전: [31. FP32-to-Dest 정밀도 실험](31-fp32-to-dest-rmsnorm.md)

## 비교한 구현

기존 split은 `npu/rmsnorm/`의 단일 코어 `unfused`다. squared의 hi/low 두 scan과
통계의 hi/middle/low 재구성을 사용한다. 새 경로는 `npu/rmsnorm_fp32_dest/`의 `compat`다.
입력을 FP32 Dest로 직접 전달하고 한 번 scan하며 데이터 비트 분할을 하지 않는다.
31번에서 설명한 로컬 LLK 보조 unpack 처리는 유지했다. 순정 API-only 경로의 성능 결과는 아니다.

두 경로 모두 square → sum → normalize의 **3개 프로그램**, 단일 코어, CB depth2,
FP32 저장·누적·출력이다. fused나 multicore와 비교하지 않았다.
동일한 BF16 입력·gamma를 FP32로 확장하고 epsilon도 binary의 같은 FP32 값을 사용했다.
기존 split host가 일반 FP32 fixture를 지원하지 않으므로 실제 FP32 9종의 속도는 짝 비교하지 않았다.

| 사례 | H×W | 조건 |
| --- | --- | --- |
| ones | 2×32 | 한 타일의 작은 입력, 값·gamma=1 |
| signed | 65×129 | 3개 row group과 타일 경계, 음수·비균일 gamma |
| fractional | 17×257 | BF16으로 양자화한 소수, 한 row group의 여러 타일 |
| signed wide | 1×4097 | 긴 폭 129타일, padding 행 포함 |

이 표의 입력들은 값 분포도 다르므로 shape 사이의 시간 차이를 폭 하나의 효과로 해석하지 않는다.

## 측정 구간과 검증

두 host에 같은 [pipeline_benchmark.hpp](../npu/rmsnorm_fp32_dest/pipeline_benchmark.hpp)를 연결했다.
기존 CLI는 유지하고 `--benchmark warmups samples` 접미사를 받는 선택 경로를 추가했다.
장치 kernel의 산술·dataflow 코드는 이번 측정을 위해 변경하지 않았다.

매 반복에서 squared·sum·output을 NaN sentinel로 채우고 queue 완료를 기다린 뒤 host clock을 시작한다.
세 프로그램을 각각 EnqueueMeshWorkload → Finish로 실행하고 마지막 Finish 직후 clock을 끝낸다.
stage마다 같은 clock으로 경과 시간도 수집했다. 입력 전송·sentinel 초기화·readback·CPU 검증,
allocation·fixture packing·device 초기화는 측정 밖이다. 첫 warmup에 포함될 수 있는 JIT도 측정 sample에서 제외했다.

4개 입력 × 2개 구현 × process3회로 **24개 process**를 실행했다. 각 process는 warmup2회와
측정5회다. **120개 timing sample**, warmup 포함 **168회 RMSNorm 계산·504회 프로그램 제출**이다.
trial마다 seed 20261112+trial로 입력 순서와 두 구현의 선후를 섞고 순차 실행했다.
여러 simulator process를 동시에 실행하지 않았고 JIT cache를 지우지 않았다.
CPU clock·affinity나 사용자의 다른 host 작업을 고정하지는 않았다. loadavg·meminfo를 metadata에 보존했다.

모든 반복에서 출력·padding·제곱·제곱합을 CPU 기준과 비교했다. 반복 출력은 bitwise 일치를
요구하고 process 사이 fingerprint도 비교했다. 두 구현 사이 입력·gamma fingerprint는 같아야 하며
출력은 기존 `atol=1e-4`, `rtol=1e-4`로 대조했다. 저장된 process 출력24개는 별도의 Python FP64
`math.fsum`·`sqrt` 기준과 다시 비교했다. 독립 대조 최대 절대 오차는 5.77561328985e-07였다.

## 전체 pipeline 결과

단위는 **ms**다. process마다 5개 sample의 중앙값을 구하고 세 process 중앙값을 다시 중앙값으로 요약했다.
마지막 열은 process 중앙값의 최소–최대 범위다. `split/compat`가 1보다 크면 이번 측정에서 compat 시간이 짧다.
이 비율은 **host/TTSim 실행 비용의 비율**이며 실제 NPU 가속률이 아니다.

| H×W | split 대표 ms | FP32-Dest compat 대표 ms | split/compat | compat 시간 변화 | process 중앙값 범위 split / compat |
| --- | --- | --- | --- | --- | --- |
| 2×32 | 337.497 | 140.636 | 2.40 | -58.3% | 318.161–347.460 / 136.033–141.626 |
| 65×129 | 8382.277 | 1721.148 | 4.87 | -79.5% | 8285.972–8563.647 / 1662.430–1728.810 |
| 17×257 | 7044.687 | 1039.679 | 6.78 | -85.2% | 6885.716–7537.964 / 1037.426–1049.392 |
| 1×4097 | 42969.044 | 14335.661 | 3.00 | -66.6% | 41408.970–43598.024 / 14322.205–14598.165 |

범위는 3회 process의 관측 변동이며 신뢰구간이나 모든 환경의 성능 보장이 아니다.
여기서 측정한 host/TTSim 시간을 실물 Wormhole latency·bandwidth·TOPS로 바꾸어 해석하지 않는다.

## 단계별 결과

각 stage의 sample 중앙값을 process 사이에서 다시 중앙값으로 요약했다. 단위는 ms다.
서로 다른 sample이 중앙값으로 선택될 수 있으므로 stage 중앙값의 합은 pipeline 중앙값과 정확히 같지 않을 수 있다.

| H×W | 구현 | square | sum | normalize |
| --- | --- | --- | --- | --- |
| 2×32 | split | 4.834 | 96.449 | 232.567 |
| 2×32 | FP32-Dest compat | 4.838 | 4.948 | 130.888 |
| 65×129 | split | 6.181 | 1351.454 | 7053.298 |
| 65×129 | FP32-Dest compat | 8.388 | 9.522 | 1699.198 |
| 17×257 | split | 5.347 | 2256.203 | 4811.292 |
| 17×257 | FP32-Dest compat | 6.824 | 7.505 | 1024.189 |
| 1×4097 | split | 15.530 | 12118.404 | 30352.332 |
| 1×4097 | FP32-Dest compat | 37.197 | 44.143 | 14244.011 |

이 비교는 unpack 명령 하나만 바꾼 ablation이 아니다. split의 RISC-V reader 비트 분할·재구성,
두 번의 scan과 추가 L1 operand가 사라진 반면, compat의 full FP32 Dest 전달·후행 보조 unpack
호환 처리와 transpose 뒤 SFPU 초기화 비용이 들어간다. stage별 관측값은 비용 변화가 큰 단계를
찾는 근거이며 instruction/semaphore stall counter로 원인을 분리 측정한 결과는 아니다.

## 전송·CB 모델과 수치 결과

`Ht=ceil(H/32)`, `Wt=ceil(W/32)`, `T=Ht*Wt`다. split은 제곱 타일을 두 번 읽고
compat는 한 번 읽으므로 DRAM read 요청 payload 모델이 `(5*T+Ht)*4096`에서
`(4*T+Ht)*4096` bytes로 줄어든다. DRAM write는 양쪽 모두 `(2*T+Ht)*4096` bytes다.
중간 DRAM capacity도 양쪽 모두 `(T+Ht)*4096` bytes다. 반면 L1 통계 operand는 세 조각에서
하나로 줄어 코어당 peak CB payload가 depth2에서 **52→36 KiB**다. 전체 L1 allocator 사용량은 아니다.

| H×W | split DRAM read 모델 KiB | compat DRAM read 모델 KiB | 두 출력 최대 절대 차이 |
| --- | --- | --- | --- |
| 2×32 | 24 | 20 | 0 |
| 65×129 | 312 | 252 | 0 |
| 17×257 | 184 | 148 | 4.76837158203e-07 |
| 1×4097 | 2584 | 2068 | 0 |

모델은 source의 요청 bytes·설정 용량이며 실제 memory transaction이나 cache hit counter가 아니다.
RISC-V 비트 처리와 동기화의 감소도 함께 작용하므로 DRAM read 감소율만으로 시간 차이를 설명하지 않는다.
새 경로가 더 정확해졌다는 실험도 아니다. 두 계산의 합산 순서가 다르고 허용 오차 내의 출력 차이를 보존했다.
실제 FP32 입력 전달의 비트 보존과 정밀도 한계는 [31번](31-fp32-to-dest-rmsnorm.md)을 참고한다.

## 재현

고정 TT-Metal v0.79.0/TTSim v1.11.1과 기존 toolchain을 사용한다. 자체 host 두 개만 빌드하며
전체 TT-Metal 재빌드·추가 패키지·driver 설치는 하지 않는다.

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/benchmark_rmsnorm_fp32_dest.sh
python3 tools/report/audit_rmsnorm_fp32_dest_perf.py --results <RESULTS>
```

첫 명령이 출력한 새 RESULTS 경로를 audit에 전달한다. 4097열 split 한 process는 warmup 포함
수 분이 걸릴 수 있으며 전체 비교는 더 오래 걸린다. source/executable SHA-256, 모든 실행 명령,
warmup·sample·stage 시간과 느린 결과를 보존한다.

- [summary](../results/rmsnorm_fp32_dest_perf/20261112T123736408091Z/summary.json), [모든 raw sample](../results/rmsnorm_fp32_dest_perf/20261112T123736408091Z/measurements.json)
- [명령·버전·source/executable SHA-256·loadavg](../results/rmsnorm_fp32_dest_perf/20261112T123736408091Z/metadata.json)
- [독립 FP64·중앙값·source snapshot 대조](../results/rmsnorm_fp32_dest_perf/20261112T123736408091Z/independent-audit.json)
- [실행 스크립트](../tools/report/benchmark_rmsnorm_fp32_dest.py), [독립 검사](../tools/report/audit_rmsnorm_fp32_dest_perf.py)

TT-NN 후속 문서는 33번부터 이어간다.
