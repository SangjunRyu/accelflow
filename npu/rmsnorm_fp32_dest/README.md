# FP32-to-Dest RMSNorm 실험

기존 `../rmsnorm/`과 분리한 단일 코어·3-stage 정밀도 실험입니다.
`fp32_dest_acc_en=true`와 입력 CB별 `UnpackToDestFp32`를 사용하며 `residual_bits`는 없습니다.
제곱·제곱합·정규화의 산술은 Dest의 SFPU에서 수행합니다.

| 모드 | 설정 | 현재 고정 조합 결과 |
| --- | --- | --- |
| default | SrcA/B TF32 입력, FP32 Dest | FP32 하위 비트 손실·RMSNorm negative control 실패 |
| direct | 순정 UnpackToDestFp32 + FP32 Dest | 후행 LLK unpack의 simulator 오류 |
| compat | 위 설정 + 디렉터리 내부 LLK 호환 처리 | transport·RMSNorm·PyTorch 대조 통과 |

`compat`는 공식 upstream 수정이 아닙니다. `fp32_unpack_compat.hpp`에서 버려지는 후행
SrcA unpack만 조정합니다. 실제 입력은 full FP32 Dest로 전달하고 mantissa를 분할하지 않습니다.
실제 칩이나 다른 revision에서는 검증하지 않았습니다. 외부 TT-Metal source는 수정하지 않습니다.

```bash
bash scripts/run_rmsnorm_fp32_dest.sh
.venv/bin/python tools/report/audit_rmsnorm_fp32_dest.py --results <RESULTS>
```

실험 구분, 정확한 수치, 제한과 로그는 [31번 보고서](../../docs/31-fp32-to-dest-rmsnorm.md)에 있습니다.
코드는 host의 `compute_config`/`make_stage`에서 시작해 probe → compatibility header → square → sum → normalize로 읽으면 됩니다.
기존 BF16 fixture와 FP32 fixture를 모두 읽지만 backward·fused·multicore·실물 성능 비교는 포함하지 않습니다.

## 같은 stage 구조의 시간 비교

```bash
bash scripts/benchmark_rmsnorm_fp32_dest.sh
```

기존 split `unfused`와 새 compat를 같은 3-stage·1코어·depth2에서 비교합니다.
warmup2/sample5·process3회이며 4097열 때문에 실행에 시간이 걸립니다.
device kernel은 바꾸지 않고 두 host에 같은 opt-in timer를 추가했습니다.
측정 범위·변동·수치는 [32번](../../docs/32-rmsnorm-fp32-dest-timing.md)을 참고하세요.
