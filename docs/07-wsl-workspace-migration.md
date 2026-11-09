# 07. WSL 작업 공간 이동

2026-10-06 사용자 요청에 따라 프로젝트 작업 공간을 Windows 파일시스템에서
WSL Ubuntu의 Linux 파일시스템으로 이동했습니다.
[진행 순서](README.md) · 이전: [06. 예제 빌드·검증](06-tt-metal-build-validation.md)

| 항목 | 경로 |
| --- | --- |
| 이전 프로젝트 | `C:\Users\ysjun5656\AccelFlow` |
| 새 프로젝트 | `/home/ysjun5656/dev/accelflow` |
| Windows 접근 경로 | `\\wsl.localhost\Ubuntu-24.04\home\ysjun5656\dev\accelflow` |
| TT-Metal 외부 checkout | `/home/ysjun5656/dev/tenstorrent/tt-metal-v0.79.0` |
| TTSim 바이너리 | `/home/ysjun5656/dev/tenstorrent/sim/libttsim_wh.so` |

숨김 파일·빈 폴더·기존 로그를 포함해 복사하고, 파일별 SHA-256과 폴더 목록을 대조합니다.
새 프로젝트 경로에서 공식 예제를 재검증한 뒤 Windows 원본 프로젝트 파일을 정리합니다.
이동 결과와 검증 기록은 `results/setup/workspace-migration.json`에 저장합니다.
과거 로그·메타데이터의 Windows 절대 경로는 당시 실행 기록이므로 유지합니다.

이동 후 IDE에서 새 WSL 프로젝트 폴더를 다시 엽니다. 기존 탭은 이전 파일 경로를
가리킬 수 있으므로 새 폴더에서 문서를 열어야 합니다.
Windows PowerShell용 `check_host_resources.ps1`은 Windows 측 PowerShell에서 실행합니다.

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/verify_ttsim.sh
```

이동에 따른 성능 향상 수치는 측정하지 않았습니다. TT-Metal 소스와 빌드 산출물은 이미
WSL 안에 있었습니다. 앞으로 AccelFlow 커널·테스트·로그도 Linux 파일시스템을 사용합니다.
Ubuntu VHDX의 backing volume은 C:이므로 저장 공간은 계속 C:에서 소비됩니다.

## 이동 결과

184개 파일과 41개 폴더의 복사·SHA-256 검증을 완료했고, Windows 원본 파일을 정리했습니다.
새 경로에서 공식 예제가 결과 21 / exit code 0으로 통과했습니다.
[전체 실행 로그](../results/setup/metal-smoke-20261006T091940Z.log).
