# 09. Vector Add 설계·성능 구조 평가

2026-10-06. [진행 순서](README.md) · 이전: [08. 자체 구현·정확성](08-vector-add.md)

**판정: 테스트한 입력에서 데이터 이동·타일 계산·패딩·CB 슬롯 재사용이 설계대로 동작했습니다.**
타일 전송은 원소별 전송보다 API 호출을 묶을 수 있지만, 실제 속도 향상은 아직 입증하지 않았습니다.
CB depth 2도 기능적으로 검증했으나 DRAM traffic 감소나 temporal overlap을 증명한 것은 아닙니다.

## 증거의 구분

| 증거 종류 | 이번에 확보한 항목 | 해석 |
| --- | --- | --- |
| 실제 실행 관측 | CPU 비교, 패딩 결과, exit code, timeout 내 완료 | 검증한 입력의 기능·진행 보장 |
| runtime API 관측 | page/aligned page/aligned size/per-bank size | 고정 runtime의 버퍼 정렬 규칙 |
| 동일 조건 실험 | 같은 BF16 입력 fingerprint, CB depth 1/2 모두 PASS | 용량 변화에도 기능 유지 |
| 코드 기반 계산 | 전송 API 호출·barrier 호출·payload·padding 비율 | 실행 구조의 예상 비용 |
| 아직 미측정 | 실물 latency/bandwidth, packet 수, stall, 시간상 overlap | 실물 성능 판정 불가 |

관측값과 계산값을 profiler counter처럼 섞어 쓰지 않습니다. Simulator stdout의 초·KHz도
실제 chip latency가 아닙니다. 같은 입력 여부는 같은 seed/generation 경로 및 FNV-1a fingerprint로 확인했습니다.
fingerprint는 입력 동일성 확인용이며 보안 해시는 아닙니다. 소스 provenance는 별도 SHA-256입니다.

## 설계 의도별 판정

| 의도 | 검증 방법 | 결과와 범위 |
| --- | --- | --- |
| A+B 계산과 결과 회수 | uint32 exact / BF16 CPU 비교 | 총 38개 테스트 통과 |
| reader → compute → writer 연결 | 실제 세 커널 실행·전체 출력 검사 | 1, 2, 3, 9, 65타일에서 PASS |
| 작은 CB 슬롯의 재사용 | depth 1과 2로 각각 65타일 처리 | 슬롯 재사용을 요구하는 길이에서 PASS |
| tail의 안전한 처리 | 1025, 2051, 8193, 65537 및 모든 padding 결과 검사 | PASS |
| local memory budget 선택 | depth만 바꾼 동일 입력 실험 | CB 구성 용량 6144 → 12288 bytes |
| depth 2에서 reader/compute overlap | 별도 timeline 없음 | 미입증 |
| 실제 성능·stall 감소 | 실물/cycle-accurate 측정 없음 | 미입증 |

65타일은 depth 2보다 훨씬 크므로 모든 타일을 한 번에 CB에 보관할 수 없습니다.
검증 성공은 테스트한 실행에서 소비 후 슬롯이 재사용되고 데이터가 훼손되지 않았다는 증거입니다.
reader가 실제로 몇 번 full-buffer 대기했는지, 모든 가능한 interleaving에서도 안전한지는
관측하지 않았습니다. 여기서 모든 scheduling에 대한 일반 증명으로 확대하지 않습니다.

## 구조상 무엇이 달라졌나

N개 원소와 `T = ceil(N/1024)` 타일에 대한 소스 기반 호출 수입니다.
아래 수치는 high-level API 호출 수이며 실제 NoC packet 수가 아닙니다.

| 항목 | uint32 RISC-V 원소별 baseline | BF16 tile pipeline |
| --- | --- | --- |
| 연산 위치 | data-movement RISC-V | Tensix tile engine |
| read transfer API | 2N | 2T |
| write transfer API | N | T |
| NoC read barrier | N | T |
| NoC write barrier | N | T |
| 별도 RISC-V store visibility check | N | compute pack/CB handoff 구조 |
| reader CB reserve/push | 없음 | 4T |
| compute CB wait/pop/reserve/push | 없음 | 6T |
| writer CB wait/pop | 없음 | 2T |
| tile register acquire/commit/wait/release | 없음 | 4T |
| DRAM payload | 12N bytes | 6144T bytes |

N=1024일 때 transfer API 호출은 원소별 3072회와 tile 3회입니다.
이는 호출 단위가 커져 제어 비용을 분산할 수 있다는 **구조적 근거**입니다.
"1024배 빠르다"는 뜻이 아닙니다. 새 CB/register 동기화 비용도 존재합니다.
작은 전송과 큰 전송이 실제 chip에서 어떻게 packet으로 분할되는지도 측정하지 않았습니다.

또한 uint32는 4 bytes, BF16은 2 bytes입니다. 위 payload 차이의 절반은 자료형 차이에서
나오므로 이를 타일링만으로 얻은 개선으로 주장하지 않습니다. Vector Add는 각 입력을
한 번 읽고 출력을 한 번 쓰는 연산이므로, 현재 pipeline이 입력 재사용을 통해 DRAM load를 줄인 것은 아닙니다.

## 메모리 정렬 — 실제 API에서 확인한 값

N=1024의 입력 DRAM buffer 하나에서 확인한 값입니다.

| 항목 | uint32 baseline | BF16 tile |
| --- | --- | --- |
| logical bytes | 4096 | 2048 |
| page bytes | 4 | 2048 |
| aligned page bytes | **32** | **2048** |
| aligned size bytes | **32768** | **2048** |
| aligned size per bank bytes | 2752 | 2048 |

baseline은 4-byte page가 32-byte stride로 정렬되어 page 정렬 공간이 논리 데이터의 8배입니다.
L1 scratch 하나도 logical 4 bytes / aligned 16 bytes로 확인했습니다.
따라서 기존의 "L1 12 bytes"는 세 scratch의 논리 데이터 크기이며 전체 실제 L1 점유량이 아닙니다.

`aligned_size()`는 고정 `buffer.cpp`에서 `num_dev_pages × aligned_page_size`로 정의됩니다.
bank별 예약 공간과 allocator bookkeeping까지 합친 전체 DRAM 사용량을 의미하지 않습니다.
**32-byte 정렬 공간이 있다고 실제 read payload도 32 bytes라는 뜻은 아닙니다.**
물리 transaction/overfetch는 별도 측정이 필요합니다.

이 관측은 baseline의 작은 page 단위가 메모리 정렬 비용을 크게 만든다는 근거입니다.
동일 uint32 자료형을 유지한 block 전송 실험이 다음의 공정한 비교 대상이 될 수 있습니다.

## padding은 작거나 경계에 걸린 벡터에서 비싸다

BF16의 유효 원소 비율은 `N/(1024T)`입니다. 최소 useful payload는 `6N`, 실제 요청 모델은 `6144T`입니다.

| N | T | 유효 원소 비율 | padding 포함 payload / useful payload |
| --- | --- | --- | --- |
| 4 | 1 | 0.391% | 256배 |
| 1024 | 1 | 100% | 1배 |
| 1025 | 2 | 50.049% | 약 1.998배 |
| 2051 | 3 | 66.764% | 약 1.498배 |
| 65537 | 65 | 98.463% | 약 1.016배 |

타일링은 언제나 유리한 구조가 아닙니다. N=4도 한 타일 전체를 이동·계산하므로 대부분이 padding입니다.
연산 집약도 모델도 useful adds 기준으로 `N/(6144T)`입니다. 실물에서는 작은 입력의 dispatch/setup 비용도
고려해야 하지만 이 비용의 수치는 측정하지 않았습니다.

## CB depth 1 대 2 — 공정한 동일 자료형 실험

두 실행은 BF16, 같은 입력, 같은 타일 수, 같은 커널, 같은 core를 사용하며 depth만 바꿨습니다.

| 항목 | Depth 1 | Depth 2 |
| --- | --- | --- |
| CB 구성 용량(3개 합계) | 6144 bytes | 12288 bytes |
| BF16 검증 케이스 | 14 PASS | 14 PASS |
| 최대 스트레스 길이 | 65537 / 65타일 | 동일 |
| DRAM payload·API 호출 모델 | 동일 | 동일 |
| 시간상 overlap·stall·speedup | 미측정 | 미측정 |

depth 2의 의도는 다음 타일을 보관할 여유를 주어 producer/consumer를 덜 강하게 묶는 것입니다.
이번 결과로 말할 수 있는 것은 **메모리 비용이 6KiB 늘고, 기능은 두 설정에서 모두 유지된다**는 점입니다.
depth 2가 더 빠른지는 reader/compute/writer의 실제 실행 시간·stall·overlap을 측정해야 판단할 수 있습니다.

## 증거와 재현

```bash
cd /home/ysjun5656/dev/accelflow
bash scripts/run_vector_add.sh
```

현재 runner는 uint32 10개, BF16 depth 1·2 각각 14개를 실행합니다.
모든 실행은 exit code 0이며, depth 실험의 입력 fingerprint·runtime buffer 정보·numerical error는 JSON에 기록됩니다.

- [설계 평가 JSON](../results/vector_add/20261006T105510834955Z/design-assessment.json)
- [자동 생성 구조 표](../results/vector_add/20261006T105510834955Z/design-assessment.md)
- [uint32 결과](../results/vector_add/20261006T105510834955Z/riscv.json)
- [BF16 depth 1 결과](../results/vector_add/20261006T105510834955Z/tiled_depth1.json)
- [BF16 depth 2 결과](../results/vector_add/20261006T105510834955Z/tiled.json)
- [실행 메타데이터](../results/vector_add/20261006T105510834955Z/metadata.json)

입력 케이스 추가로 난수 소비 순서가 달라져 이번 소수 입력은 이전 08번 실행과 다른 데이터입니다.
이번 max absolute error는 0.00439453125로, 두 depth 모두 명시한 tolerance를 통과했습니다.
오차 변화가 depth 변경에 따른 성능 효과를 뜻하지 않습니다.

## 다음에 실제 chip으로 확인할 질문

같은 BF16 구현에서 depth 1/2/4를 비교하고, 타일 정렬 크기와 tail 입력을 나눠 측정합니다.
kernel timeline, DRAM traffic/bandwidth, reader/compute/writer stall, L1 점유량을 확보하면
지금의 구조적 가설을 실물 성능 근거로 이어갈 수 있습니다. CUDA track은 별도로 실제 GPU timing을 기록합니다.

후속 완료: [10. MatMul 구현·재사용 평가](10-matmul.md)에서 동일 입력을 사용한 baseline/reuse_a 비교와 48개 검증을 완료했습니다.
