# U-Net 후속 실험 — 2026-10-04

사용자의 실행 요청으로 R022-aligned ResNet101 U-Net/scSE를 유지하며 13개 fresh run을 준비했다.
기존 RRCU의 저장 checkpoint를 같은 source_snapshot에서 재개하고, R01/R02/R03/B 완료와
동일 환경 비교가 끝난 뒤 새 실험을 한 GPU에서 순차 실행한다. 다른 환경의 과거 B 점수를 대입하지 않는다.

| 순서 | seed | ID | 변경 |
|---|---|---|---|
| 1 | 43 | B / D03 | 접촉부 CE 가중2, radius2 재현 |
| 2 | 44 | B / D03 | 같은 변경의 추가 seed 재현 |
| 3 | 42 | B / D03 | 이번 코드·환경의 기준 및 세 번째 재현 쌍 |
| 4 | 42 | X01_FREQ | OS4→OS2 얕은 skip에 축소 FreqFusion 파생 구성 |
| 5 | 42 | X02_CONN | 4상×8방향 연결 보조 head, 가중0.05 |
| 6 | 42 | X03_PIXEL | 마지막 업샘플링만 PixelShuffle |
| 7 | 42 | L01_DICE35 / L02_LOVASZ35 | CE0.5 고정, Dice/Lovasz0.35/0.15 또는0.15/0.35 |
| 8 | 42 | O01_LR2 / O02_LR4 | AdamW global LR2e-4 또는4e-4 |

150epoch, FP32, batch4, Context448/중앙224/stride112, Train70/Valid20,
420draws/epoch, 기존 sampling/photometric/D4 augmentation, cosine150.
Single Valid로 best 선택 후 같은 checkpoint의 single/D4/strict 평가.
클래스 IoU, Eutectic precision/recall/skeleton recall, 접촉부 오류, 상 혼동,
이미지별 paired bootstrap, 추론 시간과 CUDA peak memory를 보존한다.
MPS peak memory는 현재 평가기가 계측하지 않으며 null로 기록한다.

새 모듈은 별도 CPU RNG에서 초기화해 B와 공통 가중치와 crop RNG를 유지한다.
연결 head는 Train 중앙 GT 안 이웃만 감독하며 class0 Primary Si도 포함한다.
훈련 시에만 head를 사용하고 D4에서는 네 상 segmentation 출력만 평균한다.
연결 loss는 class/direction과 유효 positive/negative의 균형 평균이다.
DconnNet 아이디어에 따른 단순 보조 감독이며 원 논문의 방향 module/voting 전체 재현이 아니다.

X01은32채널 압축, ALPF5/AHPF3, Hamming-normalized kernel, low-res adaptive reassembly×2와
high-res high-frequency residual을 쓴다. offset, iterative compressed initialization은 제외한다.
Published FreqFusion 전체 재현이 아닌 축소 파생 구성이다. 저자 공개 CARAFE fallback의
group1 연산과 같은 filtering 값을 독립 구현했으며 shift accumulation과 checkpointing으로
unfold 메모리를 줄인다. 실제 속도/품질은 실험 대상이다.
출처: https://arxiv.org/abs/2408.12879 / https://github.com/Linwei-Chen/FreqFusion
연결 아이디어: https://arxiv.org/abs/2304.00145

본 실험의 loss/LR 차이는 comparison 파일의 configured_loss/configured_lr와 plan.json에 명시한다.
초기 실행 검사의 loss는 성능 결과가 아니다. 관측 점수의 작은 상승은 추가 seed에서 재확인한다.
새 모듈/D03 결합은 이 큐에 포함하지 않고 개별 기여 확인 후 진행한다.

```bash
python3 -m unittest discover -s tests -v
python3 run.py check --device cpu --size 64 --batch-size 2 --report verification/synthetic_cpu.json
TORCH_HOME=/path/to/cache python3 preflight_followup.py --device mps --data /path/to/Data --report verification/mps.json
python3 launch_followup.py --output runs/followup_20261004 --data /path/to/Data --cache /path/to/cache --prerequisite /path/to/rrcu_scse_mps_20261003 --preflight verification/mps.json
```

실행중 상태: runs/followup_20261004/trials/status.json. 각 seed의 suite_status.json과 각 run의
status.json/history.csv/logs를 함께 확인한다. Snapshot과 source/config/data hash를 보존한다.
중단할 때 launch.json의 supervisor PID가 해당 command인지 확인하고 SIGTERM을 보내면
현재 run을 종료하고 마지막 완료 epoch를 보존하며 남은 큐를 취소한다.
재개는 snapshot의 experiment_queue.py에 동일 인자와 --resume을 사용한다.
서버에서는 --device cuda를 지정하고 모든 run을 그 서버에서 새로 비교한다.
일반 CPU는 검증용이며 150epoch 본 학습 실행을 권장하지 않는다.
