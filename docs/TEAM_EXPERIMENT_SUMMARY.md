**KoMaP 2026 — 로컬 모델 구조 및 실험 정리**

현재 최고 결과는 **ImageNet 사전학습 ResNet34 U-Net + decoder scSE + D4 TTA**, Validation mIoU **78.0264%**이다. TTA 전 같은 체크포인트의 점수는 **77.6722%**다. 아래 수치는 로컬 실행의 config.json, history.json, validation_metrics.json 및 D4 metrics.json에서 확인했다.

**평가 기준과 데이터**

- Train 70장, Validation 20장, Test 10장. Test 정답이 없어 공식 Test 성능은 확인하지 않았다.
- 클래스 순서: Primary Si / Al3Ni / Eutectic Si / Al.
- 점수는 원본 해상도에서 계산한 이미지별 4-class IoU 평균을 다시 20장에 대해 평균한 mIoU다. 픽셀 정답률(accuracy)과 다르다.
- 정답과 예측에 모두 없는 클래스(union=0)의 IoU를 1로 처리했다. 다른 평가 코드와 비교할 때 이 정책도 확인해야 한다.
- 각 학습의 최고 Validation mIoU 체크포인트를 선택했다. 모든 학습은 seed 42, Apple MPS에서 수행했으며 여러 seed 반복 검증은 하지 않았다.

**현재 최고 모델 구조**

입력은 grayscale 1×224×224이며, encoder는 ImageNet 사전학습 ResNet34다. encoder를 동결하지 않고 decoder와 함께 전체 학습했다. RGB 사전학습 첫 convolution의 세 입력 채널 가중치를 합산해 1-channel convolution을 초기화했다.

입력 픽셀을 0–1로 변환한 다음 `(x − 0.674325) / 0.229455`로 정규화한다. 이 mean/std는 공유받은 팀원 보고서의 값을 사용했으며, 이번 실험에서 별도로 산출한 값은 아니다.

| 경로 | 공간 크기 | 채널 수 |
|---|---:|---:|
| 입력 | 224×224 | 1 |
| ResNet stem | 112×112 | 64 |
| Encoder layer1 | 56×56 | 64 |
| Encoder layer2 | 28×28 | 128 |
| Encoder layer3 | 14×14 | 256 |
| Encoder layer4 | 7×7 | 512 |
| Decoder4 + scSE | 14×14 | 256 |
| Decoder3 + scSE | 28×28 | 128 |
| Decoder2 + scSE | 56×56 | 64 |
| Decoder1 + scSE | 112×112 | 64 |
| Decoder0 + scSE | 224×224 | 32 |
| 1×1 convolution 출력 | 224×224 | 4 logits |

Decoder4~1은 bilinear upsampling → 같은 해상도의 encoder feature와 concatenate → 두 번의 Conv3×3–BatchNorm–ReLU → scSE 순서다. Decoder0은 encoder skip 없이 원본 해상도로 upsampling한 뒤 convolution과 scSE를 적용한다.

scSE는 decoder의 5개 단계에 적용했다. 채널 attention은 global average pooling → 채널 축소(C/16) → ReLU → 채널 복원 → sigmoid, 공간 attention은 C→1의 1×1 convolution → sigmoid로 구성된다. 출력은 `x × channel_gate + x × spatial_gate`다.

- 기본 U-Net 파라미터: 24,515,524개.
- scSE U-Net 파라미터: 24,528,043개. 추가 파라미터 12,519개.
- 최종 모델에 MIMU-Net의 RRCNN이나 FPN은 추가하지 않았다. scSE 아이디어를 선택적으로 적용한 U-Net이다.
- 기존 MIMU-Net 포트의 저자 scSE는 공간 gate가 C채널이고 채널 축소 비율도 다르다. 이번 U-Net scSE는 별도 구현이므로 저자 모듈을 그대로 이식한 모델이라고 표현하면 부정확하다.

**패치 학습 방법**

원본을 통째로 축소하지 않고 원본 해상도에서 224×224 패치를 추출한다. 후보 위치 stride는 112이며 마지막 경계까지 포함한다. Train 후보 패치는 4,892개이고, Eutectic Si 면적 비율이 10% 이상인 후보는 907개다.

- 50% 확률: Eutectic Si 비율 10% 이상인 후보에서 추출.
- 50% 확률: 전체 후보에서 추출. 이 경로에서도 Eutectic이 많은 패치가 나올 수 있다.
- 각 경로에서 먼저 해당 후보가 있는 원본 이미지를 균일하게 고른 뒤 패치를 선택한다.
- epoch당 420개 패치를 복원 추출한다. `70×6`은 총 추출 횟수이며, 모든 원본에서 정확히 6개씩 뽑는 것은 아니다.
- 증강: 0/90/180/270도 회전과 선택적 좌우 반전의 8가지 D4 변환을 무작위 적용한다. 이미지와 정답 마스크에 동일한 변환을 적용한다.
- 패치를 별도 데이터셋 파일로 생성하지 않고 학습 중 추출한다.

현재 실험에는 경계 중심 sampling, 오류 heatmap 기반 adaptive sampling, 밝기·대비·gamma·noise·blur 증강을 적용하지 않았다.

**현재 U-Net 두 모델의 공통 학습 설정**

| 항목 | 설정 |
|---|---|
| Epoch / batch size | 100 / 4 |
| Samples / optimizer steps per epoch | 420 / 105 |
| Optimizer | AdamW |
| 초기 learning rate | 3e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR, T_max=100 |
| Loss | 0.5 × CE + 0.5 × weighted Dice loss |
| Dice 클래스 가중치 | Primary Si:Al3Ni:Eutectic Si:Al = 1:1:2:1 |
| CE 클래스 가중치 | 없음 |
| AMP | 사용하지 않음 |
| 최고 체크포인트 선택 | 원본 해상도 Validation mIoU |

Weighted Dice loss는 클래스별 `1−Dice`의 가중 평균이다. 즉 `(L_Primary + L_Al3Ni + 2×L_Eutectic + L_Al)/5`이며, Eutectic loss 전체를 단순히 두 배로 만드는 설정은 아니다. Dice는 배치와 공간 차원을 합산해 계산한다.

**실험 순서 및 결과**

클래스 점수는 각 실험의 최고 전체 mIoU 체크포인트 기준이다. 클래스별로 다른 epoch의 최고값을 섞지 않았다.

| 실험 | 설정 요약 | 학습 epoch | Best epoch | mIoU (%) | Eutectic Si IoU (%) |
|---|---|---:|---:|---:|---:|
| E0 | MIMU-Net, 전체 이미지를 224로 축소, Focal + 2×Dice | 150 | 59 | 66.0385 | 38.8718 |
| E1 | MIMU-Net, 원본 패치 + Eutectic sampling + Dice 클래스 가중치 2 | 150 | 150 | 76.4688 | 53.7775 |
| E2 | E1 데이터 설정, loss를 CE/Dice 0.5:0.5로 변경 | 150 | 122 | 77.2188 | 55.4515 |
| E3 | ImageNet ResNet34 U-Net 기본형 | 100 | 71 | 76.9981 | 55.2101 |
| E4 | E3에 decoder scSE 추가 | 100 | 78 | 77.6722 | 55.9854 |
| E5 | E4의 동일 최고 체크포인트에 D4 TTA | 추가 학습 없음 | 78 | **78.0264** | **56.4462** |

MIMU-Net 실험은 저자 공개 코드의 연결 구조를 PyTorch로 포팅한 모델(6,574,713 parameters)을 scratch로 학습했다. Adam, LR=1e-3, scheduler 없음, batch 4를 사용했다. 전체 이미지 축소 실험은 원본+5개 고정 기하 변형으로 420개 샘플을 구성했고, 패치 실험부터는 온라인 D4 증강을 사용했다.

E0→E1은 패치 학습, sampling, Dice 가중치, 검증 추론 방법 등이 함께 바뀌었다. E1→E2는 Focal→CE뿐 아니라 손실 조합 계수도 바뀌었다. E2→E3은 모델·사전학습·optimizer·scheduler·epoch 수가 달라 아키텍처만의 비교가 아니다. E3→E4는 동일한 학습 설정에서 scSE 유무를 비교한 단일 seed 실험이다. 다만 모듈 추가에 따른 난수 소비 차이로 공통 decoder 초기값과 실제 추출 패치 순서까지 같다고 보장한 실험은 아니다.

**scSE와 D4의 클래스별 변화**

| 클래스 | 기본 U-Net (%) | +scSE (%) | +scSE + D4 (%) |
|---|---:|---:|---:|
| Primary Si | 88.6183 | 89.3308 | 89.6374 |
| Al3Ni | 70.5770 | 71.8480 | 72.3140 |
| Eutectic Si | 55.2101 | 55.9854 | 56.4462 |
| Al | 93.5870 | 93.5246 | 93.7078 |
| **mIoU** | **76.9981** | **77.6722** | **78.0264** |

- scSE 추가: +0.6741%p.
- 같은 scSE 체크포인트에 D4 적용: +0.3542%p.
- 기본 U-Net 대비 scSE + D4: +1.0283%p.

**추론 방법**

224×224 타일, stride 112(112픽셀 overlap)로 원본 전체를 예측한다. 겹치는 위치의 클래스 확률을 균일 평균한 뒤 argmax로 최종 마스크를 만든다. D4에서는 각 타일을 8방향으로 변환해 예측하고 원래 방향으로 되돌린 확률을 평균한다. Logits 평균이나 여러 모델 앙상블이 아니라, 동일 모델의 8-view probability 평균이다. D4 결과는 현재 Validation 20장에 대해서만 평가·저장했다.

**해석과 다음 실험 후보**

scSE 추가와 D4 적용 모두 이번 Validation에서 점수를 높였다. 다만 seed 42 한 번의 학습과 반복 사용한 Valid 20장의 best checkpoint 기준이므로, 통계적으로 확정된 일반화 성능 개선이라고 단정할 수 없다.

남은 주요 병목은 Al3Ni–Eutectic Si 혼동이다. scSE 단일 추론 검증 마스크에서 정답 Al3Ni를 Eutectic Si로 예측한 픽셀은 합계 266,607개였다. 이는 픽셀 수를 합산한 오류 진단이며, 이미지별 평균인 평가 mIoU와 집계 방식이 다르다. 단순히 Eutectic 가중치만 더 올리기보다 두 클래스의 혼동을 줄이는 방향을 검토할 근거다.

다음 후보는 Train 예측 오류 기반 sampling, 512 패치의 문맥 효과, FP/FN 방향을 구분하는 loss다. 아직 실행하지 않았으며 각각 독립 비교가 필요하다.

팀원 보고서의 512 patch, 이미지당 4 patch, boundary sampling, photometric augmentation, Tversky loss 설정을 그대로 재현한 결과는 아니다. 이번 ResNet34 U-Net은 로컬에서 새로 구현했으며, 팀원 모델의 학습 checkpoint를 가져온 것이 아니다.

**주요 파일**

- 구조: `mimu/resnet34_unet.py`
- 데이터/sampling: `mimu/data.py`
- Loss: `mimu/loss.py`
- 학습: `train.py`, 실행 관리: `run_local.py`
- D4 평가: `evaluate_d4.py`
- 현재 최고 checkpoint: `runs/resnet34_unet_scse_ce_dice/best.pt` (epoch 78)
- 단일 추론 지표: `runs/resnet34_unet_scse_ce_dice/validation_metrics.json`
- D4 지표·마스크: `runs/resnet34_unet_scse_ce_dice/d4_tta/`
- 기본 U-Net 대조군: `runs/resnet34_unet_ce_dice/`
