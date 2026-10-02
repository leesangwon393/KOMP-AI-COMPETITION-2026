# KoMaP 금속 조직 분할 — 문맥 확대 모델 실험 공유

> **Validation mIoU 78.4957% (0.7849573723)**
>
> ImageNet 사전학습 ResNet34 U-Net + scSE에 **448×448 입력 / 중앙 224×224 supervision**을 적용한 단일 모델이다. 100 epoch 중 일반 검증 최고인 **98 epoch** 체크포인트를 선택하고 D4 TTA로 평가했다. **로컬 Validation 20장 결과이며 공식 Test 점수가 아니다.**

작성일: 2026-09-30. 본 문서는 `runs/context_corrected_v1`의 완료된 실험을 정리한다.

## 1. 실험 목적과 핵심 변경

Al₃Ni와 Eutectic Si의 혼동, 얇은 Eutectic 구조의 누락을 줄이기 위해 패치 주변 문맥을 추가했다.

- 원본 이미지를 축소하지 않고 **448×448 원본 해상도 패치**를 입력한다.
- 입력의 중앙 **224×224 영역만 정답과 비교하여 loss를 계산**한다.
- 중앙 영역의 각 방향에 112픽셀씩 주변 문맥을 제공한다.
- 이미지 밖으로 나가는 문맥은 `reflect padding`으로 채운다.
- 검증과 추론도 448 입력에서 중앙 224만 사용하도록 맞춘다. 이 방식을 이하 **halo 추론**이라 부른다.

이번 모델의 decoder는 bilinear upsampling을 사용한다. DySample, PixelShuffle, clDice, 경계 보조 loss, 접촉 경계 sampling을 결합한 결과는 아니다.

## 2. 최종 성능

동일한 98 epoch 체크포인트의 결과다.

| 지표 | 일반 halo 추론 | D4 halo 추론 |
|---|---:|---:|
| **mIoU** | **78.1155%** | **78.4957%** |
| Primary Si IoU | 90.3054% | 90.5476% |
| Al₃Ni IoU | 71.5965% | 71.9666% |
| Eutectic Si IoU | 56.7852% | 57.5477% |
| Al IoU | 93.7749% | 93.9211% |

D4 적용으로 같은 체크포인트의 mIoU가 **+0.3802%p** 상승했다. 여러 모델을 결합한 앙상블은 사용하지 않았다.

## 3. 같은 평가 조건의 대조군과 비교

대조군은 기존 224 입력으로 학습한 ResNet34 U-Net + scSE다. **두 모델 모두 매 epoch 동일한 halo 추론으로 검증하여 각각 best checkpoint를 선택**했다. 대조군은 83 epoch, 문맥 확대 모델은 98 epoch가 선택됐다.

| 지표 | 대조군: 224 학습 | 문맥 확대: 448 학습 | 변화 |
|---|---:|---:|---:|
| D4 mIoU | 78.1274% | **78.4957%** | **+0.3683%p** |
| Eutectic IoU | 56.1847% | **57.5477%** | **+1.3630%p** |
| Eutectic precision | 70.6233% | **71.7885%** | +1.1652%p |
| Eutectic recall | 78.9953% | **79.9756%** | +0.9803%p |
| 얇은 skeleton recall | 71.4659% | **74.2220%** | **+2.7561%p** |
| Al₃Ni → Eutectic 오류율 | **7.3844%** | 7.8434% | +0.4590%p |
| Eutectic → Al₃Ni 오류율 | 10.6425% | **9.0552%** | −1.5873%p |

**해석:** 전체 mIoU와 Eutectic precision/recall, 얇은 구조 검출률이 함께 개선됐다. 다만 Al₃Ni를 Eutectic으로 분류하는 반대 방향 오류는 증가했으므로 모든 혼동이 줄어든 것은 아니다.

기존 DySample 100 epoch의 D4 mIoU 78.3580%보다 이번 기록이 0.1377%p 높다. 다만 모델과 추론 조건이 달라 이 차이를 문맥 확대만의 효과로 해석하면 안 된다. 문맥 확대의 주 비교 대상은 위 표의 동일 halo 평가 대조군이다.

## 4. 모델 구조

| 항목 | 설정 |
|---|---|
| 입력 | Grayscale 1채널, 448×448 |
| Encoder | ImageNet 사전학습 ResNet34, 전체 fine-tuning |
| 1채널 초기화 | RGB 첫 convolution 가중치를 입력 채널 방향으로 합산 |
| Decoder | U-Net, bilinear upsampling + skip concat + Conv/BN/ReLU ×2 |
| Decoder 채널 | 256 → 128 → 64 → 64 → 32 |
| Attention | 5개 decoder block에 scSE |
| scSE 결합 | 채널 gate 출력과 공간 gate 출력 합산 |
| 최종 출력 | 1×1 convolution, 4클래스 logits |
| 학습 loss 영역 | 출력 중앙 `[112:336, 112:336]` |
| 파라미터 수 | 24,528,043 |

마지막 원본 해상도 복원 단계에는 encoder skip이 없다. 입력 크기와 중앙 supervision을 바꾼 것이며 별도의 문맥 branch를 추가하지 않았다.

## 5. 데이터·샘플링·증강

- 제공된 Train 70장 / Valid 20장 / Test 10장 분할을 유지한다. 이 문서의 점수와 예측 마스크는 Valid에 대한 결과다.
- 정답 중앙 패치 후보는 224×224, 위치 stride 112이며 마지막 이미지 경계도 포함한다.
- 매 epoch **420개 패치를 확률적으로 추출**한다. 이미지마다 정확히 6개씩 뽑는 방식은 아니다.
- 50% 확률로 중앙 패치의 Eutectic 비율이 10% 이상인 후보 pool을 선택하고, 나머지 50%는 전체 후보 pool을 선택한다. 두 pool은 겹칠 수 있다.
- pool 선택 후 원본 이미지를 균등 선택하고, 해당 이미지 안에서 후보 위치를 균등 선택한다.
- 중앙 정답 패치와 샘플링 난수 순서는 대조군과 맞추고, 문맥 모델은 같은 중앙 위치에 448 입력을 제공한다.
- 온라인 증강은 0/90/180/270° 회전과 선택적 반전이다. 영상과 정답에 같은 변환을 적용한다.
- 입력은 `/255` 후 모델 내부에서 `(x − 0.674325) / 0.229455`로 정규화한다. 이 mean/std는 기존 설정을 유지한 값이다.
- 밝기·gamma·noise·blur 증강은 적용하지 않았다.

## 6. 학습 설정

`Loss = 0.5 × Cross Entropy + 0.5 × weighted Dice`

| 항목 | 설정 |
|---|---|
| Epoch / best epoch | 100 / 98 |
| Seed | 42 |
| Batch size | 4 |
| Samples / optimizer steps per epoch | 420 / 105 |
| Optimizer | AdamW |
| Initial LR / weight decay | 3e-4 / 1e-4 |
| Scheduler | CosineAnnealingLR, T_max=100 |
| CE 클래스 가중치 | 없음 |
| Dice 클래스 가중치 | Primary Si / Al₃Ni / Eutectic Si / Al = 1 / 1 / 2 / 1 |
| 보조 loss | 없음 |
| 실행 장치·정밀도 | Apple MPS, FP32, AMP 미사용 |
| Gradient accumulation / early stopping | 사용하지 않음 |
| Checkpoint 선택 | 매 epoch 일반 halo 검증 mIoU 최고 |

Weighted Dice는 클래스별 Dice loss를 가중합한 뒤 가중치 합으로 나눈다. `config.json`에는 공통 학습 코드의 auxiliary warmup 항목이 남아 있으나 **auxiliary_weight=0**이므로 보조 loss는 적용되지 않는다. `size=224`는 중앙 정답 영역이며 실제 학습 입력은 `train_input=448`이다.

## 7. 추론 및 평가 기준

1. 원본 영상에 112픽셀 reflect padding을 적용한다.
2. 원본 좌표에서 중앙 출력 타일 224×224를 stride 112로 배치한다.
3. 각 타일마다 주변 문맥을 포함한 448×448을 모델에 입력한다.
4. softmax 확률의 중앙 224×224만 원본 위치에 누적한다.
5. 중첩 위치의 확률을 균일 평균하고 argmax로 마스크를 만든다.

D4에서는 각 입력의 회전·반전 8개 view를 예측하고 역변환한 뒤 중앙 영역의 확률을 평균한다. 체크포인트 선택은 D4가 아닌 일반 halo 검증으로 수행한다.

**mIoU 정의:** 각 이미지에서 4클래스 IoU 평균을 구한 뒤 Valid 20장을 균등 평균한다. 정답과 예측에 모두 없는 클래스(`union=0`)는 IoU=1로 처리한다. 픽셀을 전체 합산한 global mIoU나 accuracy가 아니다. 참고로 같은 D4 예측에서 absent class를 0으로 처리하면 77.2457%, 제외하면 77.8713%다. 다른 팀 결과와 비교할 때 집계 방식과 absent-class 정책을 맞춰야 한다.

Precision/recall과 방향별 혼동률은 전체 픽셀 합산 지표다. 방향별 오류율의 분모는 해당 정답 클래스의 픽셀 수다. 얇은 skeleton은 GT Eutectic의 형태학적 skeleton 중 배경까지 거리가 2픽셀 이하인 위치로 정의한다. 물리적인 두께 측정이나 연결성의 직접적인 증거는 아니다.

## 8. 이전 평가 오류와 이번 보정

초기 문맥 실험은 448 입력의 중앙224로 학습했으나 검증에는 224 입력 전체 출력을 사용했다. 그 결과 학습/추론 조건이 달라지고 부적절한 기준으로 best checkpoint가 선택됐다.

**이번 `context_corrected_v1`은 두 모델 모두 매 epoch부터 halo 검증을 적용하여 새로 학습·선택한 결과**다. 이전 `context_hrnet_v1`의 낮은 224 입력 점수나, 그 모델의 마지막 checkpoint를 사후 보정한 결과와 혼동하지 않아야 한다.

## 9. 파일 및 재현

프로젝트 루트 기준 경로다. README만 전달하면 코드·데이터·가중치는 포함되지 않으므로 재현 시 함께 공유해야 한다.

| 파일 | 내용 |
|---|---|
| `runs/context_corrected_v1/context448/best.pt` | **공유 대상 모델, epoch 98** |
| `runs/context_corrected_v1/context448/config.json` | 실제 학습 설정 |
| `runs/context_corrected_v1/context448/history.json` | epoch별 학습 loss 및 검증 점수 |
| `runs/context_corrected_v1/context448/status.json` | 완료 상태 및 일반/D4 결과 |
| `runs/context_corrected_v1/context448/d4_tta/metrics.json` | D4 상세 지표·혼동행렬 |
| `runs/context_corrected_v1/context448/d4_tta/*.png` | Valid 원본 크기 예측 마스크 |
| `runs/context_corrected_v1/context448/morphology.json` | 얇은 구조 등 형태 진단 |
| `runs/context_corrected_v1/source_snapshot/` | 실행 시 코드 사본 |
| `context_corrected_sweep.py` | 대조군·문맥 모델 순차 학습 및 보정 평가 |
| `evaluate_context_halo.py` | 중앙 영역 halo 추론 |
| `mimu/context_hrnet.py` | ContextDataset / ContextUNet |

Apple MPS 사용 가능한 환경에서 프로젝트 루트에서 실행:

```bash
python3 context_corrected_sweep.py --launch
```

이 명령은 두 모델을 순차 실행하며, 해당 출력 폴더에 완료 결과가 있으면 재학습하지 않는다. 독립 반복 실험은 별도 출력 경로를 지정하도록 실행 코드의 `OUT`을 변경해야 한다. 기존 결과를 삭제하거나 덮어쓰지 않는다.

저장된 최고 모델의 D4 평가 예시(추가 학습 없음):

```python
from pathlib import Path
import torch
from mimu.context_hrnet import ContextUNet
from mimu.data import FullResolutionDataset
from context_corrected_sweep import evaluate_halo

torch.set_num_threads(4)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
checkpoint = torch.load(
    'runs/context_corrected_v1/context448/best.pt',
    map_location='cpu', weights_only=True,
)
model = ContextUNet(pretrained=False)  # 체크포인트를 로드하므로 다운로드 불필요
model.load_state_dict(checkpoint['model'])
model.pretrained = True  # 필수: 학습 때와 동일한 입력 정규화를 활성화
model.to(device).eval()
dataset = FullResolutionDataset(Path('Data'), 'valid')
metrics = evaluate_halo(
    model, dataset, device,
    export=Path('runs/context448_team_recheck/d4_tta'), tta=True,
)
print(metrics['miou'])  # 기록값: 0.7849573722884851
```

CPU는 느릴 수 있고 장치·라이브러리 차이로 수치가 조금 달라질 수 있다. `.eval()`과 입력 정규화 활성화, 448 입력/중앙224 유지가 중요하다. 일반 224 sliding-window 평가 명령으로 대체하면 같은 조건의 재현이 아니다.

## 10. 결론과 한계

이번 설정에서는 문맥 확대가 **mIoU +0.3683%p, Eutectic IoU +1.3630%p, 얇은 skeleton recall +2.7561%p** 개선을 보였다. 점수와 얇은 구조 검출을 함께 개선한 후보라는 점에서 의미가 있다.

다만 단일 seed42 결과이고, 같은 Valid를 여러 실험 및 checkpoint 선택에 사용했다. 또한 입력 면적이 4배로 늘어 연산량과 feature/BatchNorm 통계도 달라진다. 문맥만의 인과 효과나 독립 Test에서의 일반화 개선으로 확정할 수 없다. 후속 단계는 여러 seed로 같은 대조군 비교를 반복하는 것이다.
