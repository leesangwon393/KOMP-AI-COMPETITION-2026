# U²-Net Multi-Class Segmentation Experiment

> 목표: U²-Net의 RSU(Residual U-block) 구조를 이용해 금속 미세조직 **4-class semantic segmentation**을 실험한다.  
> 비교 기준: **U-Net baseline vs U²-Net**  
> 프레임워크: **PyTorch**

---

## 1. 실험 목적

기존 U-Net은 Encoder-Decoder 단계 사이에서 multi-scale feature를 추출한다.

U²-Net은 각 stage 내부에 작은 U-Net 형태의 **RSU(Residual U-block)** 를 삽입하여,
하나의 block 내부에서도 다양한 receptive field의 feature를 학습한다.

이번 실험에서는 다음을 확인한다.

1. U²-Net이 U-Net보다 **mIoU**를 개선하는가?
2. 작은 조직 / 큰 조직 등 크기가 다양한 class에서 성능 차이가 있는가?
3. 경계가 복잡한 class에서 U²-Net이 더 안정적인가?
4. 데이터가 적은 상황에서도 U²-Net의 multi-scale 구조가 유효한가?
5. 성능 향상이 특정 class에만 집중되는지 확인한다.

---

# 2. Task 정의

## Input

금속 미세조직 이미지

```text
Image
H × W × 3
```

## Output

픽셀별 4-class segmentation

예시:

```text
0 = Primary Si
1 = Al3Ni
2 = Eutectic Si
3 = Al
```

> 실제 데이터셋의 class ID가 다르면 `config.py` 또는 dataset loader에서 수정한다.

최종 출력:

```text
B × 4 × H × W
```

각 픽셀마다 4개 class logit을 출력한다.

---

# 3. 핵심 변경사항

원래 U²-Net 논문은 **Salient Object Detection(binary segmentation)** 용이다.

따라서 이번 실험에서는 Multi-Class Semantic Segmentation에 맞게 수정한다.

## 원본

```text
Output Channel = 1
Loss = Binary Cross Entropy
Activation = Sigmoid
```

## 변경

```text
Output Channel = NUM_CLASSES = 4
Loss = CrossEntropyLoss
Prediction = argmax(logits, dim=1)
```

학습 중에는 `Softmax`를 모델 마지막에 직접 넣지 않는다.

```python
logits = model(images)
loss = criterion(logits, masks)
```

추론할 때:

```python
pred = torch.argmax(logits, dim=1)
```

---

# 4. RSU 구조

U²-Net의 핵심은 **RSU(Residual U-block)** 이다.

```text
Input
  │
  ▼
3×3 Conv
  │
  ├───────────────────────────────┐
  │                               │
  ▼                               │
Encoder                            │
  ↓                               │
Downsample                         │
  ↓                               │
Downsample                         │
  ↓                               │
Dilated Conv                       │
  ↑                               │
Upsample + Skip Connection         │
  ↑                               │
Upsample + Skip Connection         │
  │                               │
  └────────────── Add ◄────────────┘
                  │
                  ▼
                Output
```

수식:

```text
H_RSU(x) = F1(x) + U(F1(x))
```

- `F1(x)` : 입력에서 얻은 local feature
- `U(F1(x))` : 내부 U-Net에서 얻은 multi-scale feature

---

# 5. 구현할 RSU Block

최소한 아래 block을 구현한다.

```text
RSU7
RSU6
RSU5
RSU4
RSU4F
```

## RSU7

```text
Input
 ↓
Conv
 ↓
Encoder 1
 ↓
Encoder 2
 ↓
Encoder 3
 ↓
Encoder 4
 ↓
Encoder 5
 ↓
Dilated Conv
 ↑
Decoder 5
 ↑
Decoder 4
 ↑
Decoder 3
 ↑
Decoder 2
 ↑
Decoder 1
 ↓
Residual Add
```

## RSU4F

RSU4F는 spatial resolution을 줄이지 않고 **dilated convolution**을 사용한다.

예:

```text
dilation = 1
dilation = 2
dilation = 4
dilation = 8
```

deep stage에서 해상도 손실을 줄이기 위해 사용한다.

---

# 6. 전체 U²-Net 구조

기본 구조는 다음과 같이 구현한다.

```text
Input
 │
 ▼
RSU7
 │
 ▼
RSU6
 │
 ▼
RSU5
 │
 ▼
RSU4
 │
 ▼
RSU4F
 │
 ▼
RSU4F
 │
 ───────── Bottleneck ─────────
 │
 ▼
RSU4F
 │
 ▼
RSU4
 │
 ▼
RSU5
 │
 ▼
RSU6
 │
 ▼
RSU7
 │
 ▼
Segmentation Head
 │
 ▼
4-Class Output
```

Encoder와 Decoder 사이에는 U-Net처럼 skip connection을 사용한다.

---

# 7. Deep Supervision

원래 U²-Net은 여러 decoder stage에서 **side output**을 생성한다.

이번 실험에서는 두 가지 버전을 비교 가능하게 만든다.

## Version A — Simple

최종 output만 loss 계산

```text
Final Output
    ↓
CrossEntropyLoss
```

초기 구현 / 디버깅 단계에서는 이 버전을 먼저 사용한다.

---

## Version B — Deep Supervision

각 decoder stage에서 segmentation map 생성

```text
Side Output 1 ─┐
Side Output 2 ─┤
Side Output 3 ─┤
Side Output 4 ─┤
Side Output 5 ─┤
Side Output 6 ─┤
Final Output ──┘
```

각 output을 원본 mask 크기로 resize한 뒤 loss를 계산한다.

```python
loss = (
    loss_final
    + w1 * loss_side1
    + w2 * loss_side2
    + w3 * loss_side3
    + w4 * loss_side4
    + w5 * loss_side5
    + w6 * loss_side6
)
```

초기값:

```text
final = 1.0
side1~side6 = 0.4
```

가중치는 config에서 수정 가능하게 구현한다.

---

# 8. 권장 프로젝트 구조

```text
u2net_segmentation/
│
├── README.md
│
├── requirements.txt
│
├── config.py
│
├── train.py
│
├── evaluate.py
│
├── inference.py
│
│
├── datasets/
│   ├── __init__.py
│   └── metallography.py
│
├── models/
│   ├── __init__.py
│   ├── blocks.py
│   ├── unet.py
│   └── u2net.py
│
├── losses/
│   ├── __init__.py
│   ├── dice.py
│   └── combined.py
│
├── metrics/
│   ├── __init__.py
│   └── segmentation.py
│
├── utils/
│   ├── seed.py
│   ├── logger.py
│   ├── checkpoint.py
│   └── visualization.py
│
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── visualizations/
│   └── logs/
│
└── data/
    ├── images/
    └── masks/
```

---

# 9. Dataset Loader

다음 형태를 지원하도록 작성한다.

```text
data/
├── images/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
│
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

image와 mask는 동일한 stem을 가진다고 가정한다.

예:

```text
images/sample_001.png
masks/sample_001.png
```

mask의 각 픽셀은 class index를 가진다.

```text
0, 1, 2, 3
```

반드시 확인:

```python
np.unique(mask)
```

---

# 10. 데이터 분할

데이터가 많지 않기 때문에 단순 hold-out 결과만 믿지 않는다.

## 1차 빠른 실험

```text
Train : 70%
Val   : 15%
Test  : 15%
```

seed 고정:

```text
42
```

## 최종 비교

가능하면 **5-Fold Cross Validation** 사용

```text
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

보고:

```text
Mean mIoU ± Std
Mean Dice ± Std
Class-wise IoU
```

---

# 11. Data Augmentation

작은 데이터셋이므로 augmentation을 사용한다.

Albumentations 권장.

## Train

```python
HorizontalFlip(p=0.5)
VerticalFlip(p=0.5)
RandomRotate90(p=0.5)
ShiftScaleRotate(...)
RandomBrightnessContrast(...)
Normalize(...)
```

단, 미세조직의 색 자체가 중요한 경우 color augmentation은 약하게 적용한다.

권장 시작점:

```text
brightness_limit = 0.1
contrast_limit   = 0.1
```

## Validation / Test

```text
Resize
Normalize
```

random augmentation 금지.

---

# 12. Input Size

1차 실험:

```text
256 × 256
```

GPU 여유가 있으면:

```text
512 × 512
```

비교 실험 시 U-Net과 U²-Net은 반드시 같은 image size를 사용한다.

---

# 13. Loss

## Experiment 1

```python
CrossEntropyLoss
```

가장 먼저 정상 동작을 확인한다.

---

## Experiment 2

```text
CrossEntropy + Dice Loss
```

권장:

```text
Loss = 0.5 × CE + 0.5 × Dice
```

class imbalance가 크면 weighted CE도 추가 실험한다.

```python
CrossEntropyLoss(weight=class_weights)
```

---

# 14. Class Weight

먼저 train mask에서 class pixel distribution을 계산한다.

```text
Class             Pixels      Ratio
Primary Si        ...
Al3Ni             ...
Eutectic Si       ...
Al                 ...
```

특정 class가 지나치게 작으면 class weighting 적용을 고려한다.

예:

```python
weight_i = 1 / log(1.02 + class_frequency_i)
```

단, 처음부터 임의 weight를 넣지 말고 baseline 결과를 먼저 확보한다.

---

# 15. Optimizer

기본:

```text
AdamW
```

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)
```

---

# 16. Scheduler

권장:

```text
CosineAnnealingLR
```

또는

```text
ReduceLROnPlateau
```

초기 실험에서는 단순성을 위해:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs
)
```

---

# 17. Training Config

초기값:

```yaml
seed: 42

model:
  name: u2net
  num_classes: 4
  deep_supervision: true

data:
  image_size: 256
  num_workers: 4

train:
  epochs: 100
  batch_size: 4
  learning_rate: 0.0001
  weight_decay: 0.0001

loss:
  name: ce_dice
  ce_weight: 0.5
  dice_weight: 0.5

early_stopping:
  patience: 20

checkpoint:
  monitor: val_miou
  mode: max
```

---

# 18. Metrics

Accuracy만 사용하지 않는다.

필수:

```text
mIoU
Class-wise IoU
Mean Dice
Class-wise Dice
Pixel Accuracy
```

## IoU

class `c`:

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
```

## mIoU

```text
mIoU = mean(IoU_1, IoU_2, ..., IoU_C)
```

## Dice

```text
Dice_c = 2TP_c / (2TP_c + FP_c + FN_c)
```

---

# 19. 반드시 저장할 결과

매 epoch:

```text
train_loss
val_loss
val_mIoU
val_Dice
learning_rate
```

best model:

```text
outputs/checkpoints/best_u2net.pth
```

기준:

```text
highest validation mIoU
```

---

# 20. Visualization

최종 성능 숫자만 보지 말고 prediction을 저장한다.

한 화면에:

```text
[Original Image]
[Ground Truth]
[U-Net Prediction]
[U²-Net Prediction]
```

특히 다음 sample을 별도로 확인한다.

```text
1. 작은 조직이 많은 sample
2. 큰 조직이 많은 sample
3. 경계가 복잡한 sample
4. U-Net이 실패한 sample
5. U²-Net이 실패한 sample
```

---

# 21. Error Analysis

최종적으로 다음 질문에 답한다.

## Case 1

```text
어떤 class에서 U²-Net이 U-Net보다 좋아졌는가?
```

## Case 2

```text
작은 object에서 개선되는가?
```

## Case 3

```text
큰 영역 segmentation은 어떻게 변하는가?
```

## Case 4

```text
경계 부분의 오류가 감소하는가?
```

## Case 5

```text
U²-Net이 오히려 실패하는 조건은 무엇인가?
```

---

# 22. 실험 순서

한 번에 모든 기능을 구현하지 않는다.

## Phase 1 — Dataset 확인

```text
[ ] 이미지/마스크 매칭 확인
[ ] mask unique value 확인
[ ] class distribution 확인
[ ] train/val split 확인
```

---

## Phase 2 — U-Net Baseline

```text
[ ] U-Net 구현
[ ] 4-class output 확인
[ ] CE Loss
[ ] mIoU 계산
[ ] prediction visualization
```

먼저 U-Net이 정상 학습되는 것을 확인한다.

---

## Phase 3 — RSU Unit Test

각 RSU block을 독립적으로 테스트한다.

예:

```python
x = torch.randn(2, 64, 256, 256)

block = RSU7(
    in_ch=64,
    mid_ch=32,
    out_ch=64
)

y = block(x)

print(y.shape)
```

Expected:

```text
torch.Size([2, 64, 256, 256])
```

RSU block은 spatial resolution을 유지해야 한다.

---

## Phase 4 — U²-Net

```text
[ ] RSU7
[ ] RSU6
[ ] RSU5
[ ] RSU4
[ ] RSU4F
[ ] Encoder
[ ] Decoder
[ ] segmentation head
```

먼저 deep supervision 없이 구현한다.

---

## Phase 5 — Deep Supervision

기본 U²-Net 학습이 정상적으로 돌아간 후 추가한다.

```text
[ ] side outputs
[ ] resize
[ ] multi-output loss
```

---

## Phase 6 — Model Comparison

동일 조건에서:

```text
U-Net
vs
U²-Net
```

동일해야 하는 조건:

```text
train/val split
input size
augmentation
epochs
optimizer
learning rate
loss
seed
```

---

# 23. 실험표

최종적으로 아래 표를 자동 생성할 수 있게 한다.

| Model | mIoU | Dice | Primary Si IoU | Al3Ni IoU | Eutectic Si IoU | Al IoU |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | | | | | | |
| U²-Net | | | | | | |
| U²-Net + Deep Supervision | | | | | | |

---

# 24. Ablation Study

시간이 허용되면 아래 순서로 진행한다.

## A

```text
U-Net
```

## B

```text
U²-Net
```

## C

```text
U²-Net + Deep Supervision
```

## D

```text
U²-Net + CE + Dice
```

Optional:

```text
U²-Net + Class Weight
```

---

# 25. 빠른 sanity check

본 학습 전에 **2~4개 이미지에 overfitting** 되는지 확인한다.

예:

```text
Train samples = 4
Epoch = 200
```

정상 구현이면 train loss가 크게 감소하고 training mask를 거의 외워야 한다.

이 테스트가 실패하면 아래를 먼저 점검한다.

```text
mask label
output channel
loss input shape
image-mask alignment
augmentation
interpolation
```

특히 mask resize에는 반드시 nearest interpolation을 사용한다.

```python
cv2.INTER_NEAREST
```

bilinear interpolation을 사용하면 class label이 깨질 수 있다.

---

# 26. 실행 CLI

예시:

## Install

```bash
pip install -r requirements.txt
```

## Dataset Check

```bash
python scripts/check_dataset.py \
    --image_dir data/images \
    --mask_dir data/masks
```

## Train U-Net

```bash
python train.py \
    --model unet \
    --num_classes 4 \
    --image_size 256 \
    --batch_size 4 \
    --epochs 100
```

## Train U²-Net

```bash
python train.py \
    --model u2net \
    --num_classes 4 \
    --image_size 256 \
    --batch_size 4 \
    --epochs 100
```

## Deep Supervision

```bash
python train.py \
    --model u2net \
    --num_classes 4 \
    --deep_supervision \
    --image_size 256 \
    --batch_size 4 \
    --epochs 100
```

## Evaluation

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/best_u2net.pth
```

---

# 27. requirements.txt

최소:

```text
torch
torchvision
numpy
pandas
opencv-python
albumentations
scikit-learn
matplotlib
tqdm
PyYAML
```

Optional:

```text
tensorboard
```

---

# 28. Codex 구현 요구사항

Codex는 아래 순서로 구현한다.

```text
1. 현재 repository의 데이터 구조를 먼저 확인한다.
2. 이미지와 mask의 실제 경로 / 확장자 / mask label을 확인한다.
3. 기존 코드가 존재하면 가능한 한 재사용한다.
4. U-Net baseline을 먼저 정상 동작시킨다.
5. RSU7/6/5/4/4F를 독립 module로 구현한다.
6. 각 RSU에 shape unit test를 작성한다.
7. U²-Net 전체 모델을 구현한다.
8. num_classes를 하드코딩하지 않고 config로 받는다.
9. CrossEntropyLoss 기준으로 먼저 학습한다.
10. validation mIoU 기준 best checkpoint를 저장한다.
11. class-wise IoU를 반드시 출력한다.
12. prediction visualization을 저장한다.
13. 이후 deep supervision을 추가한다.
14. U-Net과 U²-Net을 같은 조건으로 비교한다.
15. 최종 결과를 CSV로 저장한다.
```

---

# 29. 코드 작성 원칙

## Shape assertion

중요 stage마다 shape 확인이 가능하도록 한다.

```python
assert output.shape[-2:] == input.shape[-2:]
```

---

## No hardcoding

금지:

```python
num_classes = 4
image_size = 256
```

직접 박아두지 말고 config/argument에서 받는다.

---

## Reproducibility

다음 seed를 모두 고정한다.

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

가능하면 deterministic option도 제공한다.

---

## Device

자동 감지:

```text
CUDA
MPS
CPU
```

우선순위 예:

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

---

# 30. 최종 산출물

Codex는 작업 완료 후 최소 아래 파일들을 제공해야 한다.

```text
models/u2net.py
models/unet.py
datasets/metallography.py
train.py
evaluate.py
inference.py
metrics/segmentation.py
utils/visualization.py
config.py
requirements.txt
README.md
```

결과:

```text
outputs/
├── checkpoints/
│   ├── best_unet.pth
│   └── best_u2net.pth
│
├── predictions/
├── visualizations/
│
└── results.csv
```

---

# 31. Definition of Done

다음 조건을 모두 만족하면 1차 실험 완료로 본다.

```text
[ ] Dataset loader 정상 동작
[ ] Mask class 확인
[ ] U-Net 학습 성공
[ ] U²-Net forward 성공
[ ] RSU unit test 성공
[ ] Train/Validation loss 저장
[ ] mIoU 계산
[ ] Class-wise IoU 계산
[ ] Best checkpoint 저장
[ ] Prediction visualization 저장
[ ] U-Net vs U²-Net 동일 조건 비교
[ ] 결과 CSV 생성
```

---

# 32. 최종적으로 보고할 내용

최종 보고서에서는 단순히

```text
U²-Net이 더 좋았다.
```

라고 끝내지 않는다.

아래 형태로 분석한다.

```text
1. 전체 mIoU 변화
2. class별 IoU 변화
3. 작은 조직에서의 차이
4. 큰 조직에서의 차이
5. 복잡한 경계에서의 차이
6. 대표 성공 사례
7. 대표 실패 사례
8. 모델 크기 / 추론시간 차이
```

핵심 질문:

> U²-Net의 nested multi-scale feature extraction이 실제 금속 미세조직 segmentation에서 어떤 class와 어떤 형태의 영역에 효과가 있었는가?

이 질문에 답할 수 있도록 실험을 설계한다.
