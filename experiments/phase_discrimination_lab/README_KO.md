# KoMaP 서버 실험: 상 구별·경계·접촉부

2026-10-03. 목적은 Al3Ni–Eutectic Si 혼동과 가는 Eutectic 누락을 줄이는 것이다.
코드·실험 설정·데이터를 묶은 실행 패키지이며, 새 실험의 학습 결과는 아직 없다.
기존 RRCU C13은 11 epoch 완료 후 사용자의 요청으로 중단했고, 이 패키지에는 해당 가중치를 넣지 않았다.

## 먼저 돌릴 실험

| ID | 기준 B에서 바꾸는 것 | 핵심 가설 | 우선순위 |
|---|---|---|---|
| B | 없음: ResNet101 + bilinear U-Net + scSE | 같은 서버/코드의 기준 점수 확보 | 필수 |
| D01 | 학습용 다중 해상도 projection head + 라벨 기반 대조 loss | 비슷한 두 상의 내부 특징을 구별 | 1 |
| D02 | 마지막 half-resolution decoder 특징에 파생 BEM 추가 | 경계와 가는 조직 보존 | 2 |
| D03 | CE에서 Al3Ni–Eutectic 직접 접촉부만 가중치2 | 특정 두 상의 접촉부 혼동 감소 | 3 |
| D04 | D01 + D02 | 상 구별과 경계 강화의 상호작용 | 선택: 단독 결과 확인 후 |

기본 큐는 B → D01 → D02 → D03이며, D04는 자동 실행하지 않는다.
모든 실험은 **fresh seed42, 150 epoch**다. 기존 서버 R022의 D4 mIoU 0.7989694265는 참고 기록이며,
새 결과의 비교 상대는 반드시 이 패키지의 B다. 과거 R031은 관측 최고였지만 seed43 개선이 재현되지 않아 B recipe에 넣지 않았다.

## 공통 조건

- ImageNet V2 ResNet101, grayscale를 RGB로 반복, ImageNet 정규화.
- 원본 해상도 Context448 입력, 중앙224 감독, stride112, reflect halo.
- batch4, FP32, 420 draws/epoch, workers0, AdamW lr3e-4/weight decay1e-4, cosine150.
- 50% Al3Ni/Eutectic-rich sampling, 클래스 균등 선택, 해당 클래스 면적10% 이상 후보.
- 기존 약한 광도 증강과 D4 회전/반전 증강 유지.
- base loss = 0.5 CE + 0.25 batch-pooled weighted Dice + 0.25 per-image present-class weighted Lovasz.
- Dice/Lovasz 클래스 가중치 [1,1,2,1]. Single-view Valid로 best 선택하고, 완료 후 동일 best에 single/D4 평가.
- 평가 mIoU는 이미지별 4클래스 평균의 평균. union=0 클래스는1; strict(absent=0)도 함께 기록.

변형 모듈은 별도 CPU RNG로 초기화해 기존 encoder/decoder 가중치와 crop RNG를 보존한다.
BEM의 출력 projection은0으로 초기화되어 첫 forward가 B와 같다. 학습하며 차이가 생긴다.
새 패키지 B의 초기 state와 forward가 기존 R022 harness와 일치하는지 로컬에서 확인한 기록은 verification/에 있다.
같은 seed라도 CUDA 비결정적 연산과 하드웨어/환경 차이 때문에 과거 서버 결과와 비트 단위 재현을 보장하지 않는다.

## D01: PSCL에서 가져온 핵심과 구현 차이

원 논문: [Patch-sampled contrastive learning for dense prediction pretraining in metallographic images](https://www.nature.com/articles/s41598-025-32855-5).
원 논문은 라벨 패치와의 특징 유사도로 unlabeled 패치를 선택하고, image/global + patch/local 대조 사전학습 뒤 분할 fine-tuning을 한다.
공정 정보에 따른 global 학습과 MoCo/SimCLR 프레임워크도 사용한다.

**D01은 그 논문 전체나 PSCL-SimCLR의 정확한 재현이 아니다.** 우리 Train의 모든 픽셀 라벨을 쓰는
**supervised multiscale local contrastive auxiliary loss**로 바꿨다. 별도의 unlabeled 사전학습,
global image loss, 공정 메타데이터, teacher/queue는 사용하지 않는다. ImageNet 초기화로 시작해 segmentation과 함께 학습한다.

입력448의 decoder 특징112/224/448을 중앙으로 잘라 **56/112/224** 크기에서 감독한다.
각 특징에 1×1 Conv → ReLU → 1×1 Conv projection(64차원)을 붙인다.
GT를 one-hot area pooling하고 해당 클래스 점유율이90% 이상인 위치만 선택한다.
마지막224 수준은 GT와 같은 해상도여서 가는 Eutectic을 coarse label pooling으로 모두 지우지 않는다.
batch 안에서 클래스당 최대32점을 독립 RNG로 뽑는다. 같은 클래스는 positive, 다른 클래스는 negative이며 self-pair는 제외한다.
클래스와 사용할 수 있는 scale을 균등 평균한다. 두 클래스 이상/각 클래스 두 점 이상이 없으면 그 scale은 건너뛴다.

`L_total = L_base + lambda(epoch) × L_local`

temperature0.1, lambda는 epoch1–5=0, epoch6=0.01에서 올라 epoch10 이후0.05.
이 값들은 최초 후보 설정이며 논문의 검증된 KoMaP 최적값이 아니다.
추론은 segmentation head만 실행하므로 D01의 projection 경로는 추론 연산에 들어가지 않는다.

## D02: 파생 BEM

참고: [BE-Unet/RUCS 논문](https://www.techscience.com/cmc/v87n1/66100/html).
half-resolution decoder의64채널 특징에서 `edge=ReLU(x−minpool3×3(x))`를 계산한다.
dilation1/2/3의 depthwise3×3 + pointwise1×1 branch를 결합하고 sigmoid 경계 gate를 곱해
`x + 0.1 × gated_context` residual로 넣는다. scSE와 나머지 decoder는 그대로 둔다.
이 위치·채널·depthwise 설계와 초기화는 우리 변형이고, 논문의 BE-Unet/RUCS 전체 재현이 아니다.
추론에서도 BEM을 실행하므로 시간/메모리를 함께 비교한다.

## D03: 특정 상의 접촉부 CE

참고: [Al–Si 단층영상의 pixel-wise weighted loss 연구](https://www.nature.com/articles/s41598-019-56008-7).
GT에서 클래스1/2가 4-neighbor로 직접 맞닿은 픽셀을 찾고 반경2로 확장한다.
확장 영역 중 클래스1/2에 속한 픽셀만 가중치2, 나머지는1이다.
CE는 타일별 `sum(weight×CE)/sum(weight)` 후 batch 평균으로 바꾸고 Dice/Lovasz는 유지한다.
접촉부가 없으면 해당 타일은 보통 CE와 같다. 경계 밖 wraparound나 Al 기지 픽셀은 포함하지 않는다.
기존 R041은 모든 GT 경계를 강화해 −0.1731pp였으므로, 특정 상 접촉부에 제한한 별도 가설로 검증한다.

## 데이터

ZIP에 Data/를 포함한다: Train70 이미지+mask70, Valid20 이미지+mask20, Test10 이미지(정답 없음).
대조학습과 contact 가중치는 Train mask만 사용한다. Valid는 best 선택/평가용이다.
Test는 학습·후보 선택에 사용하지 않고, 마지막 선택 모델의 추론 명령만 별도로 제공한다.
파일명은 `*_image.png`와 대응하는 `*_mask.png`를 기대한다.

| ID | 클래스 | RGB mask |
|---|---|---|
| 0 | Primary Si | 76,178,76 |
| 1 | Al3Ni | 25,76,153 |
| 2 | Eutectic Si | 204,204,204 |
| 3 | Al | 153,127,76 |

데이터·가중치가 필요 없으면 Data/만 제외해도 `--data /path/to/Data`로 실행할 수 있다.
ImageNet weight는 ZIP에 넣지 않았고 첫 학습 때 torchvision에서 다운로드한다. 인터넷이 없는 서버는
`resnet101-cd907fc2.pth`를 `$TORCH_HOME/hub/checkpoints/`에 준비한다.
공식 URL: https://download.pytorch.org/models/resnet101-cd907fc2.pth
공식 파일을 로컬에서 확인한 SHA256: cd907fc2a0de2292b2ba2b27d2216bc672d84632d1cf20058d01c0c03b5e020e

## 서버 설치·검사

Linux NVIDIA GPU, Python3.10–3.12 환경. 아래 PyTorch2.7.1/torchvision0.22.1 CUDA12.8 조합은
[공식 설치 안내](https://pytorch.org/get-started/previous-versions/)에 맞춘 것이다. NVIDIA driver 호환성과 GPU 메모리는 서버에서 확인한다.

```bash
unzip KoMaP_Phase_Discrimination_Server_20261003.zip
cd KoMaP_Phase_Discrimination_Server_20261003
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cuda.txt
python -m pip install -r requirements.txt
export TORCH_HOME="$PWD/.torch_cache"
python run.py list
python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 python run.py check --device cuda --size 448 --batch-size 4 --data Data --report verification/server_cuda.json
```

check는 synthetic forward/backward/AdamW step과 데이터 쌍·palette 검사다. 실제 데이터 학습을 시작하지 않는다.
로컬 검사는 CPU PyTorch2.8.0/torchvision0.23.0에서 수행했으며 서버 CUDA 실행 검증을 대신하지 않는다.

## 실행: 기본 4개 순차 비교

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 nohup python -u run_suite.py --data Data --output runs/phase_seed42 > logs/phase_seed42.log 2>&1 &
```

같은 GPU에서 순차 실행한다. tmux 안에서는 nohup 없이 같은 명령을 실행해도 된다.
오류/중단 시 다음 후보를 실행하지 않는다. 다른 실행 폴더를 자동 덮어쓰지 않는다.
개별 실행과 중단 후 재개:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/D01.json --data Data --output runs/D01_only --device cuda
CUDA_VISIBLE_DEVICES=0 python run.py train --config configs/D01.json --data Data --output runs/D01_only --device cuda --resume
CUDA_VISIBLE_DEVICES=0 python run_suite.py --data Data --output runs/phase_seed42 --resume
```

resume은 마지막 **완료 epoch**부터 재개하며 model/optimizer/scheduler/crop RNG/contrast RNG를 복원한다.
진행 중 epoch의 일부 step은 다시 실행한다. 코드·데이터·config·핵심 환경이 바뀌면 재개를 거부한다.
epochs를150에서200으로 바꿔 기존150 계획 checkpoint를 resume하는 방식은 지원하지 않는다.

진행 확인 및 큐 전체 중단:

```bash
cat runs/phase_seed42/suite_status.json
tail -n 5 runs/phase_seed42/logs/D01.log
cat runs/phase_seed42/D01/status.json
```

nohup 시작 때 shell이 출력한 **run_suite.py의 PID**에 `kill -TERM PID`를 보내면 현재 child를 종료하고 나머지 큐도 취소한다.
이 PID는 suite_status.json의 supervisor_pid에도 기록된다. pid 필드는 현재 학습 child의 PID다.
기존 checkpoint는 보존한다. child PID만 종료해도 큐는 실패로 종료하지만 supervisor PID로 중단하는 방법을 권장한다.

## 결과 비교·선택

```bash
python run.py compare --runs runs/phase_seed42/B runs/phase_seed42/D01 runs/phase_seed42/D02 runs/phase_seed42/D03 --output comparisons/seed42
```

comparison.csv/json에 single/D4 mIoU, strict 점수, Al3Ni/Eutectic IoU, Eutectic precision/recall,
가는 skeleton recall, 양방향 상 혼동률, 접촉부 오류율, 추론 시간, peak CUDA 메모리를 기록한다.
paired_deltas.json에는 이미지별 기준 B와의 차이를 남긴다.
혼동률의 분모는 해당 GT 클래스 픽셀 수이며 전체 픽셀을 합산한다. 평가 mIoU의 이미지별 평균 방식과 다르다.
thin skeleton recall은 Zhang–Suen skeleton 전체에서 정확한 픽셀 일치를 센 것이며 물리 두께 임계값 지표가 아니다.
train_total_loss는 보조 항이 있어 실험 간 직접 순위 비교하지 말고 seg_loss와 validation을 함께 본다.

best.pt, last.pt, history.csv/json, config/data_manifest/environment.json, valid_single/d4.json,
valid_masks/가 각 run에 저장된다. single-view Valid best 선택 규칙을 모든 후보에서 유지한다.
전체 mIoU와 Al3Ni/Eutectic 양쪽 IoU가 좋아지는지 확인하고, 반대 방향 혼동이나 오검출 증가가 없는지도 본다.

유력 후보는 seed43에서 B와 다시 비교한다. D04는 단독 실험이 유망할 때만 추가한다.

```bash
python make_configs.py --seed 43 --epochs 150 --output configs_seed43
CUDA_VISIBLE_DEVICES=0 python run_suite.py --data Data --configs configs_seed43 --ids B D01 --output runs/phase_seed43
CUDA_VISIBLE_DEVICES=0 python run_suite.py --data Data --ids D04 --output runs/phase_combo_seed42
```

200 epoch가 필요하면 **B 포함 모두 같은200 계획으로 fresh 시작**한다.

```bash
python make_configs.py --seed 42 --epochs 200 --output configs_200
CUDA_VISIBLE_DEVICES=0 python run_suite.py --data Data --configs configs_200 --output runs/phase_200_seed42
```

중간 checkpoint 평가와 최종 Test 추론:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate --checkpoint runs/phase_seed42/D01/best.pt --data Data --split valid --d4 --output evaluations/D01_valid_d4
CUDA_VISIBLE_DEVICES=0 python run.py evaluate --checkpoint runs/phase_seed42/D01/best.pt --data Data --split test --d4 --output predictions/D01_test_d4
```

Test 명령의 D01은 예시다. 실제로 Valid 비교와 seed 재검증으로 선택한 모델 경로를 넣는다.
모델별 실험 결과를 확인하기 전부터 모든 모듈을 합치거나 최고 성능을 주장하지 않는다.
