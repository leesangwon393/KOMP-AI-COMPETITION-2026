# KoMaP 후속 후보 — 서버 실행 패키지

기존 79.5% 초과 후보를 발전시키는 비교 코드다. 기본 실행은 **B/D03의 seed43·44 재현 검증**이다.
구조 비교와 접촉부 loss 결합은 별도 stage로 선택한다. 각 run은 fresh ImageNet 초기화이며 150epoch다.
코드·Train70/Valid20/Test10 데이터·실험 보고·실행 검증을 함께 담았다. 학습된 checkpoint는 포함하지 않는다.

## 바로 실행

Linux NVIDIA 서버에서 ZIP을 풀고 해당 폴더로 이동한다. Python3.10 또는3.11을 권장한다.
CUDA12.8 wheel을 지원하는 NVIDIA driver가 필요하다. 다른 CUDA build를 쓰면 같은 비교 stage 전체에 동일 환경을 사용한다.

```bash
unzip KoMaP_Development_Server_20261003.zip
cd KoMaP_Development_Server_20261003
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cuda.txt
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python verify_release.py
python run.py list
python run.py check --device cuda --config configs/B.json --size 448 --batch-size 4 --data Data --report verification/server_gpu_B.json
python server_suite.py --stage validation --output runs/d03_validation
```

마지막 명령은 B43 → D03_43 → B44 → D03_44 네 run을 순차 실행하고 single/D4 평가 및 seed 요약까지 수행한다.
한 GPU에서 하나씩 학습한다. `CUDA_VISIBLE_DEVICES=0` 등으로 사용할 GPU를 지정할 수 있다.
터미널 종료 뒤에도 계속하려면 서버의 tmux 안에서 실행한다.
가중치는 첫 학습에서 torchvision이 다운로드한다. 인터넷이 없는 서버는 ImageNet 가중치를 미리 캐시에 준비해야 한다.
`TORCH_HOME=/path/to/torch_cache`로 캐시 위치를 지정할 수 있다.
ResNet101/50은 ImageNet V2, ConvNeXt-Tiny는 ImageNet V1을 쓴다. 별도 microscopy 사전학습은 사용하지 않는다.

## 실험 종류

| ID | 변경 | 구분 |
|---|---|---|
| B | ResNet101 U-Net/scSE, OS32, 기존 batch-pooled Dice | R022 recipe에 맞춘 이번 패키지 대조군 |
| D03 | B + Al3Ni/Eutectic 직접 접촉부 CE 가중2, radius2 | 서버 보고의 접촉부 loss 설정 재검증 |
| R01 | 얕은 decoder feature에 RRCU2step → 기존 scSE | 구조 단독 |
| R02 | sSE logit 안에 작은 shared recurrent residual branch | 구조 단독 |
| R03 | cSE logit 안에 GAP 이후1×1 recurrent residual branch | 구조 단독 |
| A01_CNX | ConvNeXt-Tiny + grayscale OS2 skip + 동일 U-Net/scSE | **새 설계. 과거 R024의 원본 구현 재현이 아님** |
| A02_OS16 | ResNet101 layer4 stride→dilation, OS32→16 | R039 변경에 맞춘 비교 |
| A03_R50 | ImageNet V2 ResNet50, 동일 decoder/loss/sampler | R020 recipe에 맞춘 비교 |
| 위 구조ID + `_D03` | 해당 구조 + D03 loss | 조건부 결합, 기본 실행에서 제외 |

R01은 decoder block3(448 입력에서224×224,64채널)의 기존 두 convolution을1×1압축/shared3×3Conv-BN2회/residual로 교체한다.
따라서 반복뿐 아니라 압축·공유 BN·파라미터 감소의 효과도 포함한다.
R02는 hidden16, shared3×3/GroupNorm4를2회 적용하고 sSE logit에0.1배 branch를 더한다.
R03은 GAP 이후 hidden4/shared1×1을2회 적용한다. 공간3×3 RRCU의 그대로 재현이 아닌 채널 파생 설계다.
두 gate branch의 출력 projection은0 초기화하여 첫 출력이 B와 같고, 두 optimizer step으로 내부 gradient 활성화를 확인한다.
새 recurrent module은 독립 CPU RNG에서 초기화해 공통 가중치와 crop RNG를 보존한다.
같은 구조에서 D03 유무를 비교할 때는 초기 가중치/RNG를 보존한다.
다른 backbone끼리는 초기화의 난수 소비량이 달라 crop 순서까지 동일하다고 보장하지 않는다.
동일 recipe의 시스템 비교로 해석하고, 유망한 구조는 추가 동일 seed 대조로 확인한다.

A01_CNX의 encoder는 OS4/8/16/32의96/192/384/768채널 특징과,
원본 grayscale에서 Conv3×3 stride2/BN/ReLU → Conv3×3/BN/ReLU로 만든64채널 OS2 skip을 반환한다.
과거 R024의 detail skips 코드가 제공되지 않아 이 설계를 R024 또는79.7843% 재현으로 표기하지 않는다.
A02_OS16은 `replace_stride_with_dilation=[False,False,True]`이며 해상도와 sampling/receptive-field geometry가 함께 바뀐다.

## 구조 후보 실행

전체 구조 비교는 seed42에서 B를 포함해7run이다. 이전 서버/Mac의 B 점수를 재사용하지 않는다.

```bash
python server_suite.py --stage architectures --seeds 42 --output runs/architectures42
```

특정 후보만 선택하려면 아래처럼 실행한다. B는 자동 포함된다.

```bash
python server_suite.py --stage architectures --ids R01 R02 R03 --seeds 42 --output runs/rrcu42
python server_suite.py --stage architectures --ids A01_CNX A02_OS16 A03_R50 --seeds 42 --output runs/alternatives42
```

## 조건부 결합

구조 단독과 D03이 유망할 때 선택한다. 기본값으로 모든 결합을 실행하지 않는다.
예를 들어 R02_D03은 B/D03/R02/R02_D03 네 대조를 자동 포함한다.

```bash
python server_suite.py --stage combinations --ids R02_D03 --seeds 43 44 --output runs/r02_contact
python server_suite.py --stage combinations --ids A02_OS16_D03 --seeds 43 --output runs/os16_contact43
```

모든 stage의 epoch 기본값은150이다. `--epochs 200`은 전체 대조군이 모두 fresh200epoch로 바뀌고 cosine도200에 맞춘다.
150epoch 체크포인트를200epoch로 이어 학습하는 옵션이 아니다.

## 공통 recipe 및 평가

- Train70/Valid20. Test10은 GT가 없으며 결과 비교는 Valid다.
- Grayscale→RGB 반복/ImageNet 정규화. ConvNeXt의 detail branch만 원본0–1 grayscale 사용.
- Context448/중앙224감독/stride112/reflect halo, batch4 FP32, epoch당420draws/105steps.
- 50% 희귀상 경로: Al3Ni/Eutectic 균등 선택, 중심영역10% 이상인 타일, source 균등 선택.
- 약한 밝기·대비/감마/노이즈 및 D4 증강. CE.5 + weighted Dice.25 + present-class Lovasz.25, 가중치[1,1,2,1].
- AdamW3e-4, weight decay1e-4, cosine150. Contrast/BEM 비활성화.
- 매 epoch single-view Valid mIoU로 best를 선택하고 **같은 best**를 single/D4로 평가한다. D4로 best를 다시 고르지 않는다.
- 이미지별4-class IoU 평균의 Valid20 평균. absent=1과 strict(absent=0)를 함께 저장한다. accuracy와 다르다.
- 접촉부 오류/양방향 상 혼동/Eutectic precision·recall·skeleton recall/이미지별 점수/시간·CUDA peak 메모리를 저장한다.

D계열 보고 B D4=79.7248%, D03=79.8005%는 이 패키지의 보장 점수가 아니다.
R022의79.8969% 및 R031의79.9332%도 역사적 참고 기록이다. R031은 seed43에서 이득이 재현되지 않아 기본 loss로 채택하지 않는다.
D03 역시 접촉부 오류는 줄었지만 Eutectic IoU는 낮아졌으므로 전체·상별 품질을 함께 판단한다.

## 상태 확인, 중단, 재개

```bash
cat runs/d03_validation/status.json
cat runs/d03_validation/seed43/suite_status.json
cat runs/d03_validation/seed43/B/status.json
tail -n 5 runs/d03_validation/seed43/logs/B.log
```

Ctrl-C 또는 `kill -TERM <status.json의 supervisor_pid>`는 현재 학습을 멈추고 나머지 큐도 중단한다.
마지막 **완료 epoch**의 last.pt에서 재개하며 미완료 epoch는 다시 실행한다.

```bash
python server_suite.py --stage validation --output runs/d03_validation --resume
```

동일 stage/seed/epoch/데이터/code/environment가 필요하다. 완료 run은 config/data/code/environment 검증 후 건너뛴다.
첫 epoch를 끝내기 전에 중단돼 last.pt가 없으면 해당 run은 재개할 수 없으므로 새 output에서 fresh 실행한다.
같은 output에 두 supervisor를 실행하면 lock이 막는다. 소스 변경 또는 다른 환경의 checkpoint를 섞지 않는다.

## 결과 읽기

각 run: config/data_manifest/environment, history.csv/json, best.pt/last.pt, valid_single.json/valid_d4.json,
summary.json, valid_masks/single·d4 PNG를 저장한다.
각 seed: comparisons/seed43/comparison.csv/json과 paired_deltas.json(이미지 bootstrap95% 구간)을 저장한다.
전체: seed_summary.csv/json에 **같은 seed의 B 대비 차이**를 평균·표준편차로 요약한다.
결합 stage는 구조 단독/D03 단독 대비 차이와 interaction도 저장한다.
이미지 bootstrap은 학습 seed 변동과 다르다. 두 seed의 평균·표준편차만으로 통계적 우위를 확정하지 않는다.

Test 예측은 평가/모델 결정 뒤 별도로 수행한다.

```bash
python run.py evaluate --checkpoint runs/d03_validation/seed43/D03/best.pt --data Data --split test --d4 --device cuda --output predictions/d03_seed43_test
```

## 검증과 패키지 무결성

verification/에는 CPU synthetic14개 모델의2 optimizer step, recurrent gate gradient/접촉부 가중/중단·재개 동일성,
실제 타일 평가·PNG저장·비교, 단계 선택·seed 요약 검사를 기록한다. 성능 측정 실험과 다르다.
새 encoder 후보는 실제 크기448/batch4의 CPU 실행 검사도 별도 기록한다.
CUDA 실행은 이 Mac에서 검증하지 않았으므로 위 server GPU check를 먼저 수행한다.

```bash
python verify.py
python verify.py --full-size
python run.py check --device cpu --size 64 --report verification/local_synthetic.json
```

SHA256SUMS.txt와 RELEASE_MANIFEST.json은 배포 당시 코드·데이터 checksum을 포함한다.
`verify_release.py`는 전달된 파일을 검사한다. 실험·검증 산출물 추가는 원래 파일의 checksum을 바꾸지 않는다.
README의 실행 검사 명령은 별도 report 이름을 사용해 기존 검증 파일을 덮어쓰지 않는다.
학습은 실행 명령을 입력해야 시작한다. ZIP 생성만으로 서버 학습이 시작되지는 않는다.
