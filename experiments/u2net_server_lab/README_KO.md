# KoMaP U²-Net / RSU 서버 실험 패키지

**먼저 RSU encoder + 기존 B decoder 조합 5개를 비교한다.** 기존 모델 B, 접촉부 loss B_D03, RSU 조합 U0, scSE 제거 U1, 접촉부 loss 결합 U2다. 기본 예산은 각 150epoch·seed42다. 추가 RRCU, 원형 U²-Net, deep supervision은 별도 stage로 선택한다.

메인 ZIP에는 코드, 14개 설정, 실험 계획, Train70/Valid20 원본 데이터, 실행 검증 기록을 포함한다. Test10 원본은 업로드 도구의 파일당100MiB 제한 때문에 별도 `KoMaP_U2Net_RSU_Test10_20261004.zip`에 넣었다. 학습·Valid 평가에는 메인 ZIP만 필요하다. 학습된 가중치는 포함하지 않는다. 패키지 제작 과정에서 KoMaP 데이터로 새 학습을 실행하지 않았다.

## 바로 실행

Linux NVIDIA 서버에서 아래 순서로 진행한다. Python 3.10/3.11 환경을 권장한다. 기본 설치 파일은 PyTorch 2.7.1 / torchvision 0.22.1 / cu128 wheel이다. 서버 driver에 맞는 다른 공식 CUDA wheel을 사용해도 되지만 비교하는 모든 run의 환경을 통일한다.

```bash
unzip KoMaP_U2Net_RSU_Server_20261004.zip
cd KoMaP_U2Net_RSU_Server_20261004
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cuda.txt
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python verify_release.py
python run.py list
```

Test10 추론도 하려면 두 ZIP을 같은 상위 폴더에 내려받고, 위의 `cd` 전에 `unzip KoMaP_U2Net_RSU_Test10_20261004.zip`을 추가한다. 두 ZIP은 같은 패키지 폴더 아래로 풀린다. 메인 `RELEASE_MANIFEST.json`에 Test10 ZIP의 SHA256을 기록했다. Test10 추가 후에도 기존 배포 파일의 검증은 동일하게 통과한다.

학습 전에 서버 GPU에서 실제 입력 크기로 확인한다. 아래는 합성 데이터의 forward/backward·optimizer 검사이며 실제 학습이나 가중치 다운로드를 하지 않는다.

```bash
python run.py check --device cuda --config configs/B.json --size 448 --batch-size 4 --data Data --report verification/server_gpu_B.json
python run.py check --device cuda --config configs/U2.json --size 448 --batch-size 4 --report verification/server_gpu_U2.json
python run.py check --device cuda --config configs/F3.json --size 448 --batch-size 4 --report verification/server_gpu_F3.json
```

F3는 선택 실험이므로 core만 실행할 서버에서는 F3 검사를 생략할 수 있다. 처음에는 아래 **한 stage만** 실행한다. B → B_D03 → U0 → U1 → U2를 한 GPU에서 순차 학습하고 single/D4 평가 및 비교를 자동 저장한다.

```bash
python server_suite.py --stage core --output runs/core42
```

터미널 종료 뒤에도 계속하려면 tmux 안에서 실행한다. GPU는 `CUDA_VISIBLE_DEVICES=0` 등으로 지정한다. B/B_D03은 첫 학습 때 torchvision의 ImageNet V2 ResNet101 가중치를 다운로드한다. 인터넷이 없는 서버는 공식 가중치를 torchvision 캐시에 준비한다. `TORCH_HOME=/path/to/torch_cache`로 캐시 위치를 지정할 수 있다. RSU / full U²-Net은 scratch 초기화이며 별도 사전학습 파일을 요구하지 않는다.

## 실험 목록

| ID | 구조 / 변경 | 비교 목적 |
|---|---|---|
| B | ImageNet V2 ResNet101 + 기존 U-Net/scSE | 이번 패키지의 대조군 |
| B_D03 | B + Al3Ni–Eutectic 접촉부 CE ×2, radius2 | 접촉부 loss 대조군 |
| B_SCRATCH | B 구조, 사전학습 없이 초기화 | RSU와의 초기화 차이 해석 |
| U0 | RSU encoder + 채널 adapter + B decoder/scSE | RSU backbone 조합 |
| U1 | U0에서 decoder scSE 제거 | U0−U1로 scSE 효과 확인 |
| U2 | U0 + D03 접촉부 loss | RSU와 접촉부 loss 결합 |
| U3_R01 | U0의 얕은 decoder feature를 RRCU로 교체 | feature 반복 정제 |
| U3_R02 | U0의 얕은 scSE sSE logit에 recurrent branch | 공간 gate 반복 정제 |
| U3_R03 | U0의 얕은 scSE cSE logit에 recurrent branch | 채널 gate 반복 정제 |
| F0 | 원형 full U²-Net을 4클래스 raw logits로 변환 | RSU decoder까지 포함한 대안 |
| F0_D03 | F0 + D03 | 원형 구조에서 접촉부 loss 효과 |
| F1 | F0의 5개 decoder RSU 뒤에 scSE | full U²-Net + scSE |
| F2 | F1 + D03 | full U²-Net + scSE + 접촉부 loss |
| F3 | F1 + 6개 side head deep supervision | 추가 감독 효과 |

모든 U/F 후보의 결과는 아직 없다. 과거 최고 점수를 이 후보들의 성능으로 표기하지 않는다. 우선순위와 판단 기준은 [EXPERIMENT_PLAN_KO.md](EXPERIMENT_PLAN_KO.md)에 정리했다.

## 구조를 어떻게 연결했나

U0 계열의 encoder는 원형 U²-Net의 RSU7 → RSU6 → RSU5 → RSU4 → RSU4F → RSU4F다. stage1은 원본 해상도로 처리하고, **stage2–6의 OS2/4/8/16/32 특징**을 B의 decoder에 전달한다. 원형 stage1 특징은 encoder 내부에 사용하지만 B decoder의 직접 skip에는 연결하지 않는다.

```text
grayscale → RGB 반복/정규화
  → RSU encoder: [64,128,256,512,512,512]
  → stage2–6: [128,256,512,512,512]
  → 필요 위치 1×1 adapter: [64,256,512,1024,2048]
  → 기존 B U-Net decoder + scSE → 4채널 logits
```

U0/B 비교에는 backbone, adapter, 사전학습 유무의 차이가 함께 들어간다. 순수한 RSU 블록 하나의 인과 효과로 해석하지 않는다. B_SCRATCH/U0 비교는 사전학습 차이를 줄이는 보조 대조다. 같은 설정 seed에서 crop·희귀상 선택·증강 순서를 backbone과 독립된 RNG로 맞췄다. 이 sampler 때문에 **과거 패키지 B 점수를 그대로 대조군으로 재사용할 수 없다.**

U3_R01은 decoder block3의 224×224/64채널 feature에 1×1 압축 → shared 3×3 Conv/BN 2회 → residual → scSE를 적용한다. 기존 두 convolution을 교체하므로 압축·공유 BN·파라미터 변화도 함께 비교한다. U3_R02는 hidden16의 shared 3×3/GroupNorm 공간 branch, U3_R03은 GAP 뒤 hidden4의 shared 1×1 채널 branch다. 두 gate branch는 0 출력으로 시작하고 기존 gate logit에 0.1배를 더한다. U3_R03은 공간 RRCU의 채널 파생 설계다.

F0 계열은 원형 encoder와 RSU decoder, 6개 side head, fusion head를 모두 유지한다. 각 side는 4채널이고 fusion은 24→4 convolution이다. 원형 binary SOD 코드의 sigmoid를 제거했다. 모든 loss는 raw logits를 받으며 클래스 선택은 argmax다. 여섯 번째 side head는 가장 깊은 encoder 특징에서 나온다.

F0/F1/F2의 직접 loss는 fused head만 사용하지만 여섯 side head도 fusion을 통해 학습된다. F3만 여섯 head에 직접 auxiliary loss를 준다. 각 head를 448 해상도로 정렬한 다음 중앙224를 감독하며, `L = (L_fused + 0.1 × Σ L_side) / 1.6`이다. head별 loss는 아래 공통 recipe와 같다. F1/F3의 초기 공통 가중치와 U0/U2의 초기 가중치를 같게 유지했다.

## 공통 조건과 결과 해석

- 기존 Train70/Valid20 분할을 유지한다. Test10은 GT가 없고 모델 선택에 사용하지 않는다. 입력 전체를 256/512로 resize하거나 새 분할을 만들지 않는다.
- 기존 RGB palette mask를 0=Primary Si, 1=Al3Ni, 2=Eutectic Si, 3=Al로 해독한다. grayscale 입력을 RGB로 반복하고 동일 ImageNet 정규화를 적용한다.
- Context448 / 중앙224 감독 / stride112 / reflect halo, FP32, train batch4, workers0. Epoch당420 draws /105 optimizer steps.
- 50% 희귀상 경로: Al3Ni/Eutectic 균등 선택, 중앙영역10% 이상인 타일, source 균등 선택. 약한 밝기·대비/감마/노이즈와 D4 증강.
- CE0.5 + batch-pooled weighted Dice0.25 + present-class Lovasz0.25, 클래스 가중치[1,1,2,1]. D03는 직접 접촉부 주변 CE에만 가중2를 적용한다. BEM/대조학습은 비활성화했다.
- AdamW lr3e-4 / weight decay1e-4 / cosine150. 기본 seed42. CUDA 알고리즘은 강제 deterministic 모드가 아니므로 비트 단위 재현을 보장하지 않는다.
- **Valid20 single 점수로 best를 선택한 뒤 같은 checkpoint를 single/D4로 평가**한다. D4로 best를 다시 선택하지 않는다.
- 점수는 이미지별 4클래스 IoU 평균의 Valid20 평균이다. absent=1과 strict(absent=0)를 함께 저장한다. Accuracy와 다르다.
- 접촉부 오류, 양방향 상 혼동, Eutectic precision/recall/skeleton recall, 이미지별 점수, 시간과 CUDA peak 메모리를 저장한다. 이미지 bootstrap과 seed 간 변동을 구분한다.

과거 서버 B D4=79.7248%, D03=79.8005%는 참고값이다. D03의 개선 폭은 작았고 Eutectic IoU 감소가 있었다. R022=79.8969%, R031=79.9332%도 이 패키지의 보장 성능이 아니다. R031의 추가 seed 이득은 재현되지 않아 기존 batch-pooled Dice를 유지했다. 원본 제안서는 `docs/U2NET_MULTICLASS_ORIGINAL_PROPOSAL.md`에 보관했으며 실제 실행 조건은 본 README와 configs가 기준이다.

## 추가 stage

아래 명령은 선택 실행이다. 전체를 한꺼번에 실행할 필요는 없다.

```bash
# core에서 유망한 결과가 나오면 같은 seed 대조로 재현: 4종 ×2 seeds =8 runs
python server_suite.py --stage reproduce --output runs/reproduce43_44

# 사전학습 유무 해석: B_SCRATCH/U0
python server_suite.py --stage scratch --output runs/scratch42

# RSU 조합의 RRCU 비교: U0/U3_R01/U3_R02/U3_R03
python server_suite.py --stage rrcu --output runs/rrcu42

# 원형 RSU decoder까지 포함: B/F0/F1/F0_D03/F2
python server_suite.py --stage full --output runs/full42

# deep supervision: F1/F3
python server_suite.py --stage deep --output runs/deep42
```

`--seeds 43 44`로 모든 stage의 seed를 지정할 수 있다. `--ids`로 해당 stage의 일부 후보를 선택하면 필수 대조군을 자동 추가한다. 예: `--stage rrcu --ids U3_R02`는 U0/U3_R02 두 run이다. `--epochs 200`은 대조군까지 모두 fresh200epoch / cosine200으로 변경하며 150epoch 체크포인트를 이어 학습하는 옵션이 아니다.

GPU 메모리가 부족하면 **새 output 폴더에서 stage 전체를 `--batch-size 2`로 실행**한다. 학습은420draws/210steps로 바뀌고 batch-pooled Dice도 달라지므로 batch4 결과와 같은 실험으로 묶지 않는다. 평가 batch는 config의4를 유지한다. AMP/gradient accumulation을 자동 적용하지 않는다.

## 진행 확인 / 중단 / 재개

```bash
cat runs/core42/status.json
cat runs/core42/seed42/suite_status.json
cat runs/core42/seed42/U0/status.json
tail -n 5 runs/core42/seed42/U0/history.csv
tail -n 20 runs/core42/seed42/logs/U0.log
```

Ctrl+C 또는 supervisor SIGTERM은 현재 학습을 멈추고 남은 queue를 취소한다. 동일한 data/code/config/환경과 동일한 실행 인자를 유지하고 다음처럼 재개한다.

```bash
python server_suite.py --stage core --output runs/core42 --resume
```

완료 run은 검증 후 건너뛰고 중단 run은 마지막 완료 epoch에서 복원한다. optimizer/scheduler와 sampling/loader/global RNG를 보존한다. 다른 epoch 예산, batch, seed, code, data, 환경으로 바꾸려면 새 output을 사용한다. ZIP을 푼 뒤 `verify_release.py`는 파일 무결성을 확인하는 일회성 검사다. 이후 GPU 검사 보고서를 새 파일로 추가해도 기존 배포 파일은 덮어쓰지 않는다.

## 팀원에게 받을 결과

각 stage의 `seed_summary.json/csv`, `comparisons/seed*/`, 각 run의 `summary.json`, `history.csv`, `valid_single.json`, `valid_d4.json`, `valid_masks/`, `config.json`, `environment.json`, `data_manifest.json`, `best.pt`, `last.pt`를 보존한다. GPU 검사 보고서와 log도 함께 받는다. 시간/메모리/상별 품질을 확인하고 동일 seed의 B 또는 해당 구조 대조군과 비교한다. 점수 상승만으로 RSU/RRCU의 재현 가능한 우위를 확정하지 않는다.

```text
run.py                 단일 실행 / 검사 / 평가 / 비교
server_suite.py        stage·seed 계획과 순차 실행
run_suite.py           동일 seed의 후보 순차 실행
make_configs.py        seed/epoch/batch 설정 생성
summarize_seeds.py     seed 평균·변동·상별 실험 대비
komap_phase/           모델·데이터·loss·학습·평가
configs/               14개 기본 설정
tests/, verification/  합성 입력 검증 및 실행 기록
docs/                  기존 성능 보고와 원본 제안
Data/                  Train70 / Valid20 (Test10은 별도 ZIP 추가)
```

로컬 검증은 Mac CPU / PyTorch2.8.0 / torchvision0.23.0에서 수행했다. 13개 테스트, 14개 후보의 두 optimizer step, U2/F3의 실제448·batch4 입력, 합성 fixture의 평가·checkpoint·재개 경로를 확인했다. CUDA와 서버의 고정 wheel 조합은 아직 직접 검증하지 않았으므로 위 GPU 검사를 먼저 실행한다. 이 검사들의 loss 감소는 실제 데이터 성능 근거가 아니다.

원형 RSU/U²-Net 코드와 Apache-2.0 라이선스를 동봉했다. 출처와 수정 내역은 [NOTICE.md](NOTICE.md)를 참고한다.
