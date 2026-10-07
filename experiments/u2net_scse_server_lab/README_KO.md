# U²-Net / scSE 서버 실험 패키지

우리 합금 4클래스 segmentation에서 scSE의 효과를 U²-Net 기본형과 직접 비교한다.
기존 127개 아키텍처 ZIP과 독립된 패키지다. 기존 실행 폴더의 checkpoint를 이어 학습하지 않는다.
새 실제 데이터 학습 결과나 합금으로 학습된 U²-Net 가중치는 포함하지 않는다.

## 바로 실행

ZIP을 압축 해제하고 아래 명령을 실행한다. Train 70장·Valid 20장의 이미지와 mask가 포함돼 있다.
Linux / NVIDIA CUDA 서버, Python 3.10–3.12 권장. setup은 PyTorch 2.7.1 / torchvision 0.22.1 cu128을 설치한다.
인터넷은 dependency 설치와 B의 공개 ImageNet ResNet101 가중치 다운로드에 필요하다.

```bash
cd KoMaP_U2Net_scSE_Server_20261004
bash setup.sh
bash run_server.sh
```

기본 실행은 **16개 구성 × seed 42/43/44 = 48회, 각 150 epoch**다. 전체 7,200 epoch.
작은 입력으로 16개 모델의 forward/backward/optimizer를 검사하고, 두 attention 조합은
실제 448×448 / batch 4 크기로 먼저 검사한다. GPU 검사가 실패하면 본 학습을 시작하지 않는다.
FP32를 사용하며 batch size나 학습 조건을 자동으로 낮추지 않는다.

중단했으면 같은 명령에 `--resume`을 전달한다. 완료한 run은 건너뛰고 미완료 run은
마지막 완료 epoch의 모델·optimizer·scheduler·난수·patch sampler 상태에서 재개한다.
진행 중 epoch는 다시 실행된다. 첫 epoch checkpoint가 없으면 실패 시도를 보관하고 fresh 재시작한다.

```bash
bash run_server.sh all runs/u2_attention_all_batch4 4 --resume
```

GPU 사전 검사에서 메모리가 부족하면 **전체 실험을 batch 2로 맞춘 별도 실행**을 사용한다.
batch 4 결과와 batch 2 결과를 같은 대조표로 섞지 않는다.

```bash
bash run_server.sh all runs/u2_attention_all_batch2 2
```

원격 세션이 끊겨도 계속 실행하려면 `tmux` 세션 안에서 실행한다.
개별 run 로그: `runs/u2_attention_all_batch4/seed42/logs/U2_SCSE_ALL_DEEP.log`.
전체 상태: 실행 폴더의 `status.json`; 개별 진행률: `seed42/U2_SCSE_ALL_DEEP/status.json`.

## 실험 구성

아래 U²-Net 7개를 각각 **deep supervision ON / OFF**로 비교한다. 별도로 B와 B_SCRATCH를 포함한다.

| 구성 | 적용 위치 | 비교 목적 |
|---|---|---|
| U2_PLAIN | attention 없음 | U²-Net 기준 |
| U2_SCSE_ALL | stage5d→stage1d, decoder RSU 5개 출력 뒤 | scSE 전체 decoder 효과 |
| U2_SCSE_LAST2 | stage2d·stage1d 출력 뒤 | 해상도가 높은 마지막 2개 단계의 효과 |
| U2_ECA | encoder 6개 + decoder 5개 RSU의 마지막 residual branch | 채널 attention 단독 효과 |
| U2_CBAM | side1–side6의 입력 feature, 예측·fusion 전 | 채널→공간 attention 단독 효과 |
| U2_ECA_CBAM | RSU residual branch ECA + side feature CBAM | 논문에서 제안한 조합의 파생 실험 |
| U2_ECA_SCSE | RSU residual branch ECA + decoder 5단 scSE | CBAM 조합과 비교할 후보 |
| B | 기존 ResNet101 U-Net/scSE, ImageNet V2 시작 | 현재 실용 기준 재측정 |
| B_SCRATCH | B 구조, 공개 가중치 없이 fresh 시작 | 사전학습 효과를 구분하는 대조군 |

`_DEEP`가 붙은 ID는 six-side supervision ON이다. 공통 fused loss에
`0.2 × mean(6개 side loss)`를 더한다. 각 side loss는 공통 CE/Dice/Lovasz다.
ON 그룹끼리, OFF 그룹끼리 attention 효과를 판단한다. ON과 OFF를 섞어 scSE 효과로 해석하지 않는다.

scSE는 channel gate와 spatial gate를 병렬로 적용해 `x*cSE + x*sSE`로 합친다.
ECA는 global average pooling 뒤 kernel 3의 1D convolution으로 channel gate를 만든다.
**이 패키지의 ECA는 RSU의 `hx1d`에 적용하고 `hxin` shortcut은 유지**한다.
CBAM은 average/max pooling 기반 channel gate 뒤 kernel 7 spatial gate를 순차 적용한다.
6개 feature에 적용한 CBAM 출력으로 side 예측과 6-head fusion을 모두 계산한다.
이는 attention 아이디어를 가져온 독립 구현이며, 합금 논문의 정확한 layer 배치·학습법 재현이라고 주장하지 않는다.

모든 U²-Net은 원 저자 full U2NET의 RSU·feature 폭·6-head fusion을 유지하며,
출력을 4클래스 raw logits로 바꾼다. saliency pretrained weight는 사용하지 않고 fresh scratch 시작한다.
U²-Net 공통 가중치와 전체 global RNG는 같은 seed의 모든 attention 구성에서 동일하다.
추가 attention 가중치는 별도 RNG stream에서 초기화한다. patch·증강 sampler도 모델 RNG와 분리한다.

## 학습·평가 조건

| 항목 | 설정 |
|---|---|
| 데이터 | Train70 / Valid20, 4클래스: Primary Si / Al3Ni / Eutectic Si / Al |
| 입력 | grayscale → RGB 반복, ImageNet 정규화 |
| patch / 감독 영역 | 448×448 context / 중앙 224×224 |
| sampling | epoch당 420 patches; Al3Ni/Eutectic-rich tile 50%, 희귀상 최소10% |
| optimizer | AdamW, LR3e-4, weight decay1e-4 |
| scheduler | cosine, 150epoch |
| 공통 loss | CE0.5 + batch-pooled Dice0.25 + per-image present-class Lovasz0.25 |
| Dice/Lovasz 클래스 가중치 | 1:1:2:1; CE 클래스 가중치 없음 |
| 접촉부 CE 강화 | 사용하지 않음; D03를 동시 적용하지 않음 |
| checkpoint 선택 | single-view Valid 이미지별 4클래스 mIoU 평균, union=0이면 IoU1 |
| 최종 평가 | 선택된 동일 checkpoint의 single / D4(8-view probability 평균) |
| 전체 타일 추론 | 중앙224 출력, stride112, overlap 확률 평균 |

Valid20의 반복 사용에 따른 선택 편향이 있으므로 공식 Test/generalization 성능으로 표현하지 않는다.
공식 Test 이미지와 label은 이 패키지에 포함하지 않는다.
주 비교는 같은 U²-Net 기본형 대비 ΔmIoU이고, B 대비 성능은 실용적인 보조 비교다.
B 사전학습과 scratch U²-Net 비교에는 구조와 초기화 차이가 함께 포함된다.
기존 B 점수 대신 이번 실행의 B를 같은 sampler·seed·환경으로 재측정한다.

## 실행 규모 선택

| profile | 구성 수 | 3seed run 수 | 범위 |
|---|---:|---:|---|
| all (기본) | 16 | 48 | U²-Net 7종 × supervision2종 + B/B_SCRATCH |
| core | 9 | 27 | supervision ON 7종 + B/B_SCRATCH |
| scse | 5 | 15 | ON 기본형·전체 scSE·마지막2단 scSE + B/B_SCRATCH |

```bash
bash run_server.sh scse runs/u2_scse_focus_batch4 4
bash run_server.sh core runs/u2_attention_core_batch4 4
```

profile/epoch/seed/batch/코드/데이터/환경을 바꿔 기존 output에 resume할 수 없다.
후보 run 실패는 로그에 보존하고 다음 후보로 진행한다. B 실패 시 대조군 부재로 중단한다.
부분 결과에는 실제 완료한 seed 수가 표시된다.

## 결과 확인·회수

전체 완료 후 자동으로 생성한다. 부분 실행 중에도 `summarize_seeds.py`로 재집계할 수 있다.

- `U2_ATTENTION_RESULTS_KO.md`: **동일 supervision U²-Net 기본형 대비 상승 폭**.
- `u2_paired_leaderboard.csv`: ΔmIoU 평균·paired seed 표준편차·seed 승패·상별 IoU·접촉부 오류·얇은 구조 recall.
- `u2_paired_seed_details.json`: seed별 Δ와 paired image bootstrap 95% 구간.
- `RESULTS_KO.md`, `leaderboard.csv`: 현재 B 대비 전체 순위 및 seed 평균.
- `seed42/comparisons`가 아니라 최상위 `comparisons/seed42`: B 기준 상세 비교·이미지 bootstrap.
- `seed42/<ID>/best.pt`, `last.pt`: 최고 checkpoint와 재개 상태.
- `seed42/<ID>/valid_masks/single`, `d4`: 최종 예측 mask.

```bash
.venv/bin/python summarize_seeds.py --plan runs/u2_attention_all_batch4
.venv/bin/python pack_results.py --plan runs/u2_attention_all_batch4 --output U2_scSE_results.zip --masks
```

`U2_scSE_results.zip`에는 보고서·로그·mask를 넣고 대형 checkpoint는 제외한다.
재회수 시에는 새 ZIP 파일명을 지정한다. best/last checkpoint는 서버에 유지한다.
학습 seed 변동과 image bootstrap은 구분해서 해석한다.

## 검증 범위와 자원

배포 전 로컬 CPU synthetic 검사 결과는 `verification/`에 들어 있다.
16개 구성의 두 optimizer step·추론, 실제 attention gradient, 공통 초기값/RNG,
RSU residual 경로, checkpoint/mask 저장·평가, 정확한 epoch 재개를 검사한다.
CUDA는 서버에서 `run_server.sh`가 직접 검사한다. 로컬 CPU 검사만으로 GPU 호환성·메모리를 보장하지 않는다.

`RESOURCE_ESTIMATES.csv`의 checkpoint 예상량은 tensor 크기 기반이며 실제 파일 크기와 다를 수 있다.
full U²-Net FP32는 높은 GPU memory를 요구할 수 있다. 사전 검사 로그의 peak_cuda_memory_bytes를 참고한다.
전체 결과의 best/last checkpoint만 대략 40GiB 규모이며, activation·중간 파일·mask의 여유 공간도 필요하다.

원 저자 RSU 소스는 Apache-2.0이고 `LICENSE_U2NET.txt` 및 `docs/U2NET_SOURCE_PROVENANCE.json`을 포함한다.
참고 논문: [U²-Net](https://arxiv.org/abs/2005.09007), [scSE](https://arxiv.org/abs/1808.08127),
[ECA](https://arxiv.org/abs/1910.03151), [CBAM](https://arxiv.org/abs/1807.06521),
[합금 ECA/CBAM U²-Net](https://doi.org/10.1016/j.matchar.2025.115198).
