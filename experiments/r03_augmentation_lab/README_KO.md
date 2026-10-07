# R03 데이터 증강 비교 — 2026-10-05

현재 최고 관측 모델 R03(cSE channel-logit RRCU branch,추가600parameter)의 구조를 고정한다.
공개 ImageNet V2에서 각 run을 fresh 시작하며 기존 R03 best checkpoint를 이어 학습하지 않는다.
각150epochs,420draws/epoch,FP32,batch4,context448/target224/stride112,
AdamW3e-4,CE.5+Dice.25+Lovasz.25,기존sampling/loss/evaluation을 유지한다.
Train70/Valid20,이미지별4클래스mIoU,빈클래스1. 공식Test결과가아니다.

| ID | 현재 증강 대비 변경 |
|---|---|
| A00_BASE | 원본R03증강 대조군 |
| A01_BRIGHT_HALF | 밝기±.08→±.04 |
| A02_BRIGHT_OFF | 밝기제거,대비유지 |
| A03_CONTRAST_OFF | 대비제거,밝기유지 |
| A04_GAMMA_OFF | gamma제거 |
| A05_GAMMA_HALF | gamma.9–1.1→.95–1.05 |
| A06_NOISE_OFF | Gaussian noise제거 |
| A07_NOISE_HALF | noiseσ.01→.005 |
| A08_PHOTO_OFF | 원본광도증강전부제거 |
| A09_BLUR_WEAK | Gaussianblur p.15,σ.3–.6추가 |
| A10_ILLUM_GRAD | 선형조명gradient p.3,최대±.04추가 |
| A11_BRIGHT_HALF_BLUR | 밝기±.04와약한blur의사전지정조합 |

각seed42/43/44,총36runs·5400epochs. GPU마다학습프로세스하나.
gpu-l4-aug1은A00/01/03/05/07/09,aug2는A02/04/06/08/10/11을담당한다.
각GPU에18runs. 첫번째가대조군,두번째가밝기제거부터시작한다.

난수소비량을맞춰동일seed에서원본이미지/crop/D4변환을짝지었다.
제거한증강도원본난수는소비하고,추가blur/gradient는독립CPU RNG를사용한다.
이독립RNG는lastcheckpoint에저장·복구된다.
verify_augmentation.py는12조건의동일crop/기하변환,원본baselinebitwise동일,
RNG복구후동일다음sample,범위/shape/유한값을검증한다.
로컬CPU모델forward/backward와채널RRCUgradient를확인했다.
서버에서는같은검증과448/batch4CUDAoptimizerpreflight후fresh학습한다.

최고checkpoint는single-viewValid로선택하며같은checkpoint의single/D4를평가한다.
클래스별IoU,Eutecticprecision/recall/thin-skeletonrecall,양방향혼동,접촉부오류를저장한다.
seed별A00대조군과의차이및3seed평균을우선사용한다.
12조건을Valid20에서탐색하므로관측최고설정을독립Test우위로표현하지않는다.

원본공유패키지source_snapshot은변경하지않았다.새schema3와새출력경로를사용한다.
중단된run은자동재개하지않으며예기치않은실패만완료epoch의optimizer/scheduler/RNG에서복구한다.
STOP_AUTOMATION을운영폴더에만들면새할당/자동재시작을멈춘다.

```bash
export TORCH_HOME="$PWD/torch_cache"
.venv/bin/python verify_release.py
.venv/bin/python verify_augmentation.py Data checks/augmentation_cuda_host.json
.venv/bin/python run.py check --config configs/A00_BASE.json --device cuda --size 448 --batch-size 4 --report checks/model_cuda.json
.venv/bin/python -u aug_suite.py --session gpu-l4-aug1 --data Data --output runs/augmentation
```
