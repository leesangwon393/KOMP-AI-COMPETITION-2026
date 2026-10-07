# KoMaP 논문 아키텍처 서버 실험 패키지

2026-10-04. **127개 구성**, 150epoch/seed, 원본 Train70/Valid20 포함. 학습 성능 결과나 학습 checkpoint를 포함하지 않는다. 이 ZIP을 서버에서 실행해야 새 결과가 생긴다. 원 논문 전체 구조, 라이브러리 구조의 4상 이식, 독립 파생 모듈을 `EXPERIMENTS.csv`의 `implementation` 열로 구별한다. 세부 범위는 `COVERAGE_KO.md`에 있다.

Linux NVIDIA GPU, Python3.10/3.11 기준:

```bash
unzip KoMaP_Paper_Architecture_Server_20261004.zip
cd KoMaP_Paper_Architecture_Server_20261004
python3 verify_release.py
bash setup.sh
source .venv/bin/activate
python verify.py --device cuda --ids B M_FREQ_OFFSET H_RESNET101_SWIN F_U2P_DEEP --size 448 --batch-size 4 --output verification/server_cuda
python -u server_suite.py --profile all --seeds 42 43 44 --output runs/all_3seeds
```

마지막 명령은 **127구성 ×3seed =381run, 최대57,150epoch**다. GPU1개에서 순차 실행한다. 설치/가중치 다운로드 때 인터넷이 필요하다. 가중치는 ZIP에 넣지 않고 torchvision/SMP/timm/NASA의 공개 가중치를 서버 cache에 내려받는다. 모든 run은 공개 초기 가중치 또는 명시한 scratch에서 새로 시작하며 이전 KoMaP checkpoint를 학습 초기화에 사용하지 않는다. 실제 시간은 B 완료 후 `history.csv`의 epoch 시간을 보고 예산을 잡는다. 서로 다른 모델의 시간은 다르므로 전체 시간을 단정하지 않는다.

범위를 줄이거나 1차 선별 후 추가 검증하려면:

```bash
python -u server_suite.py --profile core --seeds 42 --output runs/core42
python -u server_suite.py --profile all --seeds 42 --output runs/all42
python -u server_suite.py --ids E_EFFICIENTNET_B3 M_FREQ_OFFSET F_U2_DEEP --seeds 43 44 --output runs/selected4344
```

`core`는 대표30개다. `all`은 core를 먼저 실행한 뒤 나머지를 실행한다. `encoders`, `decoders`, `modules`, `hybrids`, `pretraining`도 선택할 수 있다. 개별 `--ids`는 항상 같은 seed의 B를 함께 실행한다. scratch 후보를 구조 효과로 해석하려면 `B_SCRATCH`도 ids에 추가한다.

중단 후 **같은 명령 끝에 `--resume`**을 붙인다. 완료된 run은 config/data/code/환경과 checkpoint를 확인한 후 건너뛴다. 미완료 run은 마지막 완료 epoch의 optimizer/scheduler/RNG/best 상태에서 재개한다. 첫 epoch 전에 실패해 checkpoint가 없는 폴더는 `failed_attempts/`에 보존한 후 다시 시작한다. Ctrl-C/SIGTERM은 진행 중 run과 queue를 정지한다. epoch 도중 종료하면 그 epoch는 다시 실행한다.

```bash
python -u server_suite.py --profile all --seeds 42 43 44 --output runs/all_3seeds --resume
```

OOM/가중치 다운로드 오류/모델 실패는 log 및 `failures.json`에 기록하고 다음 후보를 실행한다. B 실패는 해당 계획을 중단한다. 실패가 있으면 전체 완료 exit code는1이다. `--stop-on-error`는 첫 후보 오류에서도 중단한다. 배치를 자동 변경하지 않는다. batch4가 불가능한 GPU라면 **새 output에서 B를 포함한 전 계획을 batch2로** 실행한다. batch2와 batch4 결과는 직접 같은 paired 표에 합치지 않는다.

```bash
python -u server_suite.py --profile all --seeds 42 43 44 --batch-size 2 --output runs/all_3seeds_batch2
```

공통 설정은 B의 Context448/중앙224/stride112, 원본 해상도, 420crops/epoch, FP32, AdamW3e-4/weight decay1e-4/cosine150, CE0.5+Dice0.25+Lovasz0.25, 클래스 가중치[1,1,2,1]이다. Train70만 sampling/augmentation/auxiliary supervision에 사용한다. Valid20는 single-view checkpoint 선택 및 검증에 사용하며 생성기·teacher 학습에 넣지 않는다. D03는 별도 대조군이며 새 구조에 자동 적용하지 않는다.

이번 묶음은 patch/augmentation 전용 CPU RNG를 `seed+8101`로 분리했다. 동일 seed의 구조들은 모델 초기화·dropout과 관계없이 같은 crop/증강 draw를 받으며 sampler state도 last.pt에 저장한다. B의 초기 모델·loss·sampling 확률은 유지하지만 과거 패키지와 난수 draw stream은 달라진다. 따라서 같은 서버 묶음에서 다시 학습한 B와 비교하며 과거 B 수치를 이번 B 결과로 대입하지 않는다.

SMP native 디코더는 라이브러리의 출력 stride·채널·ImageNet 초기화를 사용한다. 예를 들어 DeepLabV3는 OS8, DeepLabV3+는 OS16이고 B는 OS32다. `model.output_stride=32`는 공통 decoder 경로의 규약이며 native 구조의 실제 stride를 지정하는 옵션이 아니다. 그래서 native 시스템 비교와 공통 decoder의 encoder 비교를 구별한다. `D_UNET`는 SMP decoder/ImageNet 초기화 차이를 보여주는 별도 대조군이다. ConvNeXt V1 후보는 timm의 `in12k_ft_in1k` 태그, V2는 `fcmae_ft_in1k` 태그를 명시한다. 서로 다른 사전학습 이력까지 순수 구조 효과로 해석하지 않는다. MicroNet은 B와 같은 R101에 NASA v1.0 가중치만 교체하고 입력 정규화는 ImageNet으로 고정했다. 원 논문의 preprocessing까지 동일한 재현은 아니다.

결과 위치:

| 파일 | 내용 |
|---|---|
| `leaderboard.csv`, `RESULTS_KO.md` | 구조·초기화·구현 수준·완료 seed수·평균 mIoU·paired B 대비 증가 %p |
| `comparisons/seed*/comparison.csv` | single/D4, class IoU, Eutectic precision/recall, thin skeleton recall, 접촉부 오류, 상별 confusion, 시간, parameter 수, CUDA peak memory |
| `comparisons/seed*/paired_deltas.json` | 이미지별 승/패 및 paired image bootstrap95% 구간 |
| `seed*/ID/history.csv` | epoch별 train loss/aux loss/single validation/class IoU/lr/time |
| `seed*/ID/best.pt`, `last.pt` | best single checkpoint / exact epoch resume 상태 |
| `seed*/ID/valid_masks/{single,d4}` | 동일 best checkpoint의 예측 마스크 |
| `seed*/logs/ID.log`, `failures.json` | 오류와 미실행/실패 목록 |

checkpoint는 **single-view 이미지별4클래스 mIoU 평균, 빈 union=1**로 고른다. D4는 그 동일 checkpoint의 8변환 ensemble이며 D4로 checkpoint를 다시 고르지 않는다. 빈 union=0 점수도 별도로 저장한다. Valid20 결과는 공식 Test 점수가 아니다. bootstrap 구간은 이미지 불확실성이고 학습 seed 변동 구간은 아니다. 일부 seed 실패 시 완료된 paired seed만 평균내고 partial 표시를 붙인다.

중간 결과표는 다른 터미널에서 `python summarize_seeds.py --plan runs/all_3seeds`로 갱신할 수 있다. Test10은 별도 ZIP을 동일 상위 폴더에 풀면 추가된다. 학습·Valid에는 Test10이 필요하지 않다.

서버 결과를 가져올 때 checkpoint 전체 없이 보고서만 묶을 수 있다. 마스크까지 보고 싶으면 `--masks`를 추가한다.

```bash
python pack_results.py --plan runs/all_3seeds --output all_3seeds_results.zip
```

`RESOURCE_ESTIMATES.csv`는 parameter 기반 저장량·메모리 하한을 보여준다. FP32 model+gradient+Adam states는 최소16bytes/parameter이고 activation은 추가다. 이 패키지의 last.pt에는 optimizer와 best_model 복사도 있어 best.pt와 합한 parameter 저장량은 약20bytes/parameter/run이다. 실제 디스크 파일은 buffer/history 등으로 더 커진다. 381run을 계획할 때 해당 표의 총량과 서버 여유 공간을 확인한다.

```bash
python run.py evaluate --checkpoint runs/all_3seeds/seed42/B/best.pt --data Data --split test --d4 --output test_predictions/B42 --device cuda
```

Test GT가 없으므로 mask만 생성한다. `verify.py`는 임의 tensor의 forward/backward/optimizer2step/추론 검사이며 학습 성능 검증이 아니다. 로컬 검증은 CPU이며 CUDA는 이 서버 사전 검사로 확인해야 한다. CPU 전체127개 소형 검사와 대표448/batch4 검사 범위·코드 hash는 `verification/`에 기록한다. Transformer는64, Inception은128, PAN은256 이상의 synthetic 입력을 사용한다. 실제 학습은 모두448이다. 전체127개448메모리를 로컬에서 검증한 것은 아니다.

원 논문 링크와 데이터 특성 분석은 `docs/`에 동봉했다. Eutectic Si는 면적이 약7%이며 독립 입자도 있어 전체 연결을 강제하는 topology loss를 적용하지 않았다. 서버에서 먼저 결과를 얻고 seed 재현·상별 오류·가시적 마스크를 함께 보고 채택한다.
