# KoMaP U-Net + Transformer 서버 실험 패키지

HT_BOT, HT_PYR, HT_DEC16 모델과 학습·평가·중단 후 재개 코드를 구현했다. 동일 조건 비교용 B 모델, 실행 스크립트와 검증 결과를 코드 ZIP에 포함한다. GitHub 배포에는 데이터·학습 checkpoint를 포함하지 않는다. 권한이 있는 Train70/Valid20 데이터를 Data/train 및 Data/valid에 별도로 준비한다. 사전학습 가중치와 Python 의존성은 서버에서 내려받는다.

CPU 합성 입력에서 네 모델의 forward/backward/AdamW/슬라이딩 윈도 평가, B와 초기 가중치·출력 일치, 새 분기의 난수 격리를 검증했다. HT_BOT의 동결5 → 해제6 epoch 경계를 포함한 실제 중단·재개 테스트도 통과했다. **H100에서의 CUDA448/batch4 검증과 새 모델의 실제 데이터 학습은 아직 수행하지 않았다.** CPU 검증 환경은 Python3.9.6 / torch2.8.0 / torchvision0.23.0이다. 서버 권장 환경은 Python3.10–3.12 / torch2.7.1+cu128 / torchvision0.22.1이다.

## 1. 서버에서 설치·검증

GPU 드라이버가 CUDA12.8 PyTorch 빌드를 지원하는 Linux 서버에서 실행한다. 기존 결과를 보존하고 새로운 영구 저장 경로를 사용한다. 아래 `/persistent`는 서버의 실제 저장 경로로 바꾼다. 초기 9회 기준 새 작업 공간은 최소30GB, 후속까지 포함하면40GB 이상을 권장한다. 기존32회 결과와 설치·가중치 캐시 공간은 별도다.

```bash
unzip KoMaP_UNet_Transformer_Code_20261008.zip -d /persistent/komap
cd /persistent/komap/KoMaP_UNet_Transformer_Code_20261008
# 학습 전에 승인된 데이터의 train/valid 폴더를 이 폴더의 Data/ 아래에 준비한다.
PYTHON_BIN=python3.11 bash setup_server.sh
export TORCH_HOME=/persistent/komap_torch_cache
.venv/bin/python prefetch_weights.py
.venv/bin/python preflight.py --device cuda --size 448 --batch-size 4 --report checks/cuda_preflight.json
```

`prefetch_weights.py`는 공식 ResNet101 ImageNet V2와 Swin-T ImageNet V1 파일을 내려받고 SHA256을 기록한다. pretrained 없이 생성하는 preflight는 네 모델 모두에서 실제 optimizer step 두 번과 슬라이딩 윈도 검증을 수행한다. 사용 가능한 GPU 총 VRAM의20% 여유를 요구한다. 검증에 실패하면 로그를 확인하고 선택한 FP32/batch 조건을 만족하는 GPU에서 다시 실행한다. 배치 크기나 AMP를 바꾸려면 비교용 B도 같은 새 조건으로 별도 재측정해야 한다.

원하면 서버의 지정 라이브러리 환경에서 재개 통합 테스트도 실행한다. 임시 체크포인트에 약6GB 공간이 필요하며, 64 입력 합성 데이터만 사용한다. CUDA 재개 테스트에서는 정확한 비교를 위해 deterministic=True를 사용하며, 실제150 epoch 실험의 기본 deterministic=False 설정은 유지한다.

```bash
.venv/bin/python verify_resume.py --device cuda --report checks/cuda_resume.json
```

## H100에서 배치를 늘리려면

H100에서는 메모리 여유를 측정해 batch8/16을 선택할 수 있다. 아래 명령은 batch4/8/16마다 별도 프로세스로 네 모델을 검사하고, VRAM peak·step 시간·images/sec를 기록한다. OOM이나20% 여유 기준 실패 시 해당 배치 로그를 남긴다. 결과의 `largest_passing_batch`는 메모리 적합성 기준이며 성능 최적 배치라는 뜻은 아니다.

```bash
.venv/bin/python profile_batches.py
cat checks/batch_profiles/summary.json
```

예를 들어 batch16이 통과했다면 동일 배치의 검증 보고서를 선택하고, 새 출력 폴더에서 B도 함께 재측정한다.

```bash
cp checks/batch_profiles/batch16.json checks/cuda_preflight.json
BATCH_SIZE=16 OUTPUT_DIR=/persistent/komap_runs/transformer_batch16 nohup bash start_server.sh > /persistent/komap_runs/transformer_batch16.log 2>&1 &
```

후속 실험도 `BATCH_SIZE=16`과 동일 OUTPUT_DIR를 지정한다. `run_suite.py --batch-size 16`도 지원한다. 같은 폴더의 배치를 바꾸면 identity 검사에서 거부한다. BatchNorm과 optimizer update 수가 달라지므로 예전 batch4 B 점수와 직접 비교하지 않는다. epoch당420개 샘플을 유지하며, 마지막 작은 배치를 버리지 않는다. 4/8/16에서 epoch당 step은 각각105/53/27이고, batch8/16의 마지막 배치는4개다. 큰 배치의 일반화 성능이 항상 더 좋은 것은 아니다.

## 2. 1차 실험 실행

```bash
mkdir -p /persistent/komap_runs
OUTPUT_DIR=/persistent/komap_runs/transformer_20261007 nohup bash start_server.sh > /persistent/komap_runs/transformer_20261007.log 2>&1 &
```

기본 `primary` 큐는 **B 3회 + HT_BOT 3회 + HT_PYR 3회 = 9회**, seed42/43/44, 각150 epoch다. seed42 B→BOT→PYR, seed43 B→BOT→PYR, seed44 B→BOT→PYR 순서로 실행한다. 검증된 완료 실험은 건너뛰고, 진행 중이던 실험은 `last.pt`의 마지막 완료 epoch부터 재개한다. 서버나 세션이 종료되면 위 명령을 같은 OUTPUT_DIR로 다시 실행한다. 실행 스크립트는 GPU 세션을 종료하지 않는다.

기존 H100 B의 집계 점수79.8202%만으로는 이 코드 및 선택 배치와의 완전한 일치를 확인할 수 없어 기본값은 B 재측정이다. 기존 패키지의 checkpoint를 새 모델의 시작 가중치로 쓰지 않는다. `--skip-baseline`은 이 새 패키지·동일 환경·동일 출력 폴더에서 이미 완료한 B 3회를 확인한 경우에만 허용된다.

## 3. 후속 HT_DEC16 실행

1차 결과를 확인한 뒤 같은 OUTPUT_DIR에서 실행한다. B 3회가 검증되어 있어야 한다.

```bash
PHASE=followup OUTPUT_DIR=/persistent/komap_runs/transformer_20261007 nohup bash start_server.sh > /persistent/komap_runs/transformer_dec16_20261007.log 2>&1 &
```

HT_DEC16 seed42/43/44, 각150 epoch **3회**를 추가한다. 처음부터 세 구조 모두 실행하려면 첫 실행의 `PHASE=all`을 지정한다. 이때 B까지 총12회이며, 같은 출력 폴더에서 동시에 두 큐를 실행할 수 없다.

## 4. 상태·결과 확인

```bash
cat /persistent/komap_runs/transformer_20261007/suite_status.json
tail -n 30 /persistent/komap_runs/transformer_20261007/logs/seed42_HT_BOT.log
.venv/bin/python summarize.py --output /persistent/komap_runs/transformer_20261007
```

각 `seed42/HT_BOT` 등의 폴더에 config/data manifest/environment, `history.csv`, `status.json`, `best.pt`, `last.pt`, `summary.json`, Single·D4 평가와 Valid 예측 마스크를 저장한다. `best.pt`는 Single Valid mIoU로 선택한다. D4는 그 동일 checkpoint에서8-view TTA로 계산한다. `last.pt`는 모델·optimizer·scheduler·best model·전역/loader/분기 RNG·동결 상태를 저장한다. epoch 중간 재개는 마지막 완료 epoch부터 해당 epoch을 다시 수행한다.

`reports/comparison.csv`, `seed_aggregates.json`, `RESULTS_KO.md`에서 같은 seed의 B 대비 차이, 이미지 단위 paired bootstrap10,000회, seed별 차이 표준편차를 확인한다. 승격 기준은3개 seed 모두 완료, 평균 D4 +0.10%p 이상,2/3 seed 이상 개선, Eutectic IoU·skeleton recall 평균 저하 없음, 두 방향 Al3Ni/Eutectic 혼동·contact error 평균 증가 없음이다. Bootstrap은 이미지 불확실성이고 seed 변동과 별도로 해석한다. 같은 Valid로 구조를 선택하므로 탐색 결과이며 공식 Test 성능을 주장하지 않는다.

## 5. 모델과 고정 조건

| ID | 구조 | 파라미터 | 실행 단계 |
|---|---|---:|---|
| B | ImageNet ResNet101 + 기존 scSE U-Net | 51,611,179 | 비교 기준3회 |
| HT_BOT | CNN + Swin-T, 1/32 bottleneck residual fusion | 81,822,117 | 1차3회 |
| HT_PYR | CNN + Swin-T, 1/8·1/16·1/32 residual fusion | 82,665,253 | 1차3회 |
| HT_DEC16 | U-Net 첫 decoder의1/16 feature에2개 attention block | 53,256,491 | 후속3회 |

CNN/decoder/fusion LR3e-4, Swin LR3e-5, AdamW wd1e-4, Cosine150. Swin은 epoch1–5 eval/frozen, epoch6부터 train/unfrozen이다. 처음부터 optimizer에 Swin 그룹을 포함하여 재개할 때 그룹 순서를 유지한다. 새 residual 마지막 projection을0으로 초기화하여 B 출력에서 시작한다. Decoder attention은 scratch 초기화, width256, heads4, FFN1024, dropout0, 고정2D sinusoidal position이다.

Context448 → center target224, stride112, epoch당420 draws, 기본 batch4(선택8/16), workers0, FP32를 사용한다. grayscale을 RGB로 반복해 각 encoder에서 ImageNet normalization을 한 번 적용한다. CE0.5 + Dice0.25 + Lovasz0.25, class weights[1,1,2,1], 클래스 순서 Primary Si/Al3Ni/Eutectic Si/Al이다. mIoU는 이미지별4클래스 IoU 평균을20장에 대해 평균하며, union0인 클래스는 IoU1이다. 골격은 Zhang-Suen 정확한 pixel match, contact는 클래스1/2 직접4-neighbor 접촉 주변 radius2이다.

학습 코드는 `komap_phase/hybrid.py`, `training.py`, 큐는 `run_suite.py`, 실험 설정은 `configs/`에 있다. 기존 데이터·loss·평가·baseline model 파일은 원본과 바이트 단위로 동일하며 별도 wrapper에서 Transformer를 결합한다. 이 구조들은 논문에서 착안한 우리 프로토콜용 변형이다.

- [TransFuse](https://arxiv.org/abs/2102.08005): CNN/Transformer 병렬 경로·다중 해상도 결합의 근거. 여기서는 BiFusion 대신 초기 출력 보존 residual fusion을 구현했다.
- [Swin Transformer](https://arxiv.org/abs/2103.14030): shifted-window 계층형 encoder. torchvision의 공식 pretrained Swin-T를 사용한다.
- [TransUNet](https://arxiv.org/abs/2102.04306): CNN 특징·Transformer·U-Net skip 결합 계열의 근거.
- [UTNet](https://arxiv.org/abs/2107.00781): U-Net encoder/decoder attention 배치의 근거. HT_DEC16은 작은 dense attention이며 UTNet의 efficient attention/RPE 재현은 아니다.

## 6. 개별 실행·평가

큐 실행과 별도로 새 폴더에서 한 모델만 학습할 수 있다. 설정 파일의 기본 seed는42이며,43/44는 큐가 자동 생성한다.

```bash
.venv/bin/python run.py train --config configs/HT_BOT.json --data Data --output /persistent/komap_runs/bot_single_seed42 --device cuda
.venv/bin/python run.py train --config configs/HT_BOT.json --data Data --output /persistent/komap_runs/bot_single_seed42 --device cuda --resume
.venv/bin/python run.py evaluate --checkpoint /persistent/komap_runs/transformer_20261007/seed42/HT_BOT/best.pt --data Data --output /persistent/komap_runs/bot_eval_d4 --device cuda --d4
```

소스·config·데이터 hash·GPU/PyTorch 등 환경이 다르면 학습 재개를 거부한다. 기존 결과 파일을 덮어쓰는 대신 별도 실험 폴더를 사용한다. README를 포함한 배포 파일의 SHA256은 `RELEASE_MANIFEST.json`, 구현 검증 결과는 `checks/cpu_preflight.json`, `checks/cpu_resume.json`에 있다. 큐의9회/후속3회 실행 순서, 완료 실험 생략, batch8 전달 및 배치 변경 거부 검증은 `checks/cpu_queue.json`에 있다. `docs/`에는 이전 H100 결과와 실험 장부의 전달 시점 스냅샷을 넣었다. 코드 ZIP에는 데이터 이미지·마스크, 학습 checkpoint, 사전학습 weight cache, 기존 서버 실행 프로세스를 포함하지 않는다. Git clone에서 직접 실행하는 경우에도 Data를 준비하고 같은 설치·검증 명령을 사용한다.

코드 ZIP: [다운로드](../../releases/KoMaP_UNet_Transformer_Code_20261008.zip). 최신 전체 실험 결과: [총정리](../../docs/experiment_snapshot_20261008/TOTAL_SUMMARY_KO.md).
