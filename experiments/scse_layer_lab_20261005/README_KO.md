# scSE 층별 recurrent residual · 공간 attention 실험

패키지: `KoMaP_SCSE_Layer_Spatial_Lab_20261005.zip`

이 ZIP 하나에 실행 코드, 설정, 원본 Train70/Valid20/Test10 데이터,
원본 RRCU 코드 참조본, 로컬 검증 결과가 들어 있다.
기존 학습 checkpoint는 포함하지 않는다. 각 run은 공개 ImageNet ResNet101 V2에서 fresh 시작한다.
첫 실행에는 가중치 다운로드를 위한 인터넷 연결이 필요하며, 이후 `torch_cache`를 재사용한다.

## 서버에서 바로 실행

Linux + NVIDIA GPU, Python3.10–3.12를 사용한다. L4 24GB급 GPU를 기준으로 FP32/batch4를 유지한다.
최대32회 checkpoint 보관을 위해 실행 폴더가 있는 디스크에 최소40GiB 여유를 확보한다.

```bash
unzip KoMaP_SCSE_Layer_Spatial_Lab_20261005.zip
cd KoMaP_SCSE_Layer_Spatial_Lab_20261005
bash setup_server.sh
nohup bash start_server.sh > launch.log 2>&1 < /dev/null &
```

Python3.11을 별도 지정하려면 `PYTHON_BIN=python3.11 bash setup_server.sh`를 사용한다.
setup은 이 폴더의 `.venv`에 torch2.7.1/torchvision0.22.1 CUDA12.8,
numpy1.26.4/Pillow11.2.1을 설치한다. 실행 전에 실제 CUDA 사용 가능 여부를 확인한다.
패키지를 풀고 설정이나 소스를 수정하지 않은 상태에서 실행한다.

기본 명령은 **층별 탐색 + 공간 attention 탐색 + 추가seed 검증**을 순차 실행한다.
GPU 하나에는 학습 프로세스 하나만 둔다.

```bash
# GPU가 여러 개라면 사용 가능한 한 장을 지정
CUDA_VISIBLE_DEVICES=1 nohup bash start_server.sh > launch.log 2>&1 < /dev/null &

# B/R03 재현 6회까지만 먼저 실행
nohup bash start_server.sh --stop-after reproduce > launch.log 2>&1 < /dev/null &

# 재현 검증 뒤 동일 실행 폴더에서 나머지 전체를 계속 실행
nohup bash start_server.sh --resume > launch_resume.log 2>&1 < /dev/null &

# 공간 후보 없이 최초 층별 실험만 실행: 최대20회
nohup bash start_server.sh --skip-spatial > launch.log 2>&1 < /dev/null &
```

`--skip-spatial`은 실행 폴더에 고정된다. 이 옵션으로 시작한 run을 재개할 때도 같은 옵션을 붙인다.
`--stop-after reproduce/layers/all`은 단계 종료 지점이므로 재개하면서 변경할 수 있다.
다른 프로토콜을 비교하려면 새 `--output` 폴더를 사용한다.
여러 GPU에서 같은 output을 동시에 실행하면 잠금으로 거절한다.

## 실제 구현된 실험

기존 R03은 `decoder.blocks[3]`의 scSE 채널 logit에
`0.1 × recurrent residual branch`를 더한 모델이다.
각 신규 gate의 최종 projection을0으로 초기화해 시작 시 B와 출력이 같게 한다.
기존 encoder·decoder·scSE 가중치, 전역 sampling RNG는 추가 모듈 할당으로 바뀌지 않는다.
추가 모듈은 seed+2301의 독립 초기화를 사용한다. 동일 폭의 위치는 초기 branch 가중치도 같다.
C3의 초기 가중치는 원본 R03과 tensor별 일치를 확인했다.
채널 hidden 폭은 `max(4,C/16)`이다. 32ch final도 hidden4를 쓰고, 원본 R03의64ch/hidden4는 유지한다.

| 단계 | ID | 비교 내용 | 새 학습 횟수 |
|---|---|---|---:|
| 1 재현 | B, C3 | 원본 scSE 대조군과 기존 R03; seed42/43/44 | 6 |
| 2 위치 | C0,C1,C2,C4 | 채널 분기를 한 위치에만 적용; seed42; C3 결과 재사용 | 4 |
| 3 기전 | M_T1 | 위치1위에서 반복2→1; 가중치·파라미터 수 동일 | 1 |
| 3 기전 | M_UNSHARED | 위치1위에서2회 weight sharing 해제; 추가 파라미터 기록 | 1 |
| 3 기전 | M_SPATIAL | 위치1위에서 공간 recurrent 분기만 추가 | 1 |
| 3 기전 | M_BOTH | 위치1위에서 채널+공간 recurrent 분기 | 1 |
| 4 조합 | C_TOP2,C_ALL | 상위2위치 조합 /5개 위치 전체 | 2 |
| 4 확인 | 위 후보 중 상위2개 | seed43/44 추가 확인; 이미 완료된 같은 run 재사용 | 최대4 |
| 5 공간 대조군 | 위치1위 C_i | 공간 탐색의 같은 위치 채널 대조군 seed43/44 확보 | 최대2 |
| 5 공간 | S_CBAM,CS_CBAM | CBAM 파생 공간 logit: 단독/채널 RRCU 결합 | 2 |
| 5 공간 | S_COORDINATE,CS_COORDINATE | Coordinate Attention 파생 공간 logit: 단독/채널 결합 | 2 |
| 5 공간 | S_MULTISCALE,CS_MULTISCALE | 다중 크기 depthwise 공간 logit: 단독/채널 결합 | 2 |
| 5 확인 | 공간 기법1종의 S/CS 쌍 | CS 성능으로 기법 선정, 단독/결합 둘 다 seed43/44 | 4 |

중복 완료 run을 재사용하므로 **최대32회 ×150epoch =4800epoch**다.
실제 횟수는 선택된 후보에 따라 줄어든다.
전체 층별 최적 조합을 전수 탐색하는 것은 아니며, 위치·기전·조합을 단계적으로 탐색한다.

| 위치 | 코드 | 출력 해상도 | 채널 |
|---|---|---|---:|
| C0 | decoder.blocks[0] | 입력의1/16 | 256 |
| C1 | decoder.blocks[1] | 입력의1/8 | 128 |
| C2 | decoder.blocks[2] | 입력의1/4 | 64 |
| C3 | decoder.blocks[3] | 입력의1/2 | 64 |
| C4 | decoder.final | 입력과동일 | 32 |

위치 순위는 seed42의 D4 mIoU를 사용하고, 동률은 ID 순으로 고정한다.
최고 checkpoint는 매 run의 Single 검증 점수로 선택한다.
선택된 위치와 최종 후보를 `position_selection.json`/`final_selection.json`/`spatial_selection.json`에 저장한다.
재개 시 이 선택을 바꾸거나 완료 run을 다시 학습하지 않는다.

## 공간 attention의 실제 구조와 가설

세 후보 모두 기존 scSE 공간 logit `z_s`를 보존하고
`s = sigmoid(z_s + 0.1 × delta_s)`로 보정한다.
S는 공간 branch만 추가한다. CS는 같은 위치의 원본 채널 RRCU와 공간 branch를 결합한다.
기존 scSE의 병렬 결합 `x*c + x*s`를 유지한다.

| 후보 | 구현된 delta_s | 우리 데이터에서 확인할 가설 |
|---|---|---|
| CBAM 파생 | 채널 평균·최댓값2개 map→7×7 convolution→1개 공간 logit | 픽셀별1×1 sSE보다 주변 형태 정보가 얇은 조직과 입자 구별에 도움이 되는가 |
| Coordinate 파생 | 가로·세로1D pooling→공유1×1 bottleneck→방향별 scalar logit 합산 | 긴 구조의 위치 정보가 Eutectic 연결부를 보존하는가 |
| 다중 크기 depthwise | DW3×3 + DW5×5(dilation2)→concat→1×1 공간 logit | 국소 경계와 넓은 문맥을 동시에 보면 접촉부 혼동이 줄어드는가 |

이는 **논문에서 아이디어를 가져온 scSE용 파생 구현**이다.
전체 CBAM의 순차 채널·공간 구조, 원본 CA의 채널별 좌표 map,
VAN/LKA 전체 구조를 재현한 모델이라고 표시하지 않는다.
다중 크기 구조는 자체 제안이며, 큰 커널로 얇은 경계가 흐려지는지 확인해야 한다.
Coordinate 구조는 우리 조직의 방향 다양성 때문에 회전/반전 조건에서도 결과를 확인한다.
성능 상승은 아직 관측되지 않은 가설이다.

논문:
- CBAM: https://arxiv.org/abs/1807.06521
- Coordinate Attention: https://arxiv.org/abs/2103.02907
- 큰 문맥 attention의 참고인 VAN/LKA: https://arxiv.org/abs/2202.09741
- scSE: https://arxiv.org/abs/1803.02579
- R2U-Net: https://arxiv.org/abs/1802.06955

## 모든 run에서 고정한 조건

- ResNet101 ImageNet V2, fresh 시작, FP32, batch4, 각150epoch.
- context448/중앙 감독224, stride112, epoch당420개 패치.
- AdamW lr3e-4/weight_decay1e-4,150epoch cosine.
- CE0.5 + Dice0.25 + Lovasz0.25, Eutectic 가중치2.
- Al3Ni/Eutectic-rich 패치50%, 대상 클래스·원본 이미지 균형 sampling.
- 원본 광도 증강: 밝기±.08·대비.9–1.1(p=.5), 감마.9–1.1(p=.3), Gaussian noiseσ.01(p=.15).
- 회전0/90/180/270°와 좌우 반전; image/mask 동일 기하 변환.
- Valid20 원해상도 halo 검증; 빈 클래스1; 이미지별4클래스 IoU 평균.
- Single로 checkpoint 선택 후 같은 checkpoint의 Single/D4 평가.
- Test는 학습·구조 선택·성능 평가에 사용하지 않는다.

M_UNSHARED는 파라미터 수가 늘어난다. 따라서 차이를 recurrence sharing 효과만으로 단정하지 않는다.
M_T1은 t=2와 파라미터 수가 같아서 반복 횟수 비교에 쓴다.
같은 Valid20을 반복해서 후보를 선택하므로 독립 Test 우위로 주장하지 않는다.
이미지 bootstrap95%와 seed별 Δ 평균/표준편차를 구분해서 해석한다.

## 진행 확인·결과 모으기

```bash
tail -f launch.log
watch -n 1 nvidia-smi
cat runs/layer_search/suite_status.json
tail -f runs/layer_search/logs/seed42_B.log

# 완료된 결과만 언제든 다시 집계
.venv/bin/python summarize.py --output runs/layer_search
```

각 `runs/layer_search/seed{seed}/{ID}/` 아래에 config,환경/코드hash/데이터hash,
150epoch 이력, best.pt/last.pt, Single/D4 저장 마스크, 지표를 보관한다.
결과는 `reports/comparison.csv`, `reports/seed_aggregates.json`, `reports/RESULTS_KO.md`에 모인다.
CSV에는 같은 seed B 대비 차이, 이미지 bootstrap95%, 클래스IoU,
Eutectic precision/recall/얇은 구조 recall, 접촉부 오류, VRAM/파라미터/평가 시간이 있다.
CS 행에는 같은 위치 채널 단독·같은 기법 공간 단독 대비 차이도 추가한다.
평가 시간은 타일·TTA·진단을 포함한 시간으로 단일 forward latency와 구분한다.

## 중단·재개

```bash
nohup bash start_server.sh --resume > launch_resume.log 2>&1 < /dev/null &
```

실패한 run이 있으면 다음 run을 시작하지 않는다. 해당 log를 확인한 뒤 재개한다.
마지막 완료 epoch의 model/optimizer/scheduler/RNG를 복구한다.
첫 checkpoint 이전에 종료됐으면 그 기록을 `pre_epoch_archives/`로 보관하고 fresh 재시작한다.
코드·데이터·설정·GPU 모델·torch 버전이 바뀌면 혼합 재개를 거절한다.
완료된 결과는 건너뛰고, 단계별 선택과 진행표를 복구한다.

GPU 여러 장에 분배하려면 각 서버에서 별도 output을 사용한다.
이번 supervisor는 하나의 GPU에서 순차 실행하도록 구성했으므로 기본 명령을 여러 GPU에
그대로 복제하면 동일 실험을 중복 실행한다.

## 검증 범위

로컬에서는 ResNet101의 합성 forward/backward와 각 branch gradient,
기존 B 초기 가중치·출력·sampling RNG 일치, C3의 원본 R03 가중치 일치를 확인한다.
실제 Train 데이터로 동일 crop·증강 pairing을 확인한다.
별도 합성 ResNet34 CPU 학습에서 중단·재개 후 모델과 loss/점수가 연속 실행과 일치하는지 검증한다.
단계 선택·완료 건너뛰기·변경 거부는 fixture worker로 검증한다.
로컬 검증은 합금 성능 실험 결과가 아니다.
서버 시작 시 SHA256 검증과 CUDA448/batch4 optimizer preflight를 통과해야 학습한다.

```bash
# 필요할 때 로컬 검증을 다시 실행
.venv/bin/python verify_architectures.py --device cpu --size 64 --batch-size 2 --data Data --report checks/cpu_recheck.json
.venv/bin/python verify_workflow.py --data Data --report checks/workflow_recheck.json
```
