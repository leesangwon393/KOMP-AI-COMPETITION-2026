**KoMaP 2026 개인 실험 공유 — 최고 완료 모델 (2026-09-29)**

현재까지 평가가 완료된 실험의 최고 기록은 **ImageNet ResNet34 U-Net + scSE + Eutectic clDice 보조 loss + D4 TTA**, Valid mIoU **78.2436%**이다. 단일 모델이며 모델 간 앙상블은 사용하지 않았다. DySample 실험은 이 문서 작성 시 진행 중이다.

**결과**

| 지표 | 일반 sliding-window | D4 TTA |
|---|---:|---:|
| mIoU | 77.8915% | **78.2436%** |
| Primary Si IoU | 89.8081% | 89.9791% |
| Al₃Ni IoU | 71.8037% | 72.3923% |
| Eutectic Si IoU | 56.4585% | 56.9272% |
| Al IoU | 93.4956% | 93.6758% |

100 epoch 학습 중 일반 검증 mIoU 최고인 **84 epoch** 체크포인트를 선택한 뒤 D4로 평가했다. D4 적용 효과는 +0.3522%p이다. 학습·매 epoch 일반 검증의 기록 시간 합은 약 35.6분이며, 최종 재평가·D4·체크포인트 저장 등의 전체 실행 시간과는 다르다.

**모델 구조**

- 입력: grayscale 1채널, 224×224 원본 해상도 패치.
- Encoder: ImageNet pretrained ResNet34, 전체 fine-tuning. RGB 첫 convolution 가중치를 채널 방향으로 합해 1채널로 변환.
- Decoder: U-Net, bilinear upsampling → encoder skip과 채널 concat → 3×3 Conv/BN/ReLU 두 번 → scSE.
- Decoder 채널: 256 → 128 → 64 → 64 → 32. 다섯 decoder block에 scSE 적용.
- scSE: 채널 attention과 공간 attention의 출력을 합산.
- 마지막 112→224 단계는 bilinear + convolution이며 원본 해상도 skip은 없음.
- 출력: 1×1 convolution으로 4클래스 logits. 약 2,452.8만 parameters.

**데이터·패치 추출·증강**

Train 70장, Valid 20장을 사용했다. 이미지를 통째로 224로 축소하지 않고, 원본 이미지에서 224×224 패치를 추출했다. 후보 위치의 stride는 112이며 가장자리도 포함한다.

- 매 epoch 총 420개 패치를 확률적으로 추출한다. 이미지당 정확히 6개를 뽑는 방식은 아니다.
- 50% 확률로 Eutectic 면적 비율 ≥10%인 후보 pool을 선택한다.
- 나머지 50%는 전체 후보 pool을 선택한다. 전체 pool에도 Eutectic-rich 패치가 포함된다.
- 선택한 pool 안에서 원본 이미지를 균등 선택하고, 해당 이미지의 후보 패치를 균등 선택한다.
- 증강: 0/90/180/270° 회전과 선택적 반전으로 D4 기하 증강.
- 입력을 [0,1]로 변환한 뒤 mean 0.674325, std 0.229455로 정규화한다. 이 정규화 수치는 기존 설정을 유지한 것이다.
- 본 실험에서는 brightness/gamma/noise/blur 증강을 적용하지 않았다.

**Loss**

`L = 0.5 × CE + 0.5 × weighted Dice + 0.05 × r(epoch) × clDice_Eutectic`

- CE: 별도 클래스 가중치 없음.
- Dice: Primary Si / Al₃Ni / Eutectic Si / Al 순서로 [1,1,2,1] 가중치를 적용하고 가중치 합으로 정규화.
- clDice: Eutectic softmax 확률과 정답의 soft skeleton 겹침을 이용하는 보조 loss. Eutectic에만 적용한다.
- soft skeleton 반복 수 8. 정답 Eutectic이 없는 패치는 clDice 평균에서 제외한다.
- r(epoch): 1–10 epoch는 0, 11–20 epoch에 0.1→1.0으로 증가, 이후 1.0.

목적은 얇은 Eutectic 구조의 겹침을 보완하는 것이다. clDice 아이디어를 현재 분할 모델에 적용한 것으로, 원 논문 전체 구조 재현이나 모든 Eutectic을 하나로 연결하는 제약은 아니다. 보조 loss이므로 추론 시 모델 구조나 연산은 기준 모델과 동일하다.

**학습·평가 설정**

| 항목 | 설정 |
|---|---|
| Epochs / best epoch | 100 / 84 |
| Batch size / gradient accumulation | 4 / 없음 |
| Optimizer | AdamW |
| Initial LR / weight decay | 3e-4 / 1e-4 |
| Scheduler | CosineAnnealingLR, T_max=100 |
| Seed | 42 |
| Device / precision | Apple MPS / FP32, AMP 미사용 |
| Early stopping | 미사용, 매 epoch best 및 last 저장 |
| Validation | 원본 해상도, tile 224 / stride 112 (overlap 112) |
| Overlap 병합 | 중첩 위치의 클래스 확률 평균 |
| D4 TTA | 회전·반전 8개 view를 역변환한 뒤 확률 평균 |

지표는 **각 이미지의 4클래스 mIoU를 구한 뒤 Valid 20장 평균**이다. 해당 이미지에서 정답과 예측에 모두 없는 클래스(union=0)는 IoU=1로 처리한다. 이 정책은 수치에 영향을 주므로 다른 실험과 비교할 때 반드시 일치시켜야 한다. D4는 추론 기법이며, 체크포인트 선택은 일반 검증 점수로 수행했다.

**clDice 추가 효과와 재현성**

동일 seed 42의 기존 scSE 모델 대비 D4 mIoU는 78.0264→78.2436% (+0.2173%p), Eutectic IoU는 56.4462→56.9272% (+0.4810%p)였다.

| Seed | 기준 모델 D4 mIoU | clDice D4 mIoU | 차이 |
|---|---:|---:|---:|
| 42 | 78.0264% | 78.2436% | +0.2173%p |
| 123 | 77.9288% | 77.9841% | +0.0553%p |
| 2026 | 77.9030% | 77.7976% | −0.1055%p |
| 평균 | 77.9527% | 78.0084% | +0.0557%p |

3 seed의 mIoU 표본 표준편차는 기준 모델 0.0651%p, clDice 0.2240%p이다. Eutectic IoU는 세 seed 모두 개선됐고 평균 56.3598→56.6195% (+0.2597%p)였다. 따라서 **Eutectic 개선 경향은 관찰됐지만 전체 mIoU 이득은 작고 seed에 따라 달랐다.**

최고 단일 기록과 평균 성능을 구분해야 한다. 모든 수치는 로컬 Valid 결과이며 공식 Test 점수가 아니다. 동일 Valid에서 여러 모델과 epoch를 선택했으므로 선택 편향이 존재할 수 있다.

**저장 위치**

- 최고 체크포인트: `runs/research_sweep_v1/cldice/best.pt`
- 학습 설정·곡선: 같은 폴더의 `config.json`, `history.json`
- 일반 평가: `validation_metrics.json`
- D4 평가·예측 마스크: `d4_tta/metrics.json`, `d4_tta/*.png`
- 당시 코드: `runs/research_sweep_v1/source_snapshot/`
- Seed 재검증: `runs/cldice_seed_validation/RESULTS_KO.md`

출처 논문: [clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html).
