# 실험 총정리

집계 시점: 2026-10-08T00:22:47+09:00

현재 실행 큐는 53/103회 완료다. H100 gate suite는 사용자 전달 보고서 기준32/32 완료다.
전체 장부는151행이며, 과거 학습·재평가·팀 보고·3seed 집계가 중복 범위를 포함한다. 이를 독립 학습 횟수로 합산하지 않는다.

현재 주요 프로토콜은 Train70/Valid20, context448/center224, 150epoch, batch4, FP32다. D4는 Single로 선택한 동일 checkpoint의8-view TTA다. 점수는 이미지별4클래스 IoU 평균의20장 평균이고 union0 클래스는 IoU1이다. 과거100/200epoch 및 평가 조건 불일치 기록은 따로 보존했다. 새 마스크 전체 재평가는 수행하지 않았다.

## 현재 판단

- 기본 구조는 ImageNet ResNet101 + scSE U-Net을 유지한다. H100 gate 변형은3seed에서 뚜렷한 개선이 없었다.
- D03 접촉부 CE는 Mac3seed에서 개선되었으나 별도 서버 보고3seed에서 평균 하락했다. 환경·코드·초기화 등 차이를 통제하기 전에는 일반적인 개선으로 채택하지 않는다.
- L4 R03 증강에서는 contrast OFF와 noise 절반이3/3 seed에서mIoU 개선이다. 그러나 평균 contact error는 각각+0.1063/+0.3119%p 악화하여 전체 지표 기준의 자동 채택은 보류한다.
- X02_CONN은 Mac seed42에서 개선되었지만 아직1seed다. U2_CBAM_DEEP은 완료된seed42/44에서 개선되었지만 seed43이 남았다.
- 새 Transformer3종은 코드/README/ZIP 구현·CPU 검증을 완료했다. H100 CUDA 검증과 실제 데이터 학습은 시작하지 않았다.

## 현재 실행 큐

| 슬롯 | 실험 계열 | 완료 | 현재 작업 | 확인 수준 |
|---|---|---|---|---|
| gpu-l4 | U2 구조 seeds42/43 | 12/32 | seed42/U2_ECA validating epoch146 | 최근 GPU 실행 확인 |
| gpu-l4-2 | U2 구조 seed44 | 9/16 | — | 복원 연결 오류; 런타임 종료 여부 미확인 |
| gpu-l4-aug1 | R03 증강 | 18/36 | seed42/A02_BRIGHT_OFF training epoch111 | 최근 GPU 실행 확인 |
| gpu-l4-r03-repro | T4 R03 밝기 재현 | 5/6 | seed44/A01_BRIGHT_HALF validating epoch45 | 최근 GPU 실행 확인 |
| mac | Mac 후속 | 9/13 | seed42/L01_DICE35 epoch47 | 로컬 상태 파일 기준 |

## 완료된 같은-seed 비교

모든 Δ는 각 환경의 같은-seed baseline 대비 D4 차이다. n<3은 중간 결과다. 아래 평균은 학습 seed 평균이며 이미지 bootstrap은 CSV에 따로 저장했다.

| 계열 | 모델/변경 | 완료 seed | D4 평균(%) | 평균 Δ(%p) | 개선 seed | seed Δ SD(%p) |
|---|---|---|---|---|---|---|
| L4 R03 증강 | A01_BRIGHT_HALF | 42,43,44 | 79.6993 | +0.0847 | 1/3 | 0.2278 |
| L4 R03 증강 | A03_CONTRAST_OFF | 42,43,44 | 79.7854 | +0.1707 | 3/3 | 0.1650 |
| L4 R03 증강 | A05_GAMMA_HALF | 42,43,44 | 79.6576 | +0.0430 | 1/3 | 0.2112 |
| L4 R03 증강 | A07_NOISE_HALF | 42,43,44 | 79.7561 | +0.1414 | 3/3 | 0.1430 |
| L4 R03 증강 | A09_BLUR_WEAK | 42,43,44 | 79.4527 | -0.1620 | 1/3 | 0.2286 |
| L4 U2 구조 | B_SCRATCH | 42,44 | 78.3989 | -1.2956 | 0/2 | 0.4863 |
| L4 U2 구조 | U2_CBAM_DEEP | 42,44 | 79.8035 | +0.1090 | 2/2 | 0.0751 |
| L4 U2 구조 | U2_ECA_CBAM_DEEP | 42,44 | 79.7620 | +0.0676 | 1/2 | 0.4967 |
| L4 U2 구조 | U2_ECA_DEEP | 42,44 | 79.5822 | -0.1123 | 1/2 | 0.2330 |
| L4 U2 구조 | U2_ECA_SCSE_DEEP | 42,44 | 79.6584 | -0.0360 | 1/2 | 0.6124 |
| L4 U2 구조 | U2_PLAIN | 42 | 79.4986 | -0.2861 | 0/1 | — |
| L4 U2 구조 | U2_PLAIN_DEEP | 42,44 | 79.5637 | -0.1308 | 1/2 | 0.2142 |
| L4 U2 구조 | U2_SCSE_ALL | 42 | 79.3393 | -0.4454 | 0/1 | — |
| L4 U2 구조 | U2_SCSE_ALL_DEEP | 42,44 | 79.5734 | -0.1210 | 1/2 | 0.4364 |
| L4 U2 구조 | U2_SCSE_LAST2 | 42 | 79.7976 | +0.0129 | 1/1 | — |
| L4 U2 구조 | U2_SCSE_LAST2_DEEP | 42,44 | 79.3242 | -0.3703 | 0/2 | 0.2266 |
| Mac 후속 | D03 | 42,43,44 | 79.8054 | +0.2442 | 3/3 | 0.1085 |
| Mac 후속 | X01_FREQ | 42 | 79.4297 | -0.2006 | 0/1 | — |
| Mac 후속 | X02_CONN | 42 | 80.0115 | +0.3811 | 1/1 | — |
| Mac 후속 | X03_PIXEL | 42 | 79.5947 | -0.0356 | 0/1 | — |
| T4 밝기 재검증 | A01_BRIGHT_HALF | 42,43 | 79.6647 | -0.2236 | 0/2 | 0.0043 |

## D03 재현성 비교

| 근거 | seed | 평균 Δ(%p) | 개선 seed |
|---|---|---|---|
| Mac 원자료 | 42/43/44 | +0.2442 | 3/3 |
| 별도 서버 전달 보고 | 42/43/44 | -0.0647 | 1/3 |

환경 차이가 원인이라고 단정한 것은 아니다. 서버 원자료와 동일 코드/초기화/데이터/라이브러리로 검증하기 전에는 모순을 보존한다.

## 형태·접촉 지표

양수 Eutectic IoU/skeleton recall은 개선, 양수 contact error/상간 혼동은 악화다. 단위는%p. 전체 mIoU 상승만으로 채택하지 않는다.
Mac D03의 각seed 이미지 bootstrap CI는 모두0을 포함한다. U2_CBAM_DEEP의 완료된 두seed도 모두0을 포함한다. X02_CONN seed42는[+0.0305,+0.7465]%p이지만 아직1개 학습seed이며 같은Valid를 탐색에 사용했다.

| 계열 | ID | n | Eutectic IoU Δ | Skeleton recall Δ | Contact error Δ | Al3Ni→Eutectic Δ | Eutectic→Al3Ni Δ |
|---|---|---|---|---|---|---|---|
| L4 R03 증강 | A01_BRIGHT_HALF | 3 | +0.5814 | +0.7500 | -0.3914 | +0.2650 | -0.3718 |
| L4 R03 증강 | A03_CONTRAST_OFF | 3 | +0.4558 | +0.2103 | +0.1063 | +0.0037 | +0.0189 |
| L4 R03 증강 | A05_GAMMA_HALF | 3 | +0.3253 | +1.2222 | -0.3392 | +0.2800 | -0.4647 |
| L4 R03 증강 | A07_NOISE_HALF | 3 | +0.4816 | +0.0843 | +0.3119 | -0.0545 | +0.0426 |
| L4 R03 증강 | A09_BLUR_WEAK | 3 | -0.0710 | -0.3003 | +0.5695 | -0.1212 | +0.5297 |
| L4 U2 구조 | B_SCRATCH | 2 | -2.3422 | -0.3847 | +0.3429 | +1.5305 | -0.6290 |
| L4 U2 구조 | U2_CBAM_DEEP | 2 | +0.2555 | +1.2209 | -0.8105 | +1.0941 | -1.3836 |
| L4 U2 구조 | U2_ECA_CBAM_DEEP | 2 | -0.0712 | +0.1771 | -0.4221 | +0.0830 | -0.2745 |
| L4 U2 구조 | U2_ECA_DEEP | 2 | -0.0995 | +1.2749 | -0.7505 | +1.7541 | -1.5567 |
| L4 U2 구조 | U2_ECA_SCSE_DEEP | 2 | +0.4068 | +2.9682 | -1.3180 | +2.2843 | -2.5035 |
| L4 U2 구조 | U2_PLAIN | 1 | -0.7812 | -0.0149 | -0.1141 | +1.6889 | -1.0768 |
| L4 U2 구조 | U2_PLAIN_DEEP | 2 | -0.0062 | +1.5627 | -0.9380 | +1.8244 | -1.6672 |
| L4 U2 구조 | U2_SCSE_ALL | 1 | -0.8800 | +0.2347 | -0.1495 | +1.8947 | -1.4462 |
| L4 U2 구조 | U2_SCSE_ALL_DEEP | 2 | -0.1085 | +2.0475 | -1.0115 | +1.9514 | -1.9848 |
| L4 U2 구조 | U2_SCSE_LAST2 | 1 | -0.1541 | -0.6369 | -0.1085 | +1.3198 | -0.6402 |
| L4 U2 구조 | U2_SCSE_LAST2_DEEP | 2 | -0.7004 | +0.6698 | -0.7603 | +1.6677 | -1.4622 |
| Mac 후속 | D03 | 3 | +0.2208 | +0.4045 | -1.4009 | +0.0970 | -0.0656 |
| Mac 후속 | X01_FREQ | 1 | -0.2930 | +0.5756 | +0.0341 | +0.4571 | -0.3280 |
| Mac 후속 | X02_CONN | 1 | +0.9228 | +0.0713 | -0.0985 | -0.1421 | +0.0848 |
| Mac 후속 | X03_PIXEL | 1 | +0.0623 | +0.5542 | -0.3273 | +0.3794 | -0.5221 |
| T4 밝기 재검증 | A01_BRIGHT_HALF | 2 | -0.2321 | +0.6288 | -0.3449 | +0.2712 | -0.4119 |

## H100 gate 탐색 최종

사용자 제공18:53 보고서 기준. 전체32회 원자료를 확보한 것은 아니며, 공개된 개별6행과7개3seed집계는 중복 범위다.

| ID | 3seed D4 평균(%) | B 대비 Δ(%p) | 판단 |
|---|---|---|---|
| B | 79.8202 | +0.0000 | 최종 1위 |
| CS_CBAM | 79.8190 | -0.0012 | 사실상 동률, 향상 아님 |
| C_ALL | 79.7879 | -0.0323 | 향상 없음 |
| M_UNSHARED | 79.7760 | -0.0443 | 향상 없음 |
| C0 | 79.7485 | -0.0717 | 향상 없음 |
| C3 | 79.7224 | -0.0978 | 향상 없음 |
| S_CBAM | 79.5659 | -0.2543 | 저하 |

CS_CBAM은 평균−0.0012%p이며 사실상 동률이다. 보고된seed44 paired bootstrap95% CI [−0.4060,+0.3984]%p가0을 포함한다.

## 과거 실험에서 얻은 변화

초기 MIMU 전체 이미지를224로 축소한 Single66.0385%에서 원해상도 패치/희귀상 sampling/손실 개선을 거쳐77%대로 올라갔다. 이는 여러 조건이 함께 바뀐 경로이며 단일 요인의 효과로 해석하지 않는다.
초기 ResNet34 U-Net에 scSE를 추가한 Single 차이는+0.6741%p였다. 문맥을 매 epoch 올바르게 평가한448/224 실험은 같은 대조군 대비 D4+0.3683%p였다. 과거 검증 입력 불일치로 나온30.7344% D4는 유효한 구조 비교에서 제외한다.
팀 보고의 R014 균형 sampling/증강/150epoch 경로, R019 ResNet50, R020 Lovasz, R022 ResNet101 경로를 거쳐 R022 D4 79.8969%가 기준 recipe가 되었다. 변화당 여러 조건·사전학습 차이가 있으므로 각 보고된 비교 범위를 유지한다.
초기 DySample100은+0.3316%p였지만 R022의 최종 upsampling 교체 R029는−0.1407%p였다. 타일 Dice도seed42+0.0362%p,seed43−0.2712%p로 재현되지 않았다. 초기 긍정 결과를 현재 recipe에 자동 적용하지 않는다.
DySample200은100epoch 대비+0.3728%p였으나 학습 길이와 cosine 경로가 함께 달라졌다. Train-only SSL 후 DySample200은 같은200epoch 대조군 대비−0.3547%p였다.
다수결 앙상블의 최고 저장 D4는80.1135%였고 조합 내 최고 단일 대비+0.0507%p였다. 같은 Valid에서4조합을 탐색한 추론 평가이며 새 학습이나 독립 Test 결과가 아니다.

## 다음 실험

1. 진행 중인 U2/증강/밝기 재검증/Mac loss·LR 큐를 완료하여3seed 결론을 확정한다. 추가L4의 복원 감시 연결 오류는 별도 조치가 필요하다.
2. H100에서 새 패키지의 batch4/8/16 메모리·처리 속도를 측정한다. 배치를 늘리면 B도 같은 배치로 새 폴더에서 재측정한다.
3. 1차 B3회 + HT_BOT3회 + HT_PYR3회 총9회, 결과 확인 후 HT_DEC16 3회를 추가한다.
4. 구조가 개선되면 same-server D03 및 증강 조합을 별도 검증한다. 기존Mac 결과와 H100 값을 섞어 비교하지 않는다.
5. 구조/recipe 확정 후 독립 Test를 평가한다. 현재 결과는 Valid 기반 탐색 결과다.

서버 패키지: [ZIP](../../releases/KoMaP_UNet_Transformer_Code_20261008.zip), [README](../../experiments/transformer_hybrid_lab_20261007/README_KO.md).

## 전체 장부

아래는 전체151개 기록의 갱신 시점 표다. CSV에는 원자료 경로·근거 수준·비고를 함께 보존했다. 완료되지 않은 행의Single은 중간 best이며 최종 성능이 아니다.


### 초기 로컬

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| SMOKE_MIMU_2 | 42 | MIMU-Net 2epoch 실행 확인 | 완료 | 7.7768 | — | — | — |
| E0 | 42 | MIMU-Net 전체 이미지224 축소 | 완료 | 66.0385 | — | — | — |
| E1 | 42 | MIMU-Net 원해상도 패치·Eutectic sampling·가중 Dice | 완료 | 76.4688 | — | — | — |
| E2 | 42 | MIMU-Net CE/Dice 0.5:0.5 | 완료 | 77.2188 | — | — | — |
| E3 | 42 | ImageNet ResNet34 U-Net | 완료 | 76.9981 | — | — | — |
| E4 | 42 | ResNet34 U-Net + decoder scSE | 완료 | 77.6722 | 78.0264 | +0.6741 | E3 (Single) |
| E5_D4 | 42 | E4 동일 checkpoint의 8-view D4 TTA | 완료 | 77.6722 | 78.0264 | +0.3542 | E4 Single |

### 업샘플링

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| FULLRES_SKIP | 42 | 원해상도 detail skip | 완료 | 77.5930 | 78.0161 | -0.0102 | E4 |
| PIXELSHUFFLE | 42 | PixelShuffle 업샘플링 | 완료 | 77.8527 | 78.1997 | +0.1733 | E4 |
| DYSAMPLE100 | 42 | DySample 업샘플링 | 완료 | 77.9469 | 78.3580 | +0.3316 | E4 |

### 보조 구조·손실

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| CONTRAST100 | 42 | 상 구별 contrast 보조 손실 | 완료 | 77.7976 | 78.1764 | +0.1500 | E4 |
| SNAKE100 | 42 | Snake 파생 구조 | 완료 | 77.8391 | 78.2344 | +0.2081 | E4 |
| BOUNDARY100 | 42 | 경계 보조 감독 | 완료 | 77.7942 | 78.1802 | +0.1538 | E4 |
| CLDICE42 | 42 | Eutectic clDice 보조 손실 | 완료 | 77.8915 | 78.2436 | +0.2173 | E4 |

### 샘플링

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| CONTACT25 | 42 | 접촉 경계 중심 sampling 25% | 완료 | 77.3759 | 77.8355 | -0.1909 | E4 |

### clDice 재현

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| CL_BASE_123 | 123 | 동일 seed 대조군 | 완료 | 77.5221 | 77.9288 | — | — |
| CLDICE_123 | 123 | clDice 추가 seed 검증 | 완료 | 77.6259 | 77.9841 | +0.0553 | CL_BASE_123 |
| CL_BASE_2026 | 2026 | 동일 seed 대조군 | 완료 | 77.6305 | 77.9030 | — | — |
| CLDICE_2026 | 2026 | clDice 추가 seed 검증 | 완료 | 77.5232 | 77.7976 | -0.1055 | CL_BASE_2026 |

### 문맥·HRNet

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| CONTEXT_OLD | 42 | 448 입력/중앙224 학습;224 검증 조건 불일치 | 완료·평가조건 불일치 | 26.8328 | 30.7344 | — | — |
| HRNET_W18_100 | 42 | ImageNet HRNet-W18 + stride4 head | 완료 | 73.1409 | 73.5721 | -4.4543 | E4 |

### 문맥 보정

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| CONTEXT_BASE | 42 | 매 epoch halo 검증 대조군 | 완료 | 77.8003 | 78.1274 | — | — |
| CONTEXT448 | 42 | 매 epoch 448 context/중앙224 halo 검증 | 완료 | 78.1155 | 78.4957 | +0.3683 | CONTEXT_BASE |

### 문맥 재평가

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| HALO_baseline_best | 42 | baseline_best 동일 checkpoint halo 재평가 | 완료 | 76.6551 | 78.2953 | — | — |
| HALO_context_best224 | 42 | context_best224 동일 checkpoint halo 재평가 | 완료 | 75.4055 | 75.6602 | — | — |
| HALO_context_last100 | 42 | context_last100 동일 checkpoint halo 재평가 | 완료 | 77.9383 | 78.2781 | — | — |

### 학습 길이

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| DYSAMPLE200 | 42 | DySample 100→200epoch/cosine 경로 변경 | 완료 | 78.4471 | 78.7308 | +0.3728 | DYSAMPLE100 |

### 사전학습

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| TRAIN_SSL50 | 42 | Train70만 사용한 SSL50epoch | 완료 | — | — | — | — |
| SSL_DYSAMPLE200 | 42 | ImageNet+Train-only SSL 후 DySample200 | 완료 | 78.1412 | 78.3760 | -0.3547 | DYSAMPLE200 |

### 팀 보고 R012–R044

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| ZIP-REF | 42 | ImageNet ResNet34 U-Net+scSE, 원본 Context448 recipe | 완료(팀 보고) | — | 78.4957 | — | — |
| R012 | 42 | ZIP-REF: MatSSL ResNet50로 encoder 교체; ZIP의 100e 문맥·loss·sampling 유지 | 완료(팀 보고) | — | 77.3948 | -1.1010 | ZIP-REF |
| R013 | 42 | R012: MatSSL ResNet50 유지, gray 단일채널 정규화→RGB 반복/ImageNet 채널별 정규화 | 완료(팀 보고) | — | 77.3268 | -0.0679 | R012 |
| R014 | 42 | ZIP-REF: ResNet34 유지; Al3Ni/Eutectic 균형 sampling + 약한 광도 증강 + 100→150e | 완료(팀 보고) | — | 79.3096 | +0.8139 | ZIP-REF |
| R015 | 42 | R014 + boundary; 동일-epoch 지시로 중단 | 중단 | — | — | — | R022 |
| R016 | 42 | R014: train-only 4클래스 boundary head/보조 loss, 가중치 0.1 | 완료(팀 보고) | — | 79.2849 | -0.0248 | R014 |
| R017 | 42 | R014: Eutectic Si clDice 보조 loss, 가중치 0.05 | 완료(팀 보고) | — | 79.1293 | -0.1803 | R014 |
| R018 | 42 | R014: full-resolution grayscale detail skip 추가 | 완료(팀 보고) | — | 79.3096 | +0.0000 | R014 |
| R019 | 42 | R014: ImageNet V2 ResNet50 + RGB 반복/ImageNet 정규화 | 완료(팀 보고) | — | 79.4624 | +0.1528 | R014 |
| R020 | 42 | R019: Dice 0.5 중 절반을 per-image weighted Lovasz로 교체 | 완료(팀 보고) | — | 79.6937 | +0.2312 | R019 |
| R021 | 42 | R020: Al3Ni/Eutectic-rich tile 최소 비율 10%→3% | 완료(팀 보고) | — | 79.4084 | -0.2852 | R020 |
| R022 | 42 | R020: encoder ImageNet V2 ResNet50→ResNet101 | 완료(팀 보고) | — | 79.8969 | +0.2033 | R020 |
| R023 | 42 | R022: encoder ResNet101→ImageNet V2 ResNet152 | 완료(팀 보고) | — | 79.6613 | -0.2356 | R022 |
| R024 | 42 | R022: ImageNet V1 ConvNeXt-Tiny + grayscale detail skips; loss/sampler 유지 | 완료(팀 보고) | — | 79.7843 | -0.1126 | R022 |
| R025 | 42 | R024: ConvNeXt-Tiny→ConvNeXt-Small | 완료(팀 보고) | — | 79.2360 | -0.5483 | R024 |
| R026 | 42 | R022: encoder LR 3e-4→1e-4; decoder LR 3e-4 유지 | 완료(팀 보고) | — | 78.8724 | -1.0245 | R022 |
| R027 | 42 | R022: Eutectic Si의 Dice/Lovasz 가중치 2→1; CE는 그대로 | 완료(팀 보고) | — | 79.6039 | -0.2931 | R022 |
| R028 | 42 | R022: 밝기/대비·감마·노이즈 광도 증강만 제거; fresh attempt2 | 완료(팀 보고) | — | 79.2796 | -0.6174 | R022 |
| R029 | 42 | R022: 마지막 bilinear 업샘플링만 DySample LP x2/groups4로 교체 | 완료(팀 보고) | — | 79.7563 | -0.1407 | R022 |
| R030 | 42 | R022: 감독 영역 192/224/256 균등 배율 jitter; 문맥은 2배 후448/224로 resize; 추론448 고정 | 완료(팀 보고) | — | 79.1657 | -0.7312 | R022 |
| R031 | 42 | R022: Dice batch-pooling → 타일별 spatial(2,3) 계산 후 평균; CE/Lovasz 유지 | 완료(팀 보고) | — | 79.9332 | +0.0362 | R022 |
| R032 | 43 | R022 기본 recipe의 seed43 fresh 재검증 (seed42와 다른 초기화/RNG) | 완료(팀 보고) | — | 79.7526 | -0.1444 | R022 (다른seed 참고) |
| R033 | 43 | R031 타일별 Dice의 seed43 fresh 재검증; same-seed R032와 대조 | 완료(팀 보고) | — | 79.4814 | -0.2712 | R032 |
| R034 | 42 | R022: decoder BN10개 → GroupNorm8; encoder BN 유지 | 완료(팀 보고) | — | 79.2469 | -0.6500 | R022 |
| R035 | 42 | R022: encoder BN104개 pretrained 통계 고정, affine 학습; decoder BN 유지 | 완료(팀 보고) | — | 79.7864 | -0.1105 | R022 |
| R036 | 42 | R022: CE만 타일별 손실 상위25% 평균 (k=max(1,floor(HW/4)), threshold 없음) | 완료(팀 보고) | — | 79.6440 | -0.2529 | R022 |
| R037 | 42 | R022: native crop 원점 dy/dx ±56px; clamp, targeted 희귀상10% 미달이면 원래 위치 fallback | 완료(팀 보고) | — | 79.6228 | -0.2741 | R022 |
| R038 | 42 | R022: RGB HRNet-W18 ms_in1k + stride4 merge/head; architecture+pretraining 시스템 대조 | 완료(팀 보고) | — | 74.9830 | -4.9139 | R022 |
| R039 | 42 | R022: ResNet101 layer4 stride→dilation, OS32→16; 기존 decoder 유지 | 완료(팀 보고) | — | 79.8071 | -0.0899 | R022 |
| R040 | 42 | R022: 감독224 안112-square image-mask CutMix, p=.5/batch, nonself donor; 별도 mix RNG | 완료(팀 보고) | — | 79.4035 | -0.4934 | R022 |
| R041 | 42 | R022: CE만 GT radius2 경계2/내부1 가중, 타일별 sumweight 정규화 | 완료(팀 보고) | — | 79.7239 | -0.1731 | R022 |
| R042 | 42 | R022: ImageNet V2 ResNeXt101-32x8d encoder; 동일 skip폭/decoder, architecture+pretraining 대조 | 완료(팀 보고) | — | 79.4934 | -0.4036 | R022 |
| R043 | 42 | R022: 기본 fresh 모델 생성 후 decoder0–4 scSE→Identity; 나머지 초기 state/RNG 유지 | 완료(팀 보고) | — | 79.5573 | -0.3396 | R022 |
| R044 | 42 | R022: CE.5+Dice.25+Lovasz.25 → CE.5+Dice0+Lovasz.5; classweight/present-class 정책 유지 | 완료(팀 보고) | — | 79.6379 | -0.2591 | R022 |

### 팀 D계열 서버

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| B | 42 | 기준 | 완료(사용자 보고) | 79.3590 | 79.7248 | — | — |
| D01 | 42 | 상 구별 대조학습 | 완료(사용자 보고) | 79.2446 | 79.7461 | +0.0213 | B |
| D02 | 42 | 파생 BEM | 완료(사용자 보고) | 79.1959 | 79.6669 | -0.0579 | B |
| D03 | 42 | Al3Ni–Eutectic 접촉부 CE 가중 | 완료(사용자 보고) | 79.4020 | 79.8005 | +0.0758 | B |

### 팀 D03 추가 seed 서버

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| B | 43 | 같은 seed B/D03 fresh 검증 | 완료(사용자 보고) | 79.4181 | 79.7892 | — | — |
| D03 | 43 | 같은 seed B/D03 fresh 검증 | 완료(사용자 보고) | 79.3674 | 79.6844 | -0.1049 | B |
| B | 44 | 같은 seed B/D03 fresh 검증 | 완료(사용자 보고) | 79.1824 | 79.5527 | — | — |
| D03 | 44 | 같은 seed B/D03 fresh 검증 | 완료(사용자 보고) | 78.9994 | 79.3878 | -0.1649 | B |

### RRCU Mac seed42

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| B | 42 | RRCU와 같은 코드의 대조군 | 완료 | 79.3428 | 79.6303 | — | — |
| R01 | 42 | Shallow decoder block3 RRCU(t=2) followed by original scSE | 완료 | 78.1492 | 78.8351 | -0.7953 | B |
| R02 | 42 | RRCU(t=2) inside block3 spatial attention logit branch | 완료 | 79.0070 | 79.4618 | -0.1685 | B |
| R03 | 42 | Pooled recurrent residual 1x1 unit inside block3 channel attention logit branch | 완료 | 79.5920 | 80.0627 | +0.4324 | B |

### 초기 RRCU

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| C13 | 42 | decoder RRCU 초기 시도 | 사용자 중단 | 74.6342 | — | — | — |

### L4 아키텍처

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| B | 42 | Maintained ResNet101 U-Net/scSE, R022-aligned recipe | 완료 | 79.4433 | 79.7847 | — | — |
| B_SCRATCH | 42 | R101 U-Net/scSE baseline, fresh scratch initialization | 완료 | 77.7725 | 78.1452 | -1.6395 | B |
| U2_PLAIN_DEEP | 42 | U²-Net without attention; six-side deep supervision | 완료 | 79.3610 | 79.5025 | -0.2822 | B |
| U2_SCSE_ALL_DEEP | 42 | scSE after all five decoder RSUs; six-side deep supervision | 완료 | 79.2144 | 79.3551 | -0.4296 | B |
| U2_SCSE_LAST2_DEEP | 42 | scSE after stage2d/stage1d only; six-side deep supervision | 완료 | 79.3214 | 79.5746 | -0.2100 | B |
| U2_ECA_DEEP | 42 | ECA on final residual branch inside all 11 outer RSUs; six-side deep supervision | 완료 | 79.2978 | 79.5077 | -0.2770 | B |
| U2_CBAM_DEEP | 42 | CBAM on six side-head feature maps before prediction/fusion; six-side deep supervision | 완료 | 79.6816 | 79.8406 | +0.0559 | B |
| U2_ECA_CBAM_DEEP | 42 | RSU residual-branch ECA + six pre-fusion CBAM gates; six-side deep supervision | 완료 | 79.3345 | 79.5010 | -0.2836 | B |
| U2_ECA_SCSE_DEEP | 42 | RSU residual-branch ECA + five decoder scSE gates; six-side deep supervision | 완료 | 79.1356 | 79.3157 | -0.4690 | B |
| U2_PLAIN | 42 | U²-Net without attention; fused loss only | 완료 | 79.2321 | 79.4986 | -0.2861 | B |
| U2_SCSE_ALL | 42 | scSE after all five decoder RSUs; fused loss only | 완료 | 79.2094 | 79.3393 | -0.4454 | B |
| U2_SCSE_LAST2 | 42 | scSE after stage2d/stage1d only; fused loss only | 완료 | 79.5087 | 79.7976 | +0.0129 | B |
| U2_ECA | 42 | ECA on final residual branch inside all 11 outer RSUs; fused loss only | 학습 중 | 80.0073 | — | — | B |
| B | 44 | Maintained ResNet101 U-Net/scSE, R022-aligned recipe | 완료 | 79.1628 | 79.6042 | — | — |
| B_SCRATCH | 44 | R101 U-Net/scSE baseline, fresh scratch initialization | 완료 | 78.2901 | 78.6525 | -0.9517 | B |
| U2_PLAIN_DEEP | 44 | U²-Net without attention; six-side deep supervision | 완료 | 79.3072 | 79.6249 | +0.0207 | B |
| U2_SCSE_ALL_DEEP | 44 | scSE after all five decoder RSUs; six-side deep supervision | 완료 | 79.5322 | 79.7918 | +0.1876 | B |
| U2_SCSE_LAST2_DEEP | 44 | scSE after stage2d/stage1d only; six-side deep supervision | 완료 | 78.7820 | 79.0737 | -0.5305 | B |
| U2_ECA_DEEP | 44 | ECA on final residual branch inside all 11 outer RSUs; six-side deep supervision | 완료 | 79.4248 | 79.6567 | +0.0525 | B |
| U2_CBAM_DEEP | 44 | CBAM on six side-head feature maps before prediction/fusion; six-side deep supervision | 완료 | 79.4934 | 79.7663 | +0.1621 | B |
| U2_ECA_CBAM_DEEP | 44 | RSU residual-branch ECA + six pre-fusion CBAM gates; six-side deep supervision | 완료 | 79.7860 | 80.0230 | +0.4188 | B |
| U2_ECA_SCSE_DEEP | 44 | RSU residual-branch ECA + five decoder scSE gates; six-side deep supervision | 완료 | 79.8241 | 80.0012 | +0.3970 | B |
| U2_PLAIN | 44 | U²-Net without attention; fused loss only | 실행 확인 불가(복원 연결 오류) | 77.5618 | — | — | B |

### Mac 후속 검증

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| B | 43 | Maintained ResNet101 U-Net/scSE, R022-aligned recipe | 완료 | 79.2133 | 79.3859 | — | — |
| D03 | 43 | B + direct Al3Ni/Eutectic contact weighted CE (radius2, multiplier2) | 완료 | 79.2251 | 79.7510 | +0.3651 | B |
| B | 44 | Maintained ResNet101 U-Net/scSE, R022-aligned recipe | 완료 | 79.3151 | 79.6674 | — | — |
| D03 | 44 | B + direct Al3Ni/Eutectic contact weighted CE (radius2, multiplier2) | 완료 | 79.5725 | 79.8798 | +0.2124 | B |
| B | 42 | Maintained ResNet101 U-Net/scSE, R022-aligned recipe | 완료 | 79.3428 | 79.6303 | — | — |
| D03 | 42 | B + direct Al3Ni/Eutectic contact weighted CE (radius2, multiplier2) | 완료 | 79.3360 | 79.7855 | +0.1552 | B |
| X01_FREQ | 42 | Reduced no-offset frequency fusion at OS4->OS2 skip | 완료 | 79.0158 | 79.4297 | -0.2006 | B |
| X02_CONN | 42 | Eight-direction four-class connectivity auxiliary supervision | 완료 | 79.4990 | 80.0115 | +0.3811 | B |
| X03_PIXEL | 42 | Final PixelShuffle upsampling | 완료 | 79.3033 | 79.5947 | -0.0356 | B |
| L01_DICE35 | 42 | CE.5 Dice.35 Lovasz.15 | 학습 중 | 77.6988 | — | — | B |

### R03 증강

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| A00_BASE | 42 | Current R03 augmentation control | 완료 | 79.0390 | 79.5118 | — | — |
| A01_BRIGHT_HALF | 42 | Brightness amplitude .08 to .04 | 완료 | 79.4745 | 79.8585 | +0.3467 | A00_BASE |
| A02_BRIGHT_OFF | 42 | Disable brightness; retain contrast | 학습 중 | 78.9559 | — | — | A00_BASE |
| A03_CONTRAST_OFF | 42 | Disable contrast; retain brightness | 완료 | 79.3327 | 79.8608 | +0.3490 | A00_BASE |
| A05_GAMMA_HALF | 42 | Gamma range .9-1.1 to .95-1.05 | 완료 | 79.3686 | 79.7883 | +0.2764 | A00_BASE |
| A07_NOISE_HALF | 42 | Noise sigma .01 to .005 | 완료 | 79.1625 | 79.5628 | +0.0510 | A00_BASE |
| A09_BLUR_WEAK | 42 | Add weak Gaussian blur sigma .3-.6 p=.15 | 완료 | 78.8644 | 79.5464 | +0.0346 | A00_BASE |
| A00_BASE | 43 | Current R03 augmentation control | 완료 | 79.3059 | 79.5750 | — | — |
| A01_BRIGHT_HALF | 43 | Brightness amplitude .08 to .04 | 완료 | 79.2023 | 79.5483 | -0.0267 | A00_BASE |
| A03_CONTRAST_OFF | 43 | Disable contrast; retain brightness | 완료 | 79.2576 | 79.7147 | +0.1397 | A00_BASE |
| A05_GAMMA_HALF | 43 | Gamma range .9-1.1 to .95-1.05 | 완료 | 79.1995 | 79.5623 | -0.0127 | A00_BASE |
| A07_NOISE_HALF | 43 | Noise sigma .01 to .005 | 완료 | 79.4943 | 79.8813 | +0.3063 | A00_BASE |
| A09_BLUR_WEAK | 43 | Add weak Gaussian blur sigma .3-.6 p=.15 | 완료 | 79.1582 | 79.4673 | -0.1077 | A00_BASE |
| A00_BASE | 44 | Current R03 augmentation control | 완료 | 79.3273 | 79.7572 | — | — |
| A01_BRIGHT_HALF | 44 | Brightness amplitude .08 to .04 | 완료 | 79.3332 | 79.6912 | -0.0660 | A00_BASE |
| A03_CONTRAST_OFF | 44 | Disable contrast; retain brightness | 완료 | 79.3818 | 79.7806 | +0.0234 | A00_BASE |
| A05_GAMMA_HALF | 44 | Gamma range .9-1.1 to .95-1.05 | 완료 | 79.2623 | 79.6223 | -0.1349 | A00_BASE |
| A07_NOISE_HALF | 44 | Noise sigma .01 to .005 | 완료 | 79.4687 | 79.8242 | +0.0670 | A00_BASE |
| A09_BLUR_WEAK | 44 | Add weak Gaussian blur sigma .3-.6 p=.15 | 완료 | 78.8346 | 79.3444 | -0.4128 | A00_BASE |

### 앙상블 진단

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| B42+B44+U2 | 42/44 조합 | 저장된 D4 마스크 3모델 다수결; 동률은 첫 멤버 | 완료 | — | 79.9728 | +0.1881 | 조합 내 최고 단일 |
| B42+B44+U2_scSE | 42/44 조합 | 저장된 D4 마스크 3모델 다수결; 동률은 첫 멤버 | 완료 | — | 79.9444 | +0.1597 | 조합 내 최고 단일 |
| R03+B42+B44 | 42/44 조합 | 저장된 D4 마스크 3모델 다수결; 동률은 첫 멤버 | 완료 | — | 80.0165 | -0.0462 | 조합 내 최고 단일 |
| R03+B42+U2 | 42/44 조합 | 저장된 D4 마스크 3모델 다수결; 동률은 첫 멤버 | 완료 | — | 80.1135 | +0.0507 | 조합 내 최고 단일 |

### T4 R03 밝기 재검증

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| A00_BASE | 42 | Current R03 augmentation control | 완료 | 79.4800 | 79.7824 | — | — |
| A01_BRIGHT_HALF | 42 | Brightness amplitude .08 to .04 | 완료 | 79.1564 | 79.5618 | -0.2206 | A00_BASE |
| A00_BASE | 43 | Current R03 augmentation control | 완료 | 79.5820 | 79.9943 | — | — |
| A01_BRIGHT_HALF | 43 | Brightness amplitude .08 to .04 | 완료 | 79.3754 | 79.7677 | -0.2266 | A00_BASE |
| A00_BASE | 44 | Current R03 augmentation control | 완료 | 79.3454 | 79.9169 | — | — |
| A01_BRIGHT_HALF | 44 | Brightness amplitude .08 to .04 | 학습 중 | 77.9704 | — | — | A00_BASE |

### H100 층별·공간 최종

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| B | 42 | 기존 scSE 기준 | 완료(최종 보고) | — | 80.0013 | — | — |
| CS_CBAM | 42 | channel+spatial CBAM 파생 gate | 완료(최종 보고) | — | 79.8545 | -0.1467 | B |
| B | 43 | 기존 scSE 기준 | 완료(최종 보고) | — | 79.6690 | — | — |
| CS_CBAM | 43 | channel+spatial CBAM 파생 gate | 완료(최종 보고) | — | 79.8186 | +0.1496 | B |
| B | 44 | 기존 scSE 기준 | 완료(최종 보고) | — | 79.7905 | — | — |
| CS_CBAM | 44 | channel+spatial CBAM 파생 gate | 완료(최종 보고) | — | 79.7840 | -0.0064 | B |

### H100 3seed 최종 집계

| ID | seed | 변경 내용 | 상태 | Single(%) | D4(%) | 부모 Δ(%p) | 비교 부모 |
|---|---|---|---|---|---|---|---|
| B | 42/43/44 | 최종 1위 | 완료(집계 보고) | — | 79.8202 | +0.0000 | — |
| CS_CBAM | 42/43/44 | 사실상 동률, 향상 아님 | 완료(집계 보고) | — | 79.8190 | -0.0012 | B |
| C_ALL | 42/43/44 | 향상 없음 | 완료(집계 보고) | — | 79.7879 | -0.0323 | B |
| M_UNSHARED | 42/43/44 | 향상 없음 | 완료(집계 보고) | — | 79.7760 | -0.0443 | B |
| C0 | 42/43/44 | 향상 없음 | 완료(집계 보고) | — | 79.7485 | -0.0717 | B |
| C3 | 42/43/44 | 향상 없음 | 완료(집계 보고) | — | 79.7224 | -0.0978 | B |
| S_CBAM | 42/43/44 | 저하 | 완료(집계 보고) | — | 79.5659 | -0.2543 | B |

미착수 등록45행은 [PENDING.csv](PENDING.csv)에 있다. 이외 학습 중·연결 확인 불가 작업도 남은 큐에 포함된다.
