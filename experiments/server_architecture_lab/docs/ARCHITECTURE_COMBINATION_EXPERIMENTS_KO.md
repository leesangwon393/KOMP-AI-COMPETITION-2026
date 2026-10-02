# R022 기준 아키텍처 조합 실험 순서

모든 후속 학습은 팀 README의 R022 Context448 recipe에 맞춘다. 팀 R022는 ResNet101/ImageNet V2, bilinear+scSE U-Net, Context448·중앙224 감독, 150 epoch, `0.5 CE + 0.25 weighted Dice + 0.25 per-image present-class weighted Lovasz`, rare class1/2 균형 sampling, 광도 증강이다. best 선택은 single-view halo Valid, 최종 D4는 선택 후 1회다.

팀 기록 R022는 D4 absent=1 `0.7989694265`; R029는 마지막 bilinear를 DySample LP로 교체해 `0.797563`, 즉 `-0.1407%p`였다. 본 패키지는 R022 비교에서 bilinear를 유지한다. 2026-10-01의 이 실험 순서는 제안이며 학습을 실행하지 않았다.

## 1차: 경계와 문맥 모듈

| 설정 | 변경 | 판정 |
|---|---|---|
| B | R022 recipe, fresh ImageNet V2 encoder + 새 decoder | ZIP의 B 구현 기준. 원격 R022 checkpoint와 bitwise 재현으로 간주하지 않음 |
| C01 | stem/layer1 skip에 high-pass residual `E−mean3×3(E)`를 투영해 추가 | B와 비교. R018의 raw detail skip은 별도 recipe에서 개선이 없었으므로 이 변형의 효과를 검증 |
| C02 | layer3 Q, layer4 K/V cross-attention | B 및 `CTRL_cross_concat`과 비교 |
| C03 | C01+C02 | 네 설정 B/C01/C02/C03을 함께 비교해 조합 효과 확인 |

먼저 B/C01/C02를 완료한다. 단일 모듈 중 하나가 관심 클래스 IoU 또는 paired image 결과에 개선을 보이면 C03으로 간다. C01은 MAUNet의 frequency-aware skip 아이디어에서 착안했으나 논문 전체 구현이 아니다. [MAUNet 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC7796035/)

## 2차: 문맥 크기 선택과 skip 융합

- C04: 한 decoder stage에 dilation 1/2/3 adaptive fusion. `CTRL_dilation_fixed`를 함께 둬 입력별 선택 효과와 branch 추가 효과를 구별한다. CMAA 원리를 단순화한 변형이며 APAM/LSTM 전체 구현이 아니다. [CMAA 원문](https://www.sciencedirect.com/science/article/abs/pii/S0957417425005470/)
- C05: C04+HF. C04와 C01 결과가 긍정적일 때만 실행한다.
- C06: shallow/deep skip weighted fusion과 별도 HF 경로. 먼저 `CTRL_weighted_only`를 평가해 learned skip weighting 효과를 분리한다.

## 3차: decoder 교체

- C07: PPM+FPN UPerNet-derived decoder, bilinear 유지.
- C08: nested dense skip U-Net++-derived decoder, bilinear 유지. Context448과 4개 batch에서 GPU activation memory를 먼저 확인한다.

decoder만 교체하고 encoder·loss·sampler·augmentation은 고정한다. C07/C08은 원 논문 전체 모델이나 deep-supervision 재현을 주장하지 않는다.

## 4차: encoder 사전학습·큰 hybrid

- C11: MicroNet v1.0 ResNet101 encoder 초기화만 R022 ImageNet V2 초기화와 비교한다. 이후에만 HF 추가를 검토한다. 원본 RGB preprocessing 재현이 아니라 공통 R022 입력 조건에 놓은 초기화 비교다.
- C12: 4-stage Train-only SSL. 기존 ResNet34 SSL와 MatSSL 전이가 각자 기준을 못 넘겼으므로 후순위다. Valid/Test는 SSL에 사용하지 않는다.
- C09: 병렬 ResNet101/Swin-T. `CTRL_hybrid_concat`을 포함하고 448 입력 메모리를 먼저 확인한다. [CS-UNet 원문](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/html/Alrfou_CS-UNet_A_Flexible_Segmentation_Algorithm_for_Microscopy_Images_CVPRW_2024_paper.html)
- C10: bottleneck 네 방향 Mamba-derived block. `CTRL_mamba_only`; 서버 CUDA selective-scan 환경에서만 진행한다.

## 모든 run에 고정할 비교 계약

각 후보는 fresh pretrained encoder와 새 decoder/head에서 seed42, FP32, batch4, 420 samples/epoch, 150 epoch로 시작한다. trainer/evaluator/split/class order/R022 loss와 광도·기하 증강은 같은 코드다. R022 학습 checkpoint 이어달리기는 사용하지 않는다. 구조에 필요한 optional dependency 외 recipe 변경을 섞지 않는다.

각 epoch single-view halo Valid mIoU(absent=1)로 checkpoint를 선택한다. 후보 점수는 선택된 best의 single과 D4를 모두 보고한다. 이미지별 paired delta·4클래스 IoU·혼동행렬을 함께 검토한다. 같은 20장으로 repeated selection을 하므로 단일 seed의 작은 변화는 일반화 증거가 아니다. Test에는 mask가 없으며 후보 선택에 쓰지 않는다.
