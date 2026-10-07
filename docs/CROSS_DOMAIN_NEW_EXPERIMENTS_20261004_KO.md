# 다른 도메인에서 가져올 새로운 실험 후보

조사일: 2026-10-04 KST. 논문 원문/초록, 학회·저널 정보와 저자 공개 코드를 확인했다. **조사와 실험 제안만 작성했으며 학습·실험 큐·구현 코드는 추가하지 않았다.** 기존 RRCU 실행을 변경하지 않았다.

우선순위는 우리 과제의 Al3Ni–Eutectic 혼동, 가는 Eutectic 누락, Train70 규모와 기존 B에 대한 이식 가능성으로 판단했다. 다른 데이터셋의 논문 점수를 KoMaP 예상 점수로 사용하지 않는다. 아래 적용법·비용·성공 가능성은 실험 전 가설이다. X 번호는 이 문서의 제안 ID다.

## 기존 기록과의 관계

[완료 결과](EXPERIMENT_RESULTS_KO.md)에서 boundary auxiliary head, clDice, detail skip, DySample, HRNet, OHEM, CutMix, 전체 경계 CE 가중 등을 확인했다. D01 대조학습·D02 BEM·D03 접촉부 CE는 [서버 보고](PHASE_DISCRIMINATION_SERVER_RESULTS_20261003_KO.md)에 있다. 현재 RRCU R01–R03은 별도 진행 계열이며 U²-Net/RSU는 서버 실행용 패키지를 준비한 상태다.

아래 9편의 이름/구체적 모듈은 현재 실험 기록에서 확인되지 않았다. 다만 주파수 skip과 범용 Mamba는 [과거 C계열 제안](ARCHITECTURE_COMBINATION_EXPERIMENTS_KO.md)에도 있다. X01/X06/X07을 완전히 처음 등장한 큰 방향이라고 표현하지 않는다. 과거 C 문서의 B/DySample 설명은 당시 제안 기준이며, 이번 후보의 실제 대조군은 유지 중인 R022/B recipe다.

## 후보 한눈에 보기

| 순위 / ID | 논문 | 원 도메인 | 기여의 핵심 | 우리 적용 위치 | 부담 추정 |
|---|---|---|---|---|---|
| 1 / X01 | FreqFusion, TPAMI2024 | 자연영상 dense prediction | 적응 저·고주파 필터와 재샘플링을 함께 사용하는 특징 융합 | 얕은 decoder skip 1곳 | 중간 |
| 2 / X02 | DconnNet, CVPR2023 | 의료영상 / 혈관·OCT·피부 병변 | 방향 정보를 분리하는 connectivity 모델링 | 연결 보조 head부터, 이후 방향 module | 낮음~중간 → 중간 |
| 3 / X03 | PointRend, CVPR2020 | 자연영상 semantic/instance segmentation | 불확실한 위치를 골라 특징으로 재분류 | 최종 coarse 출력 뒤 point head | 중간 |
| 4 / X04 | OCR, ECCV2020 | 장면 분할 | 상별 영역 표현을 모아 픽셀에 문맥 전달 | decoder 중간의 region-context module | 중간 |
| 5 / X05 | RMI loss, NeurIPS2019 | VOC / CamVid | 주변 픽셀 패턴의 고차 관계를 loss로 학습 | loss만 추가 | 학습 중간 / 추론 추가 없음 |
| 6 / X06 | WTConv, ECCV2024 | 분류·dense prediction | 다단계 wavelet convolution으로 큰 수용영역 | decoder residual branch 1곳 | 낮음~중간 |
| 7 / X07 | SCSegamba, CVPR2025 | 구조물 crack segmentation | 구조를 고려하는 Mamba scan + gated bottleneck | SAVSS 부분 이식 또는 full 모델 대조 | 중간~높음 |
| 8 / X08 | Dino U-Net, arXiv2025 | 의료영상 | frozen DINOv3 + detail adapter + fidelity-aware projection | 별도 encoder/decoder 대안 | 중간~높음 / 가중치·환경 필요 |
| 9 / X09 | SegRefiner, NeurIPS2023 | semantic/instance/dichotomous segmentation | discrete diffusion으로 기존 mask를 점진 정제 | 별도 mask refiner | 높음 |

## X01. FreqFusion — 특징을 합칠 때 경계와 내부를 다르게 처리

**확인한 기여:** Adaptive Low-Pass Filter(ALPF)는 내부 특징의 불필요한 고주파 변동을 줄이고, Adaptive High-Pass Filter(AHPF)는 경계 세부를 보강한다. Offset generator는 특징의 불일치와 얇은 경계를 재샘플링으로 정제한다. 논문은 TPAMI2024이며 ECCV 논문으로 표기하지 않는다. [논문](https://arxiv.org/abs/2408.12879), [저자 코드](https://github.com/Linwei-Chen/FreqFusion).

**우리 제안:** ResNet101/ImageNet encoder와 기존 loss/scSE를 유지하고, 얕은 decoder의 skip 융합 한 곳에 ALPF/AHPF를 적용한다. 먼저 offset 없이 주파수 융합을 비교하고, 유망하면 offset을 켜 별도 비교한다. 단순 고주파 residual skip C01보다 필터가 입력에 따라 달라지며 내부와 경계를 함께 조절한다. 최종 DySample 단독 R029와도 비교 위치·구성이 다르다.

**기대와 한계:** 가는 조직 손실과 같은 상 내부의 흔들림을 함께 겨냥할 수 있다. 촬영 noise를 경계로 강화하거나 저주파 smoothing이 얇은 조직을 지울 수도 있다. 공식 단일 파일에 MMCV CARAFE 및 PyTorch fallback이 있지만 fallback의 unfold 메모리·출력 동등성을 확인해야 한다. 공개 코드가 있다는 이유로 서버에서 바로 검증된 상태라고 주장하지 않는다. [핵심 구현](https://raw.githubusercontent.com/Linwei-Chen/FreqFusion/main/FreqFusion.py).

## X02. DconnNet — 픽셀의 이름과 이웃 연결을 함께 생각

**확인한 기여:** 단순 픽셀 분류에서 나아가 방향별 connectivity를 모델링하고 latent 방향 정보를 분리·전달한다. 저자 코드에는 8방향×클래스 connectivity target, bilateral voting, multiclass 경로가 있다. [논문](https://arxiv.org/abs/2304.00145), [저자 코드](https://github.com/Zyun-Y/DconnNet), [연결 target/loss](https://raw.githubusercontent.com/Zyun-Y/DconnNet/main/connect_loss.py).

**우리 제안:** 첫 비교 X02a는 B의 4클래스 출력을 유지하면서 GT에서 만든 8방향 같은-상 연결을 보조 head로 감독한다. 이는 **connectivity 기반 단순화 실험이며 DconnNet 전체 재현이 아니다.** X02a가 유망하면 X02b에서 방향을 분리하는 module과 voting을 추가해 원 논문의 구조적 기여를 비교한다. clDice R017은 skeleton 기반 loss이고, 이 후보는 방향별 픽셀 관계의 명시적 예측이다.

**기대와 한계:** Eutectic의 짧은 단절·끊김과 인접한 다른 상의 오연결을 겨냥한다. 같은 상의 모든 영역을 하나로 연결시키는 제약은 주지 않는다. 실제로 분리된 조직을 붙이거나 1픽셀 구조를 소실시키지 않는지 확인한다. 중앙224 밖 이웃은 loss에서 제외하고 방향을 정확히 정의한다. 연결 출력을 D4에서 직접 평균한다면 방향 채널 permutation도 필요하다. Voting 후 4클래스로 변환하는 경로는 별도 확인한다.

원 코드 일부 loss는 class0을 배경처럼 제외한다. 우리 class0은 Primary Si이므로 그대로 복사하면 안 된다. `.cuda()` 하드코딩, shifting과 border 처리를 현재 device/4상 convention에 맞춰야 한다.

## X03. PointRend — 애매한 픽셀을 직접 다시 분류

**확인한 기여:** coarse mask에서 적응적으로 위치를 선택하고 fine feature/coarse prediction을 사용해 point-wise prediction을 수행한다. Semantic과 instance segmentation에 모두 적용하며 추론의 반복 subdivision으로 세부를 복원한다. [논문](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html), [논문 PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf), [공식 구현](https://github.com/facebookresearch/detectron2/tree/main/projects/PointRend).

**우리 제안:** 기존 B coarse logits 뒤에 point head를 붙인다. 픽셀의 예측 불확실도와 균등 위치를 섞어 학습하고, 추론에서도 예측값만으로 위치를 고른다. 얕은 feature/coarse logits와 원본 해상도 detail 경로의 활용을 명시한다. B는 이미 OS2 decoder 출력 후 upsample을 사용하므로 어느 해상도에서 실제 새 세부 정보가 들어가는지가 중요하다.

**차이와 한계:** OHEM R036은 큰 loss 픽셀의 학습 가중을 바꿨다. PointRend는 별도 예측 head가 특정 픽셀의 최종 값을 바꾼다. Boundary head R016도 경계 감독만 추가했으므로 다르다. 확신 있게 틀린 픽셀은 uncertainty 선택만으로 놓칠 수 있다. Bilinear 업샘플링된 feature만 재사용하면 개선이 약할 수 있다. 점 선택과 추론 반복의 실제 비용을 기록한다.

## X04. OCR — 상별 영역 문맥으로 픽셀 표현을 보강

**확인한 기여:** coarse segmentation으로 상별 영역 표현을 모으고, 각 픽셀과 영역 표현 사이 관계를 계산해 object-context를 전달한다. 제목의 OCR은 문자인식이 아니라 Object-Contextual Representations다. [ECCV2020 논문 PDF](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510171.pdf), [arXiv](https://arxiv.org/abs/1909.11065), [공식 코드](https://github.com/openseg-group/openseg.pytorch).

**우리 제안:** B encoder를 유지하고 decoder 중간 특징에서 coarse4상 head → 상별 특징 집계 → 픽셀별 문맥 융합을 붙인다. 기존 HRNet R038 전체 교체와 별개다. D01은 contrastive loss로 표현을 학습했지만, OCR은 추론 때도 region-context가 픽셀 feature를 보강한다. 단순 cross-scale attention C02와 달리 관계의 기준이 상별 영역이다.

**기대와 한계:** 국소 밝기·텍스처가 비슷한 Al3Ni/Eutectic을 주변 상의 문맥으로 구별할 가설이다. Coarse 오분류가 영역 표현을 오염시키거나, 큰 영역 표현이 가는 희귀 상을 압도할 수 있다. 448 context 안의 문맥이며 영상 전체 문맥이라고 부르지 않는다.

## X05. RMI loss — loss에서 지역 패턴을 비교

**확인한 기여:** 하나의 픽셀과 주변 이웃을 다차원 표현으로 묶고, prediction/GT 사이 mutual information의 하한을 최적화해 지역 관계의 일관성을 학습한다. Model branch를 늘리지 않고 추론 추가 비용이 없다. [NeurIPS2019 논문](https://proceedings.neurips.cc/paper/2019/hash/a67c8c9a961b4182688768dd9ba015fe-Abstract.html), [논문 PDF](https://proceedings.neurips.cc/paper/2019/file/a67c8c9a961b4182688768dd9ba015fe-Paper.pdf), [저자 코드](https://github.com/ZJULearning/RMI).

**우리 제안:** B loss에 작은 RMI 항을 추가한다. 중앙 감독 영역 안의 local patch에서 4상 확률과 GT 관계를 비교한다. 먼저 기존 CE/Dice/Lovasz 비율과 접촉부 가중은 그대로 둔다. D03은 후속 별도 결합이다.

**차이와 한계:** clDice의 skeleton만을 목표로 하지 않고 일반 지역 패턴 관계를 학습한다. Batch별 희귀 상 부족으로 covariance가 불안정해질 수 있어 regularization/log-determinant와 gradient 규모를 확인한다. 공식 원 학습 환경은 오래됐으므로 loss 부분을 현재 PyTorch에 맞춰 검증해야 한다. 총 loss가 기존보다 작아졌는지는 성공 판단 기준이 아니다.

## X06. WTConv — 깊이를 늘리는 대신 wavelet에서 수용영역 확대

**확인한 기여:** 다단계 wavelet 분해 → 대역별 작은 convolution → 역변환으로 큰 수용영역을 얻는다. WTConv는 depth-wise convolution 대체로 설계되었고, shape bias가 커지는 특성이 보고된다. [ECCV2024 논문](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07137.pdf), [공식 코드](https://github.com/BGU-CS-VIL/WTConv).

**우리 제안:** ImageNet encoder를 건드리지 않고 decoder 한 위치에 WTConv residual branch를 비교한다. 일반 depth-wise branch와도 비교해 wavelet과 용량 변화의 효과를 분리한다. B의 일반3×3 convolution을 WTConv 하나로 그대로 치환하면 channel mixing까지 사라지므로 별도1×1 mixing 또는 원 경로를 유지한다. RRCU의 반복 공유 convolution과 메커니즘이 다르다.

**한계:** 금속 상 구별에 texture가 중요하면 shape bias 증가가 손해일 수 있다. FreqFusion과는 convolution 내부 처리 대 skip 융합이라는 차이가 있다. 둘을 처음부터 결합하지 않는다.

## X07. SCSegamba — 불규칙한 가는 구조를 고려하는 Mamba

**확인한 기여:** 구조물 균열 segmentation을 위한 SAVSS는 Gated Bottleneck Convolution(GBC)과 Structure-Aware Scanning Strategy(SASS)를 포함한다. 일반적인 scan보다 crack의 형태·텍스처와 semantic continuity를 겨냥한다. [CVPR2025 논문](https://arxiv.org/abs/2503.01113), [학회 페이지](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_SCSegamba_Lightweight_Structure-Aware_Vision_Mamba_for_Crack_Segmentation_in_Structures_CVPR_2025_paper.html), [공식 코드](https://github.com/Karl1109/SCSegamba).

**우리 제안:** 먼저 B의 중간 해상도 feature에 SAVSS 한 곳을 이식하는 파생 설계 또는 별도 full SCSegamba multiclass 대조를 준비한다. 단순 Mamba block만 쓰는 control과 GBC/SASS 포함 control을 분리한다. 예전 C10의 범용 bottleneck Mamba 제안과는 구체적인 형태 처리·scan이 차이다.

**한계:** 실제 crack과 Eutectic의 형태는 같지 않다. 원 논문의 scan이 금속 조직의 물리적 길을 직접 따라간다고 해석하지 않는다. Custom selective-scan의 CUDA/빌드 지원을 확인해야 하며, full scratch backbone 교체는 사전학습 차이도 포함한다.

## X08. Dino U-Net — 대형 encoder는 고정하고 변환부를 학습

**확인한 기여:** frozen DINOv3 backbone, 저수준 공간 detail을 합치는 adapter, decoder용 Fidelity-Aware Projection Module(FAPM)을 제안한다. 현재 확인한 논문은 2025 arXiv 사전 공개본이다. [논문](https://arxiv.org/abs/2508.20909), [저자 코드](https://github.com/yifangao112/DinoUNet), [DINOv3 공식 가중치 안내](https://github.com/facebookresearch/dinov3).

**우리 제안:** 작은 DINOv3 ViT-S/16부터 frozen encoder + detail adapter +4상 decoder 시스템 대조를 한다. Train70에서 큰 encoder 전체를 fine-tune하는 실험은 첫 후보로 삼지 않는다. 자체 Train70 SSL 또는 MatSSL 초기화 R012와 다른 사전학습 자원·전이 방식이다.

**한계:** patch16 특징만으로 가는 조직을 복원하기 어렵기 때문에 detail adapter를 별도 확인해야 한다. Grayscale microscopy로의 도메인 전이, 외부 가중치 확보와 extension 설치가 필요하다. 원 nnU-Net preprocessing/training까지 가져오면 현재 B와 구조 외 recipe도 달라지므로 그대로 공정 비교라고 부르지 않는다.

## X09. SegRefiner — 기존 예측 mask를 diffusion으로 수정

**확인한 기여:** coarse mask와 영상 조건에서 label 및 상태 전이 확률을 예측하며 discrete denoising diffusion으로 mask를 정제한다. Semantic/instance/dichotomous segmentation 실험을 보고한다. [NeurIPS2023 논문](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Abstract-Conference.html), [논문 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Paper-Conference.pdf), [저자 코드](https://github.com/MengyuWang826/SegRefiner).

**우리 제안:** B 예측 뒤 별도 refiner를 학습하는 대안이다. 처음에는 Train GT에 합성 오류를 만들어 훈련하고, 실제 B 오류를 훈련 입력으로 쓰려면 Train 내부 out-of-fold 예측을 준비한다. Valid mask/정답으로 refiner를 훈련하지 않는다. 원 이진/object mask 중심 구현을 네 상 categorical mask로 단순 채널 교체해 사용할 수 있다고 가정하지 않고 표현·loss·통합 방식을 추가 설계한다.

**한계:** 정확한 구조를 지우거나 다른 상으로 덮는 정제 실패, 합성 오류와 실제 오류의 차이, 추가 학습과 반복 추론 비용을 측정해야 한다. 신선한 방법이지만 현재 작은 seed별 개선을 검증하는 일보다 비용이 크므로 후순위다.

## 처음 고를 세 가지와 비교 조건

우선 **X01 주파수 융합 → X02a 연결 보조 감독 → X03 point 재분류**를 추천한다. 서로 skip 전달·학습 목표·최종 출력 정제에 개입한다. 상 혼동이 가장 중요하면 X04를 X03보다 먼저 선택해도 된다. 추론 추가 비용을 피하고 싶다면 X05가 대안이다.

첫 탐색은 같은 환경에서 `B / B+X01 / B+X02a / B+X03` 네 run으로 비교한다. 기본150epoch·seed42·FP32·batch4·context448/중앙224·sampling·증강·optimizer·loss·single best→same best D4 규칙을 유지한다. Point/connectivity 등 필수 새 감독만 별도로 명시한다. 단일 seed 탐색 뒤 유망한 후보만 같은 추가 seed의 B와 재현한다. 200epoch를 선택하면 B까지 모두 fresh200epoch로 맞춘다.

처음부터 D03/RRCU/U²-Net과 모두 결합하지 않는다. 독립 효과를 확인한 뒤 `B / +새 모듈 / +D03 / +둘 다`로 결합 기여를 비교한다. 공통 weight 초기화·sampling RNG를 가능한 한 맞추고 변경된 모델·optimizer step·parameter 수를 기록한다.

전체 single/D4 및 strict mIoU, Al3Ni/Eutectic IoU, 접촉부 오류, Eutectic precision·recall·skeleton recall, 이미지별 paired 차이, peak memory와 추론 시간을 함께 확인한다. Thin recall 개선이 오연결·오검출 증가와 맞바뀌지 않는지가 채택 기준이다.

다른 도메인 논문에서 입증된 성능은 그 논문의 데이터·평가 조건에 대한 결과다. 이 문서의 순위는 KoMaP에서 확인된 성능 순위가 아니다.
