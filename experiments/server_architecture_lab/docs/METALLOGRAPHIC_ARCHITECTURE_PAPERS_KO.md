# KoMaP 금속 조직 분할: 다른 아키텍처 논문 목록

조사일: 2026-10-01. 현재 작업인 금속 조직 영상의 픽셀 단위 분할을 기준으로 검색했다. 직접 관련 논문 12편과 인접 과제 참고 논문 2편을 정리했다. 기존에 검토한 MIMU-Net·MatSSL도 비교 위치를 명확히 하기 위해 포함했다.

구조·노벨티 상세 분석: [아키텍처 노벨티 분석](METALLOGRAPHIC_NOVELTY_BREAKDOWN_KO.md). 각 후보에 선택 ID와 모듈 단위 적용 가설을 정리했다.

적용 우선순위와 기대 효과는 우리 데이터·현재 구현에 대한 판단이다. 논문에서 KoMaP 개선을 검증한 결과가 아니다. 다른 데이터셋의 mIoU를 우리 점수와 나란히 놓아 성능 순위를 매기지 않았다. 코드 ‘공개 확인’은 저자 저장소/배포 페이지를 확인했다는 뜻이며, 다운로드·가중치 로딩·MPS 실행까지 검증했다는 뜻은 아니다. ‘미확인’은 이번 검색으로 확인하지 못했다는 뜻이다.

## 팀원 실험 결과 반영

2026-10-01 사용자가 전달한 팀원 결과. 아래 점수는 전달받은 값이며, 원본 로그·평가 조건은 아직 확인하지 않았다.

| 실험 | 전달받은 변경점 | 전달받은 점수 | 백분율 표기 |
| --- | --- | --- | --- |
| R021 | 희귀 타일 기준 완화 | 0.794084 | 79.4084% |
| R022 | ResNet101로 변경 | **0.798969** | **79.8969%** |
| R023 | ResNet152로 변경 | 0.796613 | 79.6613% |
| R024 | ConvNeXt-Tiny로 변경 | 0.797843 | 79.7843% |

팀원 결과 내 최고는 R022이다. 같은 조건이라는 전제에서 R022는 R023보다 0.2356 percentage point, R024보다 0.1126 percentage point 높다. 표본·seed 변동이 알려지지 않아 이 차이만으로 backbone의 우열을 확정하지 않는다. R021의 샘플링 변경이 R022–R024에 이어졌는지도 미확인이다.

이 결과는 사전학습 후보를 ResNet101까지 확장할 근거가 된다. NASA MicroNet 공식 목록에는 ResNet101과 ResNet152도 있다. 같은 ResNet101에서 ImageNet 초기화와 현미경 사전학습을 비교하면 backbone 용량 변경의 효과와 사전학습 효과를 분리할 수 있을 것으로 판단한다. 현재 진행 중인 ResNet34 SSL 실험은 기존 ResNet34 baseline과 먼저 비교하고, 유력하면 ResNet101으로 확장하는 순서가 적절하다. 별도 학습은 이번 논문 조사에서 시작하지 않았다.

R022를 공동 기준 모델로 확정하려면 Train/Valid/Test 분할, 점수의 지표·평균 방식, D4 여부, 타일/stride/halo, 학습 길이·checkpoint 선택, 희귀 타일 기준과 seed를 기존 실험과 대조해야 한다.

## 빠르게 볼 후보

| 후보 | 핵심 차이 | 우리 모델에서 확인할 가설 | 공개 구현 | 검토 우선도 |
| --- | --- | --- | --- | --- |
| MicroNet | 현미경 영상으로 사전학습한 CNN encoder | ResNet34 및 팀원 최고 ResNet101에서 사전학습 효과 확인 | 코드·가중치 목록 확인 | 가장 먼저 재현 검토 |
| CMAA-Net | 다중 스케일 dilation + 적응 attention + 채널 LSTM | 크기·텍스처가 비슷한 상의 혼동 감소 | 미확인 | 구조 변경 후보 1 |
| SegFormer + multi-scale cross-attention | 서로 다른 스케일의 특징 간 attention | 작은 조직과 주변 문맥의 결합 개선 | 제안 모델 코드 미확인 | 구조 변경 후보 2 |
| CS-UNet | CNN + Swin Transformer 병렬 encoder | 국소 경계와 넓은 문맥을 함께 사용 | 코드·사전학습 가중치 링크 확인 | 구조 변경 후보 3 |
| UPerNet / SegFormer 비교 연구 | 여러 Transformer encoder·decoder를 같은 SEM 과제에서 비교 | 큰 모델이 필요한지, 경량 Transformer로 충분한지 확인 | 해당 연구 구현 미확인 | 비교 실험 설계 참고 |
| MFF-UMamba | Mamba + 다중 스케일 fusion + deep supervision | 문맥 모델링 방식 자체를 변경 | 미확인 | 후순위, 실행 환경 확인 필요 |
| SAMM | 재료 영상에 맞춘 SAM2 전체 미세조정 | 다른 재료·촬영 조건으로 일반화 | 공식 코드·가중치 미확인 | 장기 후보 |

## 직접 관련 논문 12편

### 1. CMAA-Net — 2025

**CMAA: Channel-wise multi-scale adaptive attention network for metallographic image semantic segmentation**

Yongliang Sun, Xiangyang Huang. *Expert Systems with Applications*, 276, 126925.

- 구조: U-Net 형태에 Adaptive Parallel Attention Module(APAM)을 encoder·decoder에 적용한다. CMAA는 다중 스케일 dilated convolution과 attention head의 적응 점수, 채널 간 관계를 모델링하는 LSTM을 결합한다.
- 대상: MetalDAM 및 UHCS의 금속 미세조직 semantic segmentation.
- 적용 판단: 우리 과제와 직접 가깝다. 현재 scSE와 중복되는 부분이 있어, 작은 다중 스케일 모듈부터 교체 비교하는 접근이 적절하다. 모듈만 차용한 모델은 논문의 CMAA-Net 재현과 구분해야 한다.
- 코드: 공식 구현 미확인. 이번 확인 범위는 출판사 초록·공개 본문 발췌이며, 정확한 모듈 재현에는 전체 수식·구조 확인이 필요하다.

출처: [논문 및 DOI](https://www.sciencedirect.com/science/article/abs/pii/S0957417425005470), [DOI](https://doi.org/10.1016/j.eswa.2025.126925).

### 2. CS-UNet — 2024

**CS-UNet: A Flexible Segmentation Algorithm for Microscopy Images**

Khaled Alrfou 등. *CVPR Workshops 2024*, CV4MS.

- 구조: CNN과 Swin Transformer의 병렬 encoder로 국소 특징과 문맥을 얻어 결합한다. 현미경 영상 사전학습 모델을 encoder 및 Transformer decoder 초기화에 활용한다.
- 대상: Ni 기반 초합금 및 environmental barrier coating의 현미경 영상 분할, 소량 데이터·분포 변화 평가.
- 적용 판단: CNN 경계 표현을 유지하면서 문맥을 강화하는 후보다. 두 encoder를 사용하므로 현재 ResNet34 U-Net보다 학습 비용을 확인해야 한다.
- 코드: 저자 저장소와 MicroLite Swin 가중치 링크 공개 확인.
- 관련 버전: 2023년 arXiv **Transfer Learning for Microstructure Segmentation with CS-UNet: A Hybrid Algorithm with Transformer and CNN Encoders**는 같은 연구 계열이며 별도 아키텍처로 세지 않았다. 검색되는 의료 영상 동명 CS-Unet 저장소와 구분해야 한다.

출처: [CVPR 논문](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/html/Alrfou_CS-UNet_A_Flexible_Segmentation_Algorithm_for_Microscopy_Images_CVPRW_2024_paper.html), [저자 코드·가중치](https://github.com/Kalrfou/SwinT-pretrained-microscopy-models), [관련 사전 공개본](https://arxiv.org/abs/2308.13917).

### 3. SegFormer + multi-scale cross-attention — 2025

**금속 SEM 이미지 분할을 위한 멀티 스케일 크로스-어텐션 및 SegFormer 기반 시멘틱 분할 모델 개발**

변영훈, 윤준석, 이상아, 김민수, 원홍인, 윤종필. *제어·로봇·시스템학회 논문지*, 31(5), 566–572.

- 구조: ImageNet 사전학습 SegFormer의 인접 스케일 특징(1/4↔1/8, 1/8↔1/16, 1/16↔1/32)에 cross-attention을 적용하고 채널 concat으로 융합한다. 논문의 최종 모델은 B5이다.
- 대상: 적층 제조 스테인리스강 SEM 영상의 3개 클래스 semantic segmentation.
- 적용 판단: 스케일에 따라 같은 조직이 다르게 보이는 문제와 잘 맞는다. 먼저 작은 SegFormer B0/B1을 비교한 뒤 attention을 추가하면 효과를 분리할 수 있다. 작은 backbone을 쓰는 실험은 원 논문의 B5 재현과 다르다.
- 코드: 제안 모델의 공식 코드 미확인. 본문과 구조 설명은 공개되어 있다.

출처: [한국어 전체 논문](https://jicrs.icros.org/_PR/view/?aidx=47232&bidx=4265), [DOI](https://doi.org/10.5302/J.ICROS.2025.25.0063).

### 4. MFF-UMamba — 2026

**A mamba-architecture and multi-scale fusion enhanced artificial intelligent framework for precise microstructure segmentation from SEM image to EBSD labeling in medium Mn steel**

Yuanming Liu 등. *Composite Design and Manufacturing*, wqag018. 2026-06-18 온라인 공개, 조사 시점 accepted manuscript.

- 구조: Mamba 기반 표현에 multi-scale feature fusion과 deep supervision을 결합한다.
- 대상: EBSD로 라벨링한 중망간강 SEM 영상의 잔류 오스테나이트 분할; dual-phase steel에도 평가.
- 적용 판단: Transformer 외 문맥 모델링 대안이다. 현재 Mac MPS 환경에서는 구현이 요구하는 selective scan 등의 연산·의존성을 먼저 확인해야 한다. 이번 조사에서 실행 호환성은 검증하지 않았다.
- 코드: 공식 구현 미확인. 후속 조사에서 저널 본문을 추가 확인해 W-PSB 가중 융합과 SFRB/DFAB 보조 학습까지 분석했다. 상세 내용은 노벨티 분석 문서 N04에 있다.

출처: [저널 논문](https://academic.oup.com/cdm/advance-article/doi/10.1093/cdm/wqag018/8711488), [DOI](https://doi.org/10.1093/cdm/wqag018).

### 5. 금속 SEM에서 여러 Transformer 구조를 비교한 연구 — 2025

**Accelerated quantification of reinforcement degradation in additively manufactured Ni-WC metal matrix composites via SEM and vision transformers**

Mutahar Safdar 등. *Materials Characterization*, 229, 115645.

- 구조: SegFormer, Swin 기반 UPerNet, MaskFormer·Mask2Former, ViT 기반 DPT·Segmenter·SETR를 CNN DeepLabV3+와 비교한다. 새 단일 아키텍처 제안보다 재료 영상에서의 구조 선택 근거로 유용하다.
- 대상: Ni-WC 복합재 SEM의 matrix, carbide particle, dilution band, reprecipitated carbide 4개 클래스.
- 적용 판단: 해당 연구에서는 UPerNet·SegFormer가 어려운 조직 구분에서 유력했고, MaskFormer·Mask2Former는 반복 실행 간 변동이 컸다. 따라서 우리 과제에서도 대형 mask Transformer를 무조건 우선할 근거는 부족하다.
- 코드: 해당 연구의 공식 실행 코드 미확인. 각 원 아키텍처의 일반 공개 구현과 이 논문의 재현 코드는 구분해야 한다.

출처: [출판사 논문](https://www.sciencedirect.com/science/article/abs/pii/S1044580325009349), [저자 기관 공개 PDF](https://juser.fz-juelich.de/record/1047296/files/1-s2.0-S1044580325009349-main.pdf).

### 6. MicroNet encoder 사전학습 — 2022

**Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset**

Joshua Stuckner, Bryan Harder, Timothy M. Smith. *npj Computational Materials*, 8, 200.

- 구조: 10만 장 이상의 현미경 영상으로 CNN encoder를 사전학습하고 segmentation 모델에 전이한다. decoder 신구조보다는 사전학습 도메인을 바꾸는 연구다.
- 대상: 초합금·코팅 등 현미경 영상 분할, 소량 학습 데이터 조건.
- 적용 판단: 현재 encoder와 같은 ResNet34와 팀원 최고 모델의 ResNet101이 공식 MicroNet v1.0 목록에 있으며, MicroNet 단독 및 ImageNet→MicroNet 버전이 있다. decoder를 유지한 가중치 비교가 가능할 것으로 판단한다. 키 호환성, RGB→grayscale 변환, 정규화는 적용 전에 확인해야 한다. v1.1의 목록은 ResNet50이므로 버전을 고정해야 한다.
- 코드: NASA 공식 코드·가중치 목록 공개 확인. 가중치 파일 자체의 다운로드와 로딩은 이번 조사 범위에 포함하지 않았다.

출처: [논문](https://www.nature.com/articles/s41524-022-00878-5), [NASA 코드·가중치 목록](https://github.com/nasa/pretrained-microscopy-models).

### 7. Attention-guided U-Net ensemble — 2023

**Microstructural segmentation using a union of attention guided U-Net models with different color transformed images**

Momojit Biswas 등. *Scientific Reports*, 13, 5737.

- 구조: attention-guided U-Net과 dilated convolution을 사용하며 서로 다른 색 공간 영상으로 학습한 모델 결과를 결합한다.
- 대상: MetalDAM 금속 미세조직 분할.
- 적용 판단: 다중 스케일 dilation과 attention은 차용 후보다. 우리 grayscale 영상에서는 RGB→HSV 등 색 공간 변경이 새 색 정보를 만들지는 않으므로, 색 변환 ensemble 전체를 우선 적용할 이유는 약하다.
- 코드: 저자 저장소 공개 확인. 후속 정적 조사에서 attention 구현의 들여쓰기 등 수정·검증이 필요한 부분을 확인했다. 실행 검증은 하지 않았다.

출처: [논문](https://www.nature.com/articles/s41598-023-32318-9), [저자 코드](https://github.com/mb16biswas/attention-unet).

### 8. MAUNet / SASAPD — Sensors 2021 권호, 2020 온라인 공개

**Quantitative Analysis of Metallographic Image Using Attention-Aware Deep Neural Networks**

Yifei Xu 등. *Sensors*, 21(1), 43. 온라인 공개일 2020-12-23.

- 구조: 단상 영상용 MAUNet은 skip connection의 저주파 정보 중복을 줄이고 decoder에 spatial-channel attention을 적용한다. 다상 영상용 SASAPD는 anchor point 기반 검출 구조다.
- 대상: 단상 금속 조직 분석 및 다상 조직의 요소 검출.
- 적용 판단: MAUNet의 skip 정보 선택은 가는 Eutectic Si 보존을 위한 설계 참고가 된다. 논문의 단상 과제와 우리 4개 클래스 분할은 다르며, SASAPD도 semantic segmentation 모델과 동일하게 취급하면 안 된다.
- 코드: 후속 조사에서 저자가 연결한 저장소를 확인했으나 현재 README만 있어 모델 구현은 확보하지 못했다. [저자 저장소](https://github.com/ZhangYuewan/Metallographic-Image-Analysis)

출처: [논문](https://www.mdpi.com/1424-8220/21/1/43), [저자 초록·출판 정보](https://pubmed.ncbi.nlm.nih.gov/33374842/).

### 9. SAMM — 2026

**SAMM: A general-purpose segmentation model for material micrographs based on the segment anything model 2**

Jiahao Tu 등. *Advanced Powder Materials*, 5(4), 100404.

- 구조: SAM2 전체 파라미터 미세조정, cross-scale feature fusion, 픽셀 정확도와 구조 보존을 함께 고려하는 hybrid loss.
- 대상: 13개 재료 데이터셋 및 새로운 재료·촬영 조건으로의 일반화.
- 적용 판단: 범용 재료 encoder와 스케일 융합 관점에서 유망하다. 현재 4개 고정 상 클래스에 맞는 출력 head·prompt 처리·학습 비용은 상세 구현을 확인해야 한다. 초록의 높은 다른 데이터셋 점수가 KoMaP 개선을 뜻하지는 않는다.
- 코드: 공식 코드·가중치 배포 위치 미확인. 출판사 초록은 공개 데이터셋 제공을 언급하지만 배포 파일을 확인한 것은 아니다. 이번 구조 확인은 초록 수준이다.

출처: [논문](https://www.sciencedirect.com/science/article/pii/S2772834X26000126), [DOI](https://doi.org/10.1016/j.apmate.2026.100404).

### 10. MatSAM — 2025 저널, 2024 사전 공개본

**A novel training-free approach to efficiently extracting material microstructures via visual large model**

Changtai Li 등. *Acta Materialia*, 120962. 사전 공개본 제목은 **MatSAM: Efficient Extraction of Microstructures of Materials via Visual Large Model**.

- 구조: SAM에 재료 지식을 이용한 자동 point prompt 생성과 후처리를 결합한다. 분할 모델을 새로 학습하지 않는 접근이다.
- 대상: 결정립, 상, 결함 등 다양한 금속 미세조직 추출.
- 적용 판단: 사전 라벨 생성·영역 추출 보조에 유용하다. 생성한 영역을 Primary Si/Al3Ni/Eutectic Si/Al의 의미 클래스에 대응시키는 방식은 별도로 마련해야 한다.
- 코드: 저자 저장소 공개 확인.

출처: [저널 논문](https://www.sciencedirect.com/science/article/pii/S1359645425002538), [사전 공개본](https://arxiv.org/abs/2401.05638), [저자 코드](https://github.com/USTB-AI3DVIP/matsam).

### 11. MIMU-Net — 2024, 기존 검토 대상

**Metallographic image segmentation using feature pyramid based recurrent residual U-Net**

Samriddha Majumdar 등. *Computational Materials Science*, 244, 113199.

- 구조: recurrent residual feature extraction, squeeze-and-excitation, decoder feature pyramid.
- 대상: MetalDAM 미세조직 분할.
- 적용 판단: 우리 프로젝트에서 이미 검토·실험한 계열이다. 새 후보로 다시 세기보다 feature pyramid와 현재 사전학습 U-Net의 차이를 설명하는 비교 근거로 남긴다.
- 코드: 저자 저장소 공개 확인.

출처: [논문](https://www.sciencedirect.com/science/article/pii/S0927025624004208), [저자 코드](https://github.com/sammajum706/Feature-Pyramid-Recurrent-Residual-U-Net-for-Metallographic-Image-Segmentation).

### 12. MatSSL — 2025, 현재 실험의 참고 연구

**MatSSL: Robust Self-Supervised Representation Learning for Metallographic Image Segmentation**

*ICCV Workshops 2025*, CV4MS.

- 구조: 재료 영상 SSL, ResNet50의 다중 단계 특징과 gated feature fusion을 이용하는 projection head, 후속 U-Net++ 분할.
- 적용 판단: 현재 진행하는 ResNet34 단일 단계 SSL→기존 DySample U-Net 200 epoch 실험은 이 연구의 아이디어를 단계적으로 적용한 변형이다. MatSSL 전체 구조를 재현한 실험은 아니다. 다중 단계 SSL·gating 및 decoder 변경은 효과를 분리해 검토한다.
- 코드: 저자 공식 저장소 공개 확인. 논문·코드 차이의 상세 내용은 기존 `MATSSL_ARCHITECTURE_NOTES_KO.md`에 정리되어 있다.

출처: [논문](https://arxiv.org/abs/2507.18184), [공식 코드](https://github.com/aimat-ust/MatSSL).

## 인접 과제 참고 2편

### 13. SAM-I-Am — 2024 사전 공개본 / 2025 저널

**SAM-I-Am: Semantic Boosting for Zero-shot Atomic-Scale Electron Micrograph Segmentation**

*Computational Materials Science*, 246, 113400.

SAM이 생성한 mask를 기하·텍스처 정보로 필터링하고 합치는 접근이다. 주된 대상이 원자 규모 TEM 영상이므로 KoMaP의 4개 상 분할과 과제가 다르다. foundation model 출력의 후처리 설계 참고로 분류한다. 공식 코드 배포 위치는 이번 검색에서 미확인.

출처: [사전 공개본](https://arxiv.org/abs/2404.06638), [저널 논문](https://www.sciencedirect.com/science/article/pii/S0927025624006219).

### 14. Cellpose-SAM 기반 결정립 분석 — 2026

**Bridging Foundation Models and ASTM Metallurgical Standards for Automated Grain Size Estimation from Microscopy Images**

Abdul Mueez, Shruti Vyas. *CVPR Workshops 2026*, CVMI.

Cellpose-SAM을 금속 결정립에 맞춰 조정하고 topology-aware gradient tracking 및 결정립 크기 측정을 결합한다. 결정립 instance segmentation·계수 과제이므로 상별 semantic segmentation의 대체 모델로 바로 평가할 수는 없다. 조직의 연결·분리를 보존하는 접근과 소량 데이터 학습을 참고할 수 있다. 공식 코드 공개 확인.

출처: [CVPR 논문](https://openaccess.thecvf.com/content/CVPR2026W/CVMI/html/Mueez_Bridging_Foundation_Models_and_ASTM_Metallurgical_Standards_for_Automated_Grain_CVPRW_2026_paper.html), [저자 코드](https://github.com/mueez-overflow/ASTM-Grain-Size-Estimator).

## 우리 프로젝트에서 검토할 순서

1. **현재 SSL 실험을 끝내고 동일 평가 조건의 기존 모델과 비교한다.** SSL loss 감소만으로 분할 개선을 판정하지 않는다.
2. **팀원 최고 R022의 조건을 대조하고 MicroNet ResNet34/ResNet101 가중치 호환성을 확인한다.** 같은 backbone 안에서 구조·decoder·샘플링·loss·학습 길이를 유지하고 사전학습만 바꾸는 비교가 가장 간단하다. 필요한 입력 정규화는 가중치의 요구 조건을 확인해 기록한다.
3. **다중 단계 SSL/gating을 별도 검토한다.** 앞서 정한 MatSSL 단계별 계획을 유지한다.
4. **CMAA의 다중 스케일·적응 attention부터 구조 변경 후보로 검토한다.** 공식 구현이 확인되지 않았으므로 전체 모델 재현과 아이디어 차용을 구분한다.
5. **작은 SegFormer baseline→cross-attention 추가를 비교한다.** backbone 변경과 fusion 변경의 효과를 분리한다.
6. **CS-UNet을 독립 구조 후보로 검토한다.** pretrained CNN/Swin 선택과 계산량을 먼저 확인한다.
7. **Mamba·SAMM은 구현·가중치·MPS 실행 조건을 확보한 뒤 검토한다.**

비교 시 같은 Train/Valid 분할, 패치·평가 규칙, checkpoint 선택과 D4 조건을 유지한다. architecture에 따라 한 epoch의 작업량이 달라질 수 있으므로 200 epoch뿐 아니라 시간·메모리도 기록한다. 전체 mIoU와 함께 Eutectic Si IoU, precision/recall, 가는 조직 recall 및 이미지별 차이를 확인한다. 최초 후보 비교 이후 유력한 방법은 여러 seed에서 개선이 반복되는지 확인한다.
