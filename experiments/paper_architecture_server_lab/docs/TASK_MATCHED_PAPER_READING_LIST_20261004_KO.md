# 우리 4상 금속 조직 분할에 맞춘 논문 읽기 목록

작성·검색 기준: 2026-10-04. 추천 순서는 우리 데이터·실험에 대한 판단이며 논문이 우리 데이터에서 검증됐다는 뜻은 아니다. 원 논문의 다른 데이터셋 점수를 우리 Valid mIoU와 비교하지 않는다. 이번 작업은 문헌 조사이며 추가 모델 학습이나 실행 큐 변경은 하지 않았다.

## 선정 근거: 데이터와 실제 실패 양상

`Data/train` 70장, `Data/valid` 20장, `Data/test` 10장을 직접 집계했다. Test 라벨은 없다. RGBA 저장 형식이지만 모든 원본의 RGB 채널은 동일해서 실제 입력 정보는 회색조다. 이미지 크기는 675×480부터 1440×1024까지 다르다. 크기 차이만으로 현미경 배율 차이를 단정하지 않는다. 정확한 합금 조성, 촬영 장비, 물리적 픽셀 크기는 이번 조사에서 확인하지 않았다.

| 클래스 | Train 전체 픽셀 비중 | Valid 전체 픽셀 비중 | Train에서 출현한 이미지 수 |
|---|---:|---:|---:|
| Primary Si | 9.4283% | 10.1690% | 69/70 |
| Al3Ni | 26.3143% | 21.1565% | 70/70 |
| Eutectic Si | 6.9769% | 7.6561% | 70/70 |
| Al | 57.2806% | 61.0185% | 70/70 |

이 수치는 전체 픽셀을 합산한 면적 비중이다. 평가의 이미지별 4클래스 mIoU 평균과 집계 방식이 다르다. Eutectic Si는 영상 출현 자체가 드문 클래스가 아니라 **면적이 적고 작거나 얇은 구조가 많은 클래스**다. 많은 crop을 만들어도 원본 이미지 다양성이 늘어나는 것은 아니다.

원본·GT 예시 `A101_step3_sample3`, `A125_step2_sample1`에서 큰 각진 입자, 굵은 판상·침상 조직, 미세한 점·가는 조직이 함께 관찰됐다. 모든 이미지의 형태를 정량 분류한 결과는 아니다. Eutectic Si를 하나의 연결된 혈관망처럼 가정할 근거는 없다. 밝기만으로 유사한 상을 구별하기 어렵다는 문제에 맞춰 형태·텍스처·주변 문맥을 다루는 연구를 우선한다.

기존 `runs/confusion_audit_v1/RESULTS_KO.md`의 R34/scSE 분석에서는 Eutectic Si의 FP가 Al3Ni와 자주 혼동됐다. 그 수치는 현재 R101 기준선의 새 오류 재측정이 아니다. 서버 D03 추가 seed43/44는 접촉부 오류를 줄였지만 D4 평균은 B보다 0.1349%p 낮았다. 따라서 경계 CE 강화 하나만으로 전체 성능을 해결한다는 가설보다 **상 구분에 쓰는 표현과 복원 품질을 개선하는 가설**을 우선한다. 추가 seed는 2개여서 보편적 우열을 확정한 결과는 아니다.

집계 원자료: [DATA_CHARACTERISTICS_PAPER_SELECTION_20261004.json](DATA_CHARACTERISTICS_PAPER_SELECTION_20261004.json). 서버 결과: [D03_ADDITIONAL_SEEDS_SERVER_RESULTS_20261004_KO.md](D03_ADDITIONAL_SEEDS_SERVER_RESULTS_20261004_KO.md).

## 먼저 읽을 12편

### 1. CMAA — 작은 금속 조직 데이터에서 다중 스케일 상 구분

**Yongliang Sun, Xiangyang Huang. “CMAA: Channel-wise multi-scale adaptive attention network for metallographic image semantic segmentation.” Expert Systems with Applications 276, 126925 (2025).** [논문](https://www.sciencedirect.com/science/article/abs/pii/S0957417425005470) · [DOI](https://doi.org/10.1016/j.eswa.2025.126925).

- 논문 내용: APAM, 다중 스케일 dilated convolution과 adaptive attention, 채널 관계를 모델링하는 LSTM을 사용한다. MetalDAM/UHCS에서 평가했다.
- 우리에게 중요한 이유: 적은 라벨, 복잡한 텍스처, 조직 크기 차이라는 문제가 직접 맞는다. **구조 아이디어 읽기 최우선**.
- 적용 제안: B의 중간 decoder/skip 한 곳에서 다중 스케일 분기와 적응형 융합을 비교한다. 이미 scSE가 있으므로 attention을 겹쳐 넣기 전에 scSE 대체 또는 다중 스케일만 추가한 대조군을 둔다. 전체 LSTM 설계는 후순위.
- 확인 범위: 출판사 초록·공개 설명. 정확한 배치·채널 수·전체 재현 설정은 본문 확인 필요. 공식 코드 미확인. 우리 구현·학습 결과 없음.

### 2. CS-UNet — 미세한 텍스처와 넓은 문맥을 함께 사용

**Khaled Alrfou, Tian Zhao, Amir Kordijaz. “CS-UNet: A Flexible Segmentation Algorithm for Microscopy Images.” CVPR Workshops, CV4MS, pp. 8114–8119 (2024).** [논문·PDF](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/html/Alrfou_CS-UNet_A_Flexible_Segmentation_Algorithm_for_Microscopy_Images_CVPRW_2024_paper.html) · [저자 코드](https://github.com/Kalrfou/SwinT-pretrained-microscopy-models).

- 논문 내용: CNN과 Transformer encoder를 병렬로 사용하고, 각 encoder의 사전학습을 선택해 결합한다. 재료 현미경 영상의 적은 데이터·분포 변화에서 평가한다.
- 우리에게 중요한 이유: Al3Ni–Eutectic Si는 작은 위치 정보와 주변 조직 문맥을 함께 봐야 한다는 가설에 부합한다.
- 적용 제안: 먼저 기존 CNN을 유지하고 낮은 해상도 문맥 branch 하나만 융합한다. 이는 CS-UNet의 아이디어를 빌린 변형이지 원 논문 전체 재현이 아니다. 전체 이중 encoder는 Train 70장·연산비를 고려해 후순위.
- 확인 범위: CVF 초록·저자 공개 저장소. 우리 구현·학습 결과 없음. 같은 연구의 preprint를 별도 논문으로 중복 집계하지 않았다.

### 3. MicroNet — 구조보다 사전학습을 바꾸는 후보

**Joshua Stuckner, Bryan Harder, Timothy M. Smith. “Microstructure segmentation with deep learning encoders pre-trained on a large microscopy dataset.” npj Computational Materials 8, 200 (2022).** [논문](https://www.nature.com/articles/s41524-022-00878-5) · [NASA 코드·가중치](https://github.com/nasa/pretrained-microscopy-models).

- 논문 내용: 대규모 현미경 영상으로 사전학습한 encoder를 미세조직 분할에 전이한다.
- 우리에게 중요한 이유: 원본 70장에서는 새로 학습하는 복잡한 구조보다 초기 표현 개선이 유리할 가능성이 있다.
- 적용 제안: 같은 R101 U-Net/scSE에서 ImageNet, MicroNet, ImageNet→MicroNet 초기화만 비교한다. 공개 저장소에 R101 v1.0 두 경로가 있다. 기본 최신 v1.1은 R50이라 버전을 명시해야 한다. 입력 정규화도 맞춘다.
- 확인 범위: Nature 본문·NASA 저장소. 예전 MatSSL 실험의 악화가 별개의 MicroNet 전이까지 부정하지는 않는다. 현재 B에 대한 동조건 우위는 미확인.

### 4. MAM-UNet — 광학 현미경 공정 조직 분할

**Meng’ao Li et al. “MAM-UNet: A multiple attention mechanism UNet for the eutectic segmentation of superalloys.” Journal of Microscopy 301(3), 408–421 (2026; online 2025-12-19).** [논문](https://onlinelibrary.wiley.com/doi/10.1111/jmi.70055) · [DOI](https://doi.org/10.1111/jmi.70055).

- 논문 내용: 광학 현미경의 니켈 초합금 공정 영역을 ECA와 CBAM을 넣은 U-Net으로 분할한다.
- 우리에게 중요한 이유: 현미경 명암과 공정 조직 분할이라는 점은 가깝다. 단, 이 논문의 공정 영역과 우리 Eutectic Si 단일 상은 같은 라벨 정의가 아니다.
- 적용 제안: 기존 scSE를 ECA 또는 CBAM으로 각각 교체하는 비교부터 한다. 세 attention을 동시에 겹친 결과로 각 모듈의 효과를 주장하지 않는다.
- 확인 범위: Wiley 초록·서지. 전체 배치 및 공식 코드 미확인. 우리 구현·학습 결과 없음.

### 5. Automatic Si phase extraction — 재료·영상 문제의 직접적인 참고

**“Automatic Si phase extraction from microscopic images of Al-Si alloys by unsupervised machine learning and supervised deep learning.” Materials Today Communications 42, 111468 (2025).** [논문](https://www.sciencedirect.com/science/article/abs/pii/S2352492824034512) · [DOI](https://doi.org/10.1016/j.mtcomm.2024.111468).

- 논문 내용: Al–Si 광학 영상에서 K-means/U-Net을 비교하고 깨끗한 영상과 산화로 잡음이 생긴 영상을 섞는 훈련, 데이터 수, Si 비중을 살핀다.
- 우리에게 중요한 이유: Si 조직·명암·영상 품질이라는 문제를 직접 다룬다. 새 아키텍처보다 데이터 구성 아이디어를 얻는 논문이다.
- 적용 제안: 실제 Train의 밝기·대비와 미세상 비중에 따른 crop 분포를 점검하고 그 범위에서 증강을 비교한다. 우리 영상에 산화 잡음이 있다고 단정하지 않는다. K-means는 Al3Ni가 있는 4상 GT를 대신할 근거가 없다.
- 확인 범위: 출판사 초록. 저자 코드 미확인. 논문의 accuracy를 우리 mIoU와 비교하지 않는다.

### 6. Al–Si comparative study — 픽셀 점수와 조직 형태 오류를 함께 해석

**Mahmut Furkan Kalkan et al. “A Comprehensive Comparative Study on Semantic Segmentation for Automated Microstructural Measurement in Al–Si Alloys.” Journal of Materials Engineering and Performance 34, 10741–10753 (2025; online 2024-11-25).** [논문](https://link.springer.com/article/10.1007/s11665-024-10442-5).

- 논문 내용: U-Net 등 다섯 모델로 다양한 Al–Si 조직을 분할하고 픽셀 지표뿐 아니라 입자 개수·평균 면적 오류를 비교한다. 다중 라벨 hypereutectic 조건에 Si 종류별 분석도 있다.
- 우리에게 중요한 이유: Primary Si와 Eutectic Si의 형태를 구분하는 작업에 가깝다. 비슷한 IoU에도 조직 측정 결과가 다를 수 있다.
- 적용 제안: mIoU를 주목표로 유지하면서 클래스 면적 비중, 작은 Eutectic Si 누락, 붙음·끊김을 보조 분석한다. 연결 성분 수를 물리적 입자 수로 동일시하지 않는다.
- 확인 범위: 출판사 초록·서지. 새 모듈 제안 논문이 아니라 태스크 이해·오류 분석 참고. 코드 미확인.

### 7. EISC — 다중 스케일 예측의 일관성, 최신 금속 조직 연구

**Honggang Li, Yiming Zhang, Shiyu Du. “EISC: Enforcing Consistency Across Different Scales for Metallographic Image Segmentation.” Electronics 15(15), 3425 (2026-08-03).** [논문](https://www.mdpi.com/2079-9292/15/15/3425) · [DOI](https://doi.org/10.3390/electronics15153425).

- 논문 내용: U-Net/Mean Teacher 기반으로 decoder 내부 및 teacher–student 사이의 다중 스케일 예측 일관성을 학습한다. 금속 조직의 소수·작은 구조를 다룬다.
- 우리에게 중요한 이유: 작은 Eutectic Si와 큰 입자 사이의 표현 차이, 라벨 부족에 맞는 학습 아이디어다.
- 적용 제안: 원래 방식은 별도 비라벨 훈련 데이터가 필요하다. 현재 라벨이 있는 Train 70장으로는 보조 decoder 예측 일관성만 먼저 검토할 수 있으며 이는 변형이다. 큰 스케일 예측에 맞추다 얇은 구조를 지울 위험을 점검한다. Valid/Test를 훈련에 편입하지 않는다.
- 확인 범위: 출판사 검색에 노출된 본문·버전 이력. 단일 seed 결과이며 본문에 일부 설정 표기가 서로 다르므로 재현 시 점검 필요. 코드 미확인. 우리 실험 없음.

### 8. FreqFusion — decoder의 내부 일관성과 경계 복원을 함께 개선

**Linwei Chen et al. “Frequency-aware Feature Fusion for Dense Image Prediction.” IEEE TPAMI 46(12), 10763–10780 (2024).** [공개 논문](https://arxiv.org/abs/2408.12879) · [저자 코드](https://github.com/Linwei-Chen/FreqFusion) · [DOI](https://doi.org/10.1109/TPAMI.2024.3449959).

- 논문 내용: adaptive low-pass, high-pass, offset resampling을 결합해 feature의 내부 불일치와 경계 위치 오류를 줄인다. ECCV 논문이 아니라 TPAMI 논문이다.
- 우리에게 중요한 이유: 비슷한 텍스처의 내부 오분류와 얇은 구조의 복원 손실을 동시에 다룰 수 있다. 금속 조직에서 직접 검증된 결과는 아니다.
- 적용 제안: B decoder 한 단계에서 ALPF/AHPF 융합부터 비교한 후 offset 여부를 분리한다.
- 우리 상태: `followup_unet_lab`의 X01은 offset 등이 빠진 독립 축소 구현이다. 검증 코드는 준비됐지만 이번 선정 시점에 완료 성능 결과는 없다. 원 논문 전체 재현이라고 부르지 않는다.

### 9. OCR — 유사한 상을 클래스 문맥으로 구분

**Yuhui Yuan, Xilin Chen, Jingdong Wang. “Object-Contextual Representations for Semantic Segmentation.” ECCV (2020).** [공개 논문](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510171.pdf) · [저자 코드](https://github.com/openseg-group/openseg.pytorch).

- 논문 내용: coarse class map으로 클래스별 영역 표현을 만들고 각 픽셀이 그 표현을 참조한다. 문자 인식 OCR과 다른 의미다.
- 우리에게 중요한 이유: Al3Ni와 Eutectic Si의 국소 명암이 비슷해도 같은 영상의 상별 표현을 이용할 가능성이 있다.
- 적용 제안: B 중간 decoder feature에 4클래스 soft-region aggregation을 추가한다. Eutectic Si가 coarse map에서 지워지면 잘못된 상 표현을 강화할 수 있어 클래스 recall도 확인한다. 자연 영상에서 얻은 이득이 우리 데이터의 이득을 보장하지 않는다.
- 확인 범위: ECCV 공개 본문. 기존 HRNet 전체 모델의 결과와 OCR 모듈 단독 효과는 구별한다. 우리 구현·학습 결과 없음.

### 10. U²-Net — RSU 안에서 여러 크기의 형태를 포착

**Xuebin Qin et al. “U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection.” Pattern Recognition 106, 107404 (2020).** [공개 논문](https://arxiv.org/abs/2005.09007) · [저자 코드](https://github.com/NathanUA/U-2-Net) · [DOI](https://doi.org/10.1016/j.patcog.2020.107404).

- 논문 내용: ReSidual U-block(RSU) 안에 작은 U구조를 넣어 여러 receptive field의 정보를 결합한다. 원 태스크는 이진 salient object detection이다.
- 우리에게 중요한 이유: 큰 Primary Si와 미세한 Eutectic Si의 크기 차이에 대응하는 CNN 구조 아이디어다.
- 적용 제안: RSU를 일부 단계에 적용하거나 4클래스 출력으로 수정한 모델을 비교한다. 최종 sigmoid 이진 출력을 그대로 쓰지 않는다. 전체 scratch 모델과 ImageNet B의 비교는 구조와 사전학습 효과가 섞이므로 대조군을 구분한다.
- 우리 상태: `u2net_server_lab`은 준비됐지만 사용자가 전달한 B/D03 추가 seed 결과에는 U²-Net 학습이 없다. 성능 우위 미확인.

### 11. DconnNet — 학습 중 픽셀의 국소 방향 연결성을 가르침

**Ziyun Yang, Sina Farsiu. “Directional Connectivity-based Segmentation of Medical Images.” CVPR (2023).** [공개 논문](https://arxiv.org/abs/2304.00145) · [저자 코드](https://github.com/Zyun-Y/DconnNet).

- 논문 내용: 방향별 연결성 표현으로 픽셀 관계를 학습한다. 원 방법의 구조와 voting 단계까지가 전체 모델이다.
- 우리에게 중요한 이유: 얇은 조직의 끊김과 인접한 다른 상의 잘못된 붙음을 보조 학습으로 다룰 수 있다.
- 적용 제안: 4상 각각의 8방향 국소 연결성 aux head부터 비교한다. 클래스0은 Primary Si여서 의료 코드의 background 처리 관례를 그대로 옮기지 않는다. 떨어진 조직까지 강제로 연결하지 않는다.
- 우리 상태: X02는 training-only 보조 head를 넣은 변형이며 원 DconnNet 전체가 아니다. 완료 성능 결과 없음. 작은 aux loss도 전체 mIoU로 채택 여부를 판단한다.

### 12. Dynamic Snake Convolution / DSCNet — 얇고 굽은 조직을 따라가는 convolution

**Yaolei Qi et al. “Dynamic Snake Convolution Based on Topological Geometric Constraints for Tubular Structure Segmentation.” ICCV (2023).** [논문·PDF](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_Dynamic_Snake_Convolution_Based_on_Topological_Geometric_Constraints_for_Tubular_ICCV_2023_paper.html) · [저자 코드](https://github.com/YaoleiQi/DSCNet).

- 논문 내용: 가변적인 snake convolution, 여러 방향 feature, topology 제약으로 관상 구조를 분할한다.
- 우리에게 중요한 이유: 일부 가는 침상 조직의 형태에 맞는 convolution을 설계하는 참고다. 큰 입자·독립된 점상 조직에는 같은 사전 가정이 맞지 않을 수 있다.
- 적용 제안: 낮은 층/고해상도 decoder 한 곳의 convolution만 비교한다. 전역 연결성을 강제하는 loss는 함께 가져오지 않는다.
- 우리 상태: 예전 R34의 DSConv 계열 변형에서 소폭 이득을 본 기록이 있지만 현재 R101 B에서 동조건 재현은 미확인. 원 DSCNet 전체의 검증 결과로 해석하지 않는다.

## 추가로 읽을 5편: 우선순위와 차이를 분명히 둠

| 논문 | 가져올 아이디어 | 우선순위·현재 범위 |
|---|---|---|
| [금속 SEM 이미지 분할을 위한 멀티 스케일 크로스-어텐션 및 SegFormer 기반 시멘틱 분할 모델 개발](https://jicrs.icros.org/_PR/view/?aidx=47232&bidx=4265), 변영훈 외, ICROS 31(5), 566–572 (2025), DOI 10.5302/J.ICROS.2025.25.0063 | 서로 다른 해상도 feature를 cross-attention으로 연결 | 한국어로 구조를 읽기 좋은 자료. 금속 SEM의 다른 상 구성이다. 기존 B의 두 중간 해상도 융합부터 제안; 전체 SegFormer 교체보다 후순위. 출판사 본문 확인. |
| [MFF-UMamba](https://doi.org/10.1093/cdm/wqag018), Yuanming Liu 외, Composite Design and Manufacturing (2026; online 06-18), “A mamba-architecture and multi-scale fusion enhanced artificial intelligent framework for precise microstructure segmentation from SEM image to EBSD labeling in medium Mn steel” | 장거리 문맥, 다중 스케일 융합, deep supervision | 저대비 금속 상 구분에 관련됨. SEM 입력/EBSD 라벨, 두 상 조건은 우리 데이터와 다르다. 원고 초록 확인; 현재 출판된 호 정보로 연결되나 본문 변경 전체 대조는 하지 않음. 공식 코드·MPS 호환성 미확인. 연산·재현 부담 때문에 후순위. |
| [PF-DiffSeg](https://arxiv.org/abs/2508.00896), Nguyen 외, “Phase-fraction guided denoising diffusion model for augmenting multiphase steel microstructure segmentation via micrograph image-mask pair synthesis” (2025 preprint) | 상 면적 비중을 조건으로 image–mask 쌍 생성, 소수 상 강화 | 라벨 70장이라는 문제와 잘 맞는 장기 후보. 별도 생성 모델 비용 및 합성 GT 품질 검증 필요. 우선 Train의 상 비중을 고려한 sampling 개선으로 가설을 점검. 이번 확인은 arXiv 초록이며 정식 출판 여부를 단정하지 않음. |
| [MAUNet](https://www.mdpi.com/1424-8220/21/1/43), Yifei Xu 외, “Quantitative Analysis of Metallographic Image Using Attention-Aware Deep Neural Networks,” Sensors 21(1), 43 (2021; online 2020-12-23) | skip에서 저·고주파 정보와 attention을 다루는 관점 | FreqFusion과 함께 읽을 금속 조직 선행 연구. 논문의 single-phase segmentation과 별도 multiphase detection을 우리 4상 semantic segmentation과 구별. 공개 초록 범위 확인. |
| [MIMU-Net](https://www.sciencedirect.com/science/article/pii/S0927025624004208), Samriddha Majumdar 외, “Metallographic image segmentation using feature pyramid based recurrent residual U-Net,” Computational Materials Science 244, 113199 (2024) · [저자 코드](https://github.com/sammajum706/Feature-Pyramid-Recurrent-Residual-U-Net-for-Metallographic-Image-Segmentation) | recurrent residual, scSE, feature pyramid | 이미 검토·실험 계열이 존재한다. 새 미실험 후보로 중복 추천하기보다 CMAA/RSU와 설계 차이를 비교하는 배경 자료. 출판사 공개 설명 확인. |

## 읽고 구현으로 옮길 때의 순서

1. **태스크 이해:** Al–Si 두 논문으로 Si 라벨, 영상 품질, 형태 오류를 먼저 본다.
2. **모듈 우선:** CMAA의 multi-scale 분기와 FreqFusion을 읽는다. 구조에서 어디서 세부 형태가 사라지는지, 어떤 융합으로 복원하는지 본다.
3. **데이터가 적을 때의 표현:** MicroNet을 읽고 같은 R101에서 사전학습만 교체하는 실험을 검토한다.
4. **문맥 강화:** CS-UNet/OCR 중 하나의 작은 branch 또는 module을 검토한다.
5. **별도 구조·보조 학습:** U²-Net/RSU, connectivity, EISC를 각각 검토한다. MAM은 이미 사용하는 scSE와 중복을 따져 대체 비교한다.

구현 우선순위는 읽기 순서와 다를 수 있다. X01/X02 및 U²-Net 패키지는 이미 준비되어 있으므로 문헌 읽기는 그 구현이 생략한 부분을 이해하는 데도 쓴다. 준비된 코드가 있다는 사실은 성능이 검증됐다는 뜻이 아니다.

새 후보의 1차 비교에서는 ImageNet 초기화, crop, augmentation, sampling, AdamW, 기존 CE+Dice+Lovasz, epoch 예산을 고정하고 한 요소만 바꾼다. 후보 채택은 동일 조건 B와 복수 seed에서 비교한다. 최고 checkpoint는 single-view로 고르고 해당 checkpoint의 D4를 평가하며, 전체 이미지-macro 4상 mIoU와 상별 IoU/recall을 함께 보고 접촉부·얇은 구조 지표는 해석용으로 둔다. 원 논문 설정을 그대로 복제한 실험과 모듈만 빌린 변형은 결과표에서 구분한다.

조사 범위: 공식 출판사, CVF/ECVA, 저자 arXiv, 저자/NASA 저장소를 사용했다. 전 논문의 PDF 전체를 읽어 재현 설정까지 감사한 체계적 문헌 검토는 아니다. 일부 출판사 페이지는 접근 제한으로 공식 검색에 공개된 초록/본문 설명을 사용했으며 각 항목에 확인 범위를 적었다.
