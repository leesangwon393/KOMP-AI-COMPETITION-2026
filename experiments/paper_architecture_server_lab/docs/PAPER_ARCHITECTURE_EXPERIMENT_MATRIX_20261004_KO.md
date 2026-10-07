# 논문별 구조·백본·역할과 전체 실험 설계

조사일: 2026-10-04. 범위는 직전 읽기 목록의 12편과 추가 5편, 총 17편이다. 논문의 제안 구조를 빠짐없이 목록화하고, 공개 원문에서 확인한 주요 비교 모델과 backbone variants를 추가했다. 모든 유료 본문의 비교표·부록을 확보한 상태는 아니므로 **논문에 등장한 모든 대조군까지 완전히 추출했다고 주장하지 않는다.** 미확인 설계는 임의로 채워 재현이라고 부르지 않는다. 표의 장점은 메커니즘 및 우리 데이터에 대한 가설이며 KoMaP 성능 검증이 아니다.

이번 산출물은 구조 조사와 실험 설계다. 새 학습, 기존 queue 수정, 서버 실행은 하지 않았다. 기존 prepared 변형과 논문 전체 구조는 아래에서 구별한다.

## 기준선과 실험 변경 위치

현재 기준 구조 B는 `ImageNet V2 ResNet101 → U-Net decoder/scSE → 4상 raw logits`다. Context448/중앙224, epoch당420 crops, batch4, 150epoch, AdamW 3e-4, CE0.5+Dice0.25+Lovasz0.25를 유지한 대조군이다. D03는 기준 구성으로 채택하지 않는다.

| 위치 | 역할 | 바뀌는 대표 후보 |
|---|---|---|
| backbone/encoder | 명암·텍스처·형태·문맥을 여러 해상도의 feature로 추출 | EfficientNet, SE-ResNeXt, DenseNet, Swin, MiT, VMamba, RSU |
| decoder | 저해상도 의미와 고해상도 위치를 결합해 픽셀 예측 복원 | U-Net++, FPN, PSPNet, PAN, DeepLabV3+, CS decoder, IFD |
| skip/fusion/attention | 어느 특징과 어느 해상도를 얼마나 사용할지 조절 | APAM, CMAA, scSE, ECA, CBAM, OCR, FreqFusion, cross-attention, W-PSB |
| 학습·출력 | 학습 신호, 초기 표현, 예측의 일관성 또는 방향 관계를 변경 | MicroNet, EISC, connectivity/voting, deep supervision, PF-DiffSeg |

## 17편의 제안 구조: encoder부터 출력까지

| ID | 논문 | 사용 구조와 구성 요소 | 각 요소의 역할·장점 | 우리 데이터에서 확인할 가설 | 확인 수준 / 실험 상태 |
|---|---|---|---|---|---|
| P01 | [CMAA-Net, 2025](https://doi.org/10.1016/j.eswa.2025.126925) | U-Net형 encoder–decoder; APAM을 넣은 SECB 특징 추출; CMAA의 다중 dilation·adaptive multi-head attention·channel LSTM; APAM decoder | APAM은 공간·채널 중요도를 병렬 조절, dilation은 여러 조직 크기를 비교, LSTM은 채널 관계를 학습 | Al3Ni와 Eutectic Si의 유사한 텍스처를 문맥·스케일로 구분 | 공개 설명 확인. 채널 수·배치·정확한 연산/공식 코드 추가 확보 필요. 새 full 구현 없음 |
| P02 | [CS-UNet, 2024](https://openaccess.thecvf.com/content/CVPR2024W/CV4MS/html/Alrfou_CS-UNet_A_Flexible_Segmentation_Algorithm_for_Microscopy_Images_CVPRW_2024_paper.html) | CNN + Swin-T 병렬 encoder; bottleneck/skip의 concat·채널 투영; Swin형 patch-expanding decoder; 관련 preprint에는 ResMLP 변형 및 microscopy pretraining | CNN의 세부 형태와 Swin의 넓은 문맥을 여러 복원 단계에서 함께 사용 | 미세 Eutectic Si를 보존하면서 비슷한 상을 구분 | [관련 preprint 본문·부록](https://arxiv.org/html/2308.13917) 확인. preprint 세부 실험을 CVPRW 최종본과 동일하다고 단정하지 않음. 19 CNN 목록은 아래. full 구현 없음 |
| P03 | [MicroNet, 2022](https://www.nature.com/articles/s41524-022-00878-5) | 여러 CNN의 현미경 분류 사전학습 → 분류 head 제거 → U-Net/U-Net++/LinkNet/FPN/PSPNet/PAN/DeepLabV3+ | 구조와 별개로 encoder의 초기 텍스처 표현을 바꿈 | 70장 원본에서 더 적합한 초기 표현이 일반화에 도움이 되는가 | 본문·[NASA 가중치 목록](https://github.com/nasa/pretrained-microscopy-models) 확인. 현재 B 동조건 결과 없음 |
| P04 | [MAM-UNet, 2026](https://doi.org/10.1111/jmi.70055) | U-Net + ECA + CBAM | ECA: 채널 관계를 가볍게 조절; CBAM: 채널·공간 위치 강조 | scSE보다 공정 조직 텍스처 선택에 유리한가 | 초록 확인. 정확한 삽입 위치·encoder variant는 본문 확인 필요. full 구현 없음 |
| P05 | [Automatic Si phase extraction, 2025](https://doi.org/10.1016/j.mtcomm.2024.111468) | K-means, supervised U-Net; clean/noisy 혼합 훈련; K-means 기반 라벨 생성 비교 | 명암 군집과 학습 모델 비교, 영상 품질·상 비중에 따른 데이터 구성 | 실제 Train 명암 범위와 상 비중에 맞는 증강·sampling | 초록 확인. 별도 신형 backbone 제안 아님. K-means는 4상 GT 대체 근거 없음 |
| P06 | [Al–Si comparative study, 2025](https://doi.org/10.1007/s11665-024-10442-5) | DeepLabV3, FCN, LinkNet, SegNet, U-Net 5종 | 서로 다른 문맥 추출·위치 복원 방식을 동일 재료에서 비교 | 큰 Primary Si와 작은 Eutectic Si의 누락·붙음·면적 오차 | 초록에서 5종 확인. 각 encoder·구현 variant는 전체 본문 확인 필요 |
| P07 | [EISC, 2026](https://doi.org/10.3390/electronics15153425) | U-Net student + EMA U-Net teacher; 여러 decoder 분할 head; 내부 스케일/teacher–student 스케일 일관성 | 비라벨 영상의 일관성으로 라벨 부족 보완, 여러 크기 예측을 정렬 | 작은 구조와 큰 조직의 해석이 일관되는가 | 공개 본문 검색 발췌 확인. 별도 비라벨 Train 확보 필요. Train70으로 일부 감독 실험은 파생 방식. 새 구현 없음 |
| P08 | [FreqFusion, 2024](https://arxiv.org/abs/2408.12879) | 고/저해상도 feature fusion; ALPF generator + offset generator/resampling + AHPF generator | 내부 불일치를 줄이고 downsampling으로 사라진 경계 상세 복원 | 텍스처 내부 혼동과 얇은 Eutectic Si의 흐려짐 | 논문·공식 코드 확인. X01은 no-offset 등 축소 변형; full 결과 아님 |
| P09 | [OCR, 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510171.pdf) | dilated ResNet101 또는 HRNet-W48; coarse class-region head → spatial gather → pixel–region 관계 → context fusion → segmentation | 각 픽셀이 같은 클래스 영역의 표현을 참조 | Al3Ni/Eutectic Si의 국소 명암 혼동 감소 | 본문 확인. 현재 B에 OCR만 추가하는 실험은 별도 파생. full 구현 없음 |
| P10 | [U²-Net, 2020](https://arxiv.org/abs/2005.09007) | RSU encoder + RSU decoder; 6 side outputs + fusion head; 큰 U²-Net/작은 U²-Net†(U2NETP) | 블록 내부 U구조로 여러 receptive field 확보; 여러 단계 감독 | 큰 입자와 미세상 크기 차이에 대응 | 원 구조/코드 확인. 패키지 준비됐지만 받은 서버 결과에는 학습 없음. multiclass head로 변경 필요 |
| P11 | [DconnNet, 2023](https://arxiv.org/pdf/2304.00145) | 공개 코드 ResNet34; SDE(방향 prior·channel slicing·sub-path excitation); IFD(space/feature flow); 8방향 connectivity; bilateral voting/RCA; SDL | 방향 관계를 특징 추출부터 복원·출력까지 연결 | 얇은 조직의 끊김과 다른 상에 대한 잘못된 붙음 감소 | 본문·[저자 ResNet34 코드](https://github.com/Zyun-Y/DconnNet/blob/main/model/DconnNet.py) 확인. X02 aux head는 full SDE/IFD/voting 구현 아님 |
| P12 | [DSCNet, 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_Dynamic_Snake_Convolution_Based_on_Topological_Geometric_Constraints_for_Tubular_ICCV_2023_paper.html) | Dynamic Snake Convolution; 다중 방향 feature fusion; tubular topology 제약 | 굽은 관상 형태를 따라 sampling, 구조 보존 | 침상 형태에는 유리할 수 있으나 독립된 작은 입자를 잘못 연결하지 않는가 | 공개 논문 설명·코드 링크 확인. 기존 DSConv 파생 결과는 원 DSCNet full 결과 아님 |
| P13 | [SegFormer + cross-attention, 2025](https://jicrs.icros.org/_PR/view/?aidx=47232&bidx=4265) | MiT-B5의 4 stage; 인접 3쌍 cross-attention; 해상도/채널 정렬; concat + segmentation head. 비교: SegFormer-B0/B5, Mask2Former, MobileViT-U-Net | 스케일 간 관련 특징을 선택해 전달 | 작은 조직과 주변 큰 구조의 문맥을 함께 학습 | 본문 확인. B5를 B2로 바꾸는 것은 크기를 줄인 별도 변형. 구현 없음 |
| P14 | [MFF-UMamba, 2026](https://doi.org/10.1093/cdm/wqag018) | VMamba/VSS·SS2D 계열 U형 특징 추출/복원; W-PSB; SFRB; DFAB; shallow/deep supervision | 장거리 문맥, shallow/deep 가중 융합, 여러 깊이의 감독 | 유사한 상 구분과 저대비 형태 복원 | 공개 설명·기존 본문 조사 확인. exact VMamba version/코드 및 scan 연산 실행 추가 확인 필요 |
| P15 | [PF-DiffSeg, 2025 preprint](https://arxiv.org/pdf/2508.00896) | phase-fraction 조건부 6-stage diffusion U-Net; scale당 residual block 2개; 깊은 cross-attention; image-mask 동시 생성; 2× diffusion SR. downstream: ResNet50 + U-Net/U-Net++/LinkNet/MA-Net | 상 비중을 조절한 image–mask 증강; SR로 생성 해상도 보완 | 적은 Eutectic Si 면적을 포함한 다양성 확장 | 본문·Table A2/A3 확인. 생성기/분할기는 별개. 아직 구현 없음 |
| P16 | [MAUNet / SASAPD, 2021](https://www.mdpi.com/1424-8220/21/1/43) | MAUNet: low/high-frequency 분리 encoder/skip + spatial/channel dual-path attention decoder. SASAPD: SAPD 기반 detector + attention pyramid + hard anchor-point reweight/feature selection | MAUNet은 상세 정보 전달; SASAPD는 어려운 검출 sample·feature level 선택 | MAUNet의 skip 전달이 미세상 누락을 줄이는가 | 공개 설명 확인. 정확한 모든 layer 설정 추가 확인 필요. SASAPD는 검출 모델로서 4상 픽셀 분할의 직접 대조군 아님 |
| P17 | [MIMU-Net, 2024](https://doi.org/10.1016/j.commatsci.2024.113199) | recurrent residual encoder + scSE + feature pyramid decoder | 반복 특징 정제, 채널/위치 선택, 크기가 다른 조직 복원 | recurrence와 pyramid의 효과가 현재 B에서도 유효한가 | 논문 공개 설명·기존 코드 조사. 기존 MIMU/RRCU 계열과 중복 확인; full와 일부 recurrence 이식 구분 |

## 백본 가족별 역할과 모든 확인된 variant

아래 MicroNet 가중치 목록은 현재 공개 저장소의 v1.0/Im→Micro v1.0 목록이다. 논문에서 분류 사전학습한 모든 모델과 모든 segmentation 조합이 현재 공개 목록에 일대일 대응한다고 가정하지 않는다. B6/B7, plain VGG11/13은 Im→Micro 목록에서 확인되며 scratch Micro 목록에는 없다. v1.1 기본 모델은 R50이다.

| 가족 | 확인된 MicroNet 공개 variants | 역할·장점 / 우리 실험 판단 |
|---|---|---|
| ResNet | 18, 34, 50, 101, 152 | residual 경로로 깊은 CNN 학습. R101 구조 유지한 가중치 교체가 깨끗한 대조. R152는 깊이 증가 후보 |
| SE-ResNet | 50, 101, 152 | encoder 내부 채널 중요도 조절. decoder scSE와 위치가 다름 |
| ResNeXt | 50_32x4d, 101_32x8d | grouped convolution으로 여러 특징 경로 확보. 비슷한 텍스처 표현 다양성 가설 |
| SE-ResNeXt / SENet | SE-ResNeXt50_32x4d, 101_32x4d; SENet154 | grouped/residual 특징 + channel gating. encoder 비교 우선 후보 |
| EfficientNet | B0–B7; scratch Micro 공개는 B0–B5 | depth/width/resolution 균형, MBConv/SE. B3/B4 대표 비교부터; 큰 모델이 반드시 우위라는 근거 없음 |
| DenseNet | 121, 161, 169, 201 | 이전 층 특징을 이어 받아 재사용. 얕은 텍스처 전달 가설; activation 메모리 측정 필요 |
| DPN | 68, 68b, 92, 98, 107, 131 | residual과 dense 경로를 함께 활용. 전체 목록 포함, 대표 DPN92 비교 가능 |
| Inception | V4, ResNet-V2 | 서로 다른 크기의 convolution 경로, ResNet-V2는 residual 결합. 다중 형태 표현 후보 |
| Xception | Xception | depthwise separable convolution으로 공간/채널 연산 분리 |
| MobileNet | V2 | 가벼운 inverted residual. 최고 점수뿐 아니라 속도/메모리 대조군 |
| VGG | 11_bn, 13_bn, 16_bn; plain11/13(Im→Micro만) | 단순 CNN 비교. 메모리/계산 대비 현대 후보보다 후순위 |

위 목록의 출처: [NASA official weights](https://github.com/nasa/pretrained-microscopy-models). 역할은 각 구조의 일반적 설계 해석이며 MicroNet에서 모두 우위를 보였다는 뜻이 아니다.

**CS-UNet 관련 preprint Table10의 19 CNN:** SE-ResNet50/101/152, SE-ResNeXt50_32x4d/101_32x4d, SENet154, ResNeXt101_32x8d, InceptionV4, InceptionResNetV2, DenseNet161/201, VGG13_bn/16_bn, MobileNetV2, EfficientNetB1/B2/B3/B4/B5. 위 CNN 각각과 Swin-T를 결합하는 것이 preprint의 실험 범위다. 현재 R101과 Swin-T 결합은 우리 데이터용 별도 선택이다. [Table10 원문](https://arxiv.org/pdf/2308.13917)

| 추가 encoder | 등장 논문 / 출처 | 역할·장점 | 우리 비교 후보 |
|---|---|---|---|
| Swin-T | CS-UNet | shifted-window attention과 계층적 특징; 국소/넓은 문맥 연결 | Swin-T 단독 + 공통 decoder, CNN+Swin 병렬 각각 |
| MiT-B0/B5 | SegFormer cross-attention | overlapped patch embedding, efficient attention, multi-scale features | 원 논문 B0/B5와 별도 중간 크기 B2 |
| HRNet-W48 | OCR | 고해상도 경로를 유지하며 다른 해상도와 교환 | OCR native 비교. 기존 HRNet 결과를 OCR 유무 실험으로 오해하지 않음 |
| VMamba/VSS | MFF-UMamba | SS2D로 여러 방향의 장거리 정보 처리 | 원 논문 구현에 맞춘 version 확정 후 비교 |
| RSU | U²-Net | 블록 내부 U구조로 여러 receptive field를 묶음 | RSU+B decoder, full U²-Net, U2NETP |
| [ConvNeXt V2](https://github.com/facebookresearch/ConvNeXt-V2) | 직전 17편 밖의 **추가 백본 후보** | 큰 depthwise kernel, LayerNorm, GRN, FCMAE 사전학습 | Tiny/Base. 기존 ConvNeXt 실험이 V1이면 V2와 구별 |

## 논문에서 실제 비교한 분할 구조와 decoder 역할

중복 이름은 한 번만 실험 등록한다. 아래는 MicroNet 본문, CS preprint, Al–Si 비교 초록, SegFormer 본문, PF-DiffSeg 본문에서 확인한 비교 구조다. 단순 related-work 언급을 저자가 실제 학습한 모델로 세지 않았다.

| 구조 | 역할 / 장점 | 실제 비교 출처 | 우리 실험 구성 |
|---|---|---|---|
| U-Net | encoder 세부 특징을 skip으로 복원 | 여러 논문 | B 및 encoder 교체의 공통 decoder |
| U-Net++ | nested dense skip으로 깊이 차이를 단계적으로 연결 | MicroNet, PF-DiffSeg | 같은 R101로 decoder 차이 비교 |
| LinkNet | encoder feature에 효율적인 residual decoder 결합 | MicroNet, Al–Si, PF-DiffSeg | 같은 encoder 대조 |
| FPN | top-down pyramid와 lateral feature fusion | MicroNet | 여러 크기의 상 복원 비교 |
| PSPNet | pyramid pooling으로 넓은 문맥 집계 | MicroNet | 작은 구조 과평활 여부도 확인 |
| PAN | pyramid attention으로 문맥·해상도 융합 | MicroNet | 다중 스케일 attention 대조 |
| DeepLabV3+ | ASPP 문맥 + 얕은 feature의 경계 복원 | MicroNet | R101 유지, dilation/OS도 명시 |
| DeepLabV3 | ASPP 다중 dilation 문맥 | Al–Si | V3+와 별도 구조로 등록 |
| FCN | CNN feature를 convolution head로 픽셀 분류 | Al–Si | exact FCN variant 확인 후 비교 |
| SegNet | pooling index를 decoder unpooling에 활용 | Al–Si | 원 encoder/decoder 결합으로 비교 |
| MA-Net | position/multi-scale fusion attention decoder | PF-DiffSeg | 논문 R50와 우리 R101 비교 구분. MAUNet/MAM-UNet과 다른 모델 |
| Swin-Unet | Swin형 encoder와 decoder, patch expansion | CS preprint | 순수 Swin형 대조 |
| TransDeepLab | Swin계열 encoder + window-scale pyramid | CS preprint | CS 논문의 표기는 TransDeepLabV3+. source implementation 확인 |
| HiFormer | CNN/Swin 특징 + Double-Level Fusion | CS preprint | CS-UNet과 다른 CNN/Transformer 결합 대조 |
| SegFormer | MiT + all-MLP feature fusion head | SegFormer cross-attention | B0/B5 native baseline, cross-attention 유무 비교 |
| Mask2Former | multi-scale pixel decoder + masked query attention + mask/class head | SegFormer cross-attention | native loss/matching 포함. 정확한 encoder는 본문·구현 확인 필요 |
| MobileViT-U-Net | 경량 CNN/Transformer encoder + U-Net | SegFormer cross-attention | exact MobileViT size 확인 후 비교 |

출처: [MicroNet methods](https://www.nature.com/articles/s41524-022-00878-5), [CS preprint](https://arxiv.org/pdf/2308.13917), [Al–Si](https://doi.org/10.1007/s11665-024-10442-5), [SegFormer 연구](https://jicrs.icros.org/_PR/view/?aidx=47232&bidx=4265), [PF-DiffSeg](https://arxiv.org/pdf/2508.00896).

## 전체 구조를 실험하기 위한 비교 계획

실험은 전부 목록에 포함하되 다음 비교 목적을 구분한다. 대규모 cross-product를 먼저 만들면 structure/pretraining/loss 변화가 섞이고 예산도 불필요하게 커진다. 각 full 구조와 모듈별 효과를 따로 기록한다.

| 실험 묶음 | 반드시 포함할 대조 | 결과로 답할 질문 |
|---|---|---|
| E: encoder 교체 | B + 각 CNN/Swin/MiT/VMamba/RSU encoder + 공통 U-Net형 decoder | encoder 표현이 우리 데이터에서 더 좋은가? feature adapter·얕은 skip 추가 여부 기록 |
| D: decoder 교체 | 같은 R101·같은 초기화 + 위 decoder variants | 위치 복원·다중 스케일 결합이 더 좋은가? |
| F: 논문 전체 구조 | P01/P02/P04/P07–P14/P16-MAUNet/P17 full; 원 architecture와 4상 head | 전체 조합이 B를 이기는가? scratch/native loss/pretraining 차이까지 전체 시스템 성능으로 기록 |
| M: 구성 요소 제거/추가 | APAM only, dilation only, LSTM 추가, ECA only, CBAM only, FreqFusion offset 유무, OCR aux 유무, RSU/deep supervision, SDE/IFD/voting, DSConv/topology, W-PSB/SFRB/DFAB, RRCU/FPN | full 결과의 이득이 어느 요소에서 왔는가? 원 ablation 표는 확보한 범위로 맞추고 나머지는 우리 설계로 표시 |
| T: 사전학습 | 같은 encoder의 ImageNet / MicroNet / Im→Micro; CS의 Swin ImageNet/MicroLite 경로 | architecture와 초기 표현의 효과 분리 |
| A: 증강/일관성 | P05의 품질·상 비중 구성, P07의 labeled/unlabeled 구성, P15의 real-only/real+synthetic | 데이터와 학습 전략이 더 좋은가? 생성기/EMA 비용도 기록 |

SASAPD는 같은 논문에 쓰인 아키텍처이므로 표에서 제외하지 않았지만, 검출 라벨을 필요로 한다. semantic mIoU 비교표에 detector 결과를 그대로 넣지 않는다. 검출 응용까지 실제 실행하려면 독립된 출력·평가 실험으로 다룬다. PF-DiffSeg는 downstream 모델 4개만 실행하면 논문 전체 실험을 했다는 뜻이 아니며 generator/SR를 별도 구현해야 한다.

1차 screening은 모두 fresh seed42,150epoch를 기본으로 한다. 이는 원 논문의 epoch/optimizer 복제가 아니라 우리 비교 예산이다. 가중치 입력 정규화 및 native loss가 필수인 구조는 예외 설정을 명시한다. native full와 공통 recipe 이식 성능은 별도 열에 저장한다. 현미경 사전학습 사용 모델과 ImageNet-only 모델을 순수 구조 효과라고 비교하지 않는다.

유망 후보는 같은 환경의 B와 seed43/44를 추가해 확인한다. 모든 후보를 3seed까지 실행하는 계획으로 확장할 수 있다. checkpoint는 single-view로 선택하고 동일 checkpoint의 D4를 평가한다. whole-image macro 4상 mIoU, class IoU/recall, 접촉부/가는 구조 보조 지표, 시간, peak memory를 저장한다. 448 입력·FP32·batch4가 불가능한 모델은 공통 batch2 대조군 등 별도 regime로 묶는다. 무조건 batch를 몰래 바꿔 한 표에 섞지 않는다.

우리 데이터의 Eutectic Si에는 독립된 입자/단절 구조도 있으므로 topology loss로 전체 연결을 강제하지 않는다. EISC에는 별도 unlabeled Train을 확보하거나 Train70에서만 supervised consistency 변형으로 실험한다. Valid/Test는 teacher 학습·생성기 학습에 편입하지 않는다. D03를 모든 새 architecture에 자동 결합하지 않는다.

우선 encoder 대표군은 ConvNeXt V2-T/B, EfficientNet-B3/B4, SE-ResNeXt50_32x4d, DenseNet161, Swin-T, MiT-B2/B5, R101 Im→Micro, RSU다. 이는 KoMaP에서의 사전 우열이 아니라 표현 방식·비용 다양성을 위한 순서다. 전체 encoder 목록과 모든 full 구조는 후속 목록에서 유지한다.

현재 구현 상태는 [읽기 목록](TASK_MATCHED_PAPER_READING_LIST_20261004_KO.md), [U²-Net package](../experiments/u2net_server_lab/README_KO.md), [followup package](../experiments/followup_unet_lab/README_KO.md)를 참고한다. 공식 코드 공개 여부와 실제 재현 완료를 구별한다. 논문 full structure를 실행했다고 기록하려면 고정 source/version, module parity, 입력/출력 검사, 학습 loss, 초기화 및 artifact를 확인해야 한다.
