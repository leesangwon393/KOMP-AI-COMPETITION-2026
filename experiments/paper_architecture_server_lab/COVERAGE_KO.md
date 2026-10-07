# 논문별 실제 실행 범위

127개는 논문127편이나 완전 재현127종이라는 뜻이 아니다. 공통 decoder의 encoder53개, decoder13개, native9개, module27개, hybrid20개, pretraining2개, control3개다. `EXPERIMENTS.csv`에서 정확한 ID/config를 확인한다.

| 논문/후보 | 이 ZIP의 실행 구성 | 구현 수준 및 미포함 부분 |
|---|---|---|
| P01 CMAA-Net | APAM / multi-dilation / channel LSTM / 조합 CMAA 모듈 | 독립 파생 모듈. 원 SECB 배치·adaptive multi-head 등 전체 CMAA-Net 미재현 |
| P02 CS-UNet | Swin-T/S + 공통 decoder; 원 preprint19CNN 및 R101 각각 + Swin-T 병렬 융합 | 20개 derived hybrid. concat/projection + 공통 decoder이며 원 patch-expanding decoder와 다름 |
| P03 MicroNet | 공개 목록40CNN의 ImageNet encoder 비교; R101 Micro/Im→Micro v1.0 weight transfer;7decoder 포함9decoder | 모든40encoder에 모든Micro weight 조합을 붙이지 않음. 공통 encoder와 native decoder를 분리 |
| P04 MAM-UNet | decoder ECA, CBAM, ECA+CBAM, no-attention 대조 | 정확한 원 삽입 위치 미확보로 full MAM 재현 아님 |
| P05 Si phase extraction | 기존 clean/noisy photometric sampling 유지 | K-means로4상 GT를 덮어쓰지 않음. 새 구조 없음 |
| P06 Al–Si comparison | U-Net/DeepLabV3/LinkNet/FCN-R101/SegNet | library/independent multiclass 구조. 원 논문 미확인 encoder까지 동일이라고 주장하지 않음 |
| P07 EISC | 세 scale deep supervision, stop-gradient scale consistency | Train70 supervised 변형. EMA teacher/unlabeled regime 미포함 |
| P08 FreqFusion | content-adaptive no-offset / bounded learned-offset fusion | pure PyTorch 파생. 원 CARAFE/mmcv resampling 전체와 다름 |
| P09 OCR | B/R101 또는 HRNet-W48 + class prototype gather/pixel-region context/aux CE | 파생 OCR. 원 HRNet OCR 전체 neck 및 학습 recipe 미재현 |
| P10 U²-Net | full U2NET / U2NETP, fused-only / six side-head deep supervision; RSU/common decoder 및 deep | Apache2 공식 RSU snapshot. 원 binary sigmoid를4상 raw logits로 변경, R022 loss 및0.2평균 side loss 사용 |
| P11 DconnNet | B8방향×4클래스 auxiliary head/BCE | connectivity 파생. SDE/IFD/RCA/bilateral voting/SDL 전체 미포함 |
| P12 DSCNet | decoder fine feature5-point axial cumulative offsets/multidirectional fusion | 독립 snake 변형, topology/PH loss 미포함. 독립 미세 입자의 연결을 강제하지 않음 |
| P13 SegFormer cross-attention | native SegFormer B0/B2/B5; MiT common decoder; B2/B5 adjacent reduced-KV cross-attention | native SegFormer는 SMP. cross attention은 메모리 제한을 둔 파생이며 원 attention식/neck 전체와 다름 |
| P14 MFF-UMamba | learned skip gate/scale weight, deep-head supervision | weighted-skip 아이디어만 반영. VMamba SS2D/SFRB/DFAB 전체 미포함 |
| P15 PF-DiffSeg | 논문 downstream R50+Unet/Unet++/LinkNet/MA-Net의 real-only 대조 | 조건부 diffusion image-mask 공동 생성기 및 SR 미포함. 생성 pair의 의미 정합 검증이 필요 |
| P16 MAUNet/SASAPD | low/high-frequency detail + channel/spatial gate | MAUNet 아이디어 변형. SASAPD는 detector/검출 라벨·평가가 필요해4상 semantic suite에서 제외 |
| P17 MIMU-Net | recurrent feature/spatial/channel gate; FPN 별도 decoder | 일부 recurrence/scSE/pyramid 비교. 원 recurrent encoder+FPN 전체 MIMU 재현 아님 |
| 추가 ConvNeXt V1/V2 | V1 Tiny/Small, V2 Tiny/Base + 공통 decoder | timm tag 고정. V2 FCMAE 공개 finetuned checkpoint |
| 추가 MobileViT/HRNet | MobileViT-S, HRNet-W18/W48 + 공통 decoder | 공통 adapter를 쓴 encoder 비교 |

Swin-Unet, TransDeepLab, HiFormer, Mask2Former는 조사표의 비교 대상으로 남아 있지만 **이번 실행 목록에 없다**. 원 구현의 decoder, window-size/positional embedding 변환, pretrained 연결 또는 matching/native loss까지 검증하지 않은 placeholder를 넣지 않았다. VMamba도 CUDA selective-scan 구현과 정확한 upstream version이 확정되지 않아 미포함이다. CNN+Swin/common decoder를 Swin-Unet/HiFormer라고 부르지 않는다. MobileViT/common decoder는 원 논문의 size 미확인 MobileViT-U-Net과 동일하다고 주장하지 않는다.

따라서 이 ZIP은 현재 검증 가능한 넓은 구조 비교와 모듈 가설을 실행하는 패키지다. 위 미포함 전체 구조를 실행했다고 결과표에 기록하면 안 된다. 학습 후 우수 구성에 모듈을 조합하는 탐색은 이번 one-change catalogue와 별도의 paired B 계획으로 진행한다.
