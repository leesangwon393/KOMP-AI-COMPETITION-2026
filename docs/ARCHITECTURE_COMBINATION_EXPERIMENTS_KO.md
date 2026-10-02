# 금속 조직 분할: 아키텍처 결합 실험 후보

작성일: 2026-10-01. **실험 제안만 작성했으며 새 학습·실행 코드·실험 큐를 생성하지 않았다.** 아래 C 번호는 계획 ID이며 실제 실행 번호가 아니다.

모듈의 기원은 [노벨티 분석](METALLOGRAPHIC_NOVELTY_BREAKDOWN_KO.md)에 정리했다. 아래 조합은 우리 과제를 위한 파생 설계다. 논문 전체 재현이나 새로운 학술적 노벨티가 검증된 모델을 뜻하지 않는다. 기대 효과·부담은 가설이며 실행 측정값이 아니다.

## 기준 모델과 비교 조건

팀원 최고 결과는 **R022 ResNet101, 0.798969**다. 원본 decoder·학습·평가 설정은 아직 미확인이다.

이 문서의 제안 기준 **B = ResNet101 + 현재 로컬 U-Net decoder + scSE + 최종 DySample**이다. 이는 현재 ResNet34 계열에서 encoder를 바꾼 기준 설계이며, **B가 R022와 같은 모델이거나 79.8969%를 낸 모델이라고 가정하지 않는다.** 사용자가 후보를 선택하면 먼저 R022 설정과 대조해 기준을 확정한다. 기준 실험도 지금 실행하지 않는다.

원칙은 같은 기준 모델에서 모듈 추가·교체 효과를 비교하는 것이다. 200 epoch가 기본 학습 길이 후보지만, 입력 크기·sampling·loss·scheduler·split·checkpoint 선택·추론 타일과 D4는 확정된 기준 조건을 따른다. 구조 변경 중 sampling이나 loss까지 바꾸지 않는다. 입력 정규화가 다른 공개 가중치는 필요한 변경을 기록한다.

## 후보 12개

| ID | 조합 | 겨냥하는 문제 | 부담 추정 | 선택 시 비교 대상 |
| --- | --- | --- | --- | --- |
| **C01** | B + 얕은 고주파 residual skip | 가는 Eutectic Si와 경계가 사라짐 | 낮음~중간 | B |
| **C02** | B + 낮은 해상도 stage 간 cross-attention | 주변 문맥과 작은 조직의 상 구분 | 중간 | B, 동일 위치 concat fusion 대조 |
| **C03** | B + 고주파 skip + cross-attention | 경계 보존과 문맥 구분을 동시에 개선 | 중간 | B, C01, C02 |
| **C04** | ResNet101 U-Net + adaptive multi-dilation fusion + DySample | 조직 크기에 따라 예측이 흔들림 | 중간 | B, 고정 fusion 대조 |
| **C05** | C04 + 고주파 skip | 크기 변화 대응과 가는 조직 보존 | 중간 | B, C01, C04 |
| **C06** | B의 일부 skip을 가중 shallow/deep fusion으로 교체 + 고주파 보조 경로 | skip에서 경계와 의미 정보의 균형 | 낮음~중간 | B, 가중 fusion 단독, C01 |
| **C07** | ResNet101 + UPerNet decoder + 최종 DySample | 다중 스케일 문맥 융합과 출력 복원 | 중간 | B, 일반 업샘플링 UPerNet |
| **C08** | ResNet101 + U-Net++ + scSE + 최종 DySample | encoder/decoder의 특징 차이와 정보 손실 | 중간~높음 | B, attention·upsampling을 맞춘 U-Net++ 대조 |
| **C09** | ResNet101 + Swin-T 병렬 encoder + U-Net fusion decoder + DySample | 국소 텍스처와 넓은 문맥의 상호 보완 | 높음 | B, 같은 hybrid의 단순 concat 대조 |
| **C10** | B + bottleneck VSS/Mamba block + 고주파 skip | 긴 문맥과 얇은 구조의 상호 보완 | 높음·환경 의존 | B, VSS 단독, C01 |
| **C11** | MicroNet 초기화 ResNet101 + U-Net + 고주파 skip + DySample | 재료 텍스처 표현과 경계 보존 | 분할 단계 낮음~중간 | B, C01, MicroNet 초기화 단독 |
| **C12** | multi-stage SSL ResNet101 + U-Net + 고주파 skip + cross-attention + DySample | 도메인 표현·경계·문맥의 결합 | 높음·사전학습 포함 | B, C03, SSL 초기화 단독 |

부담은 B에 대한 설계상 상대 추정이다. GPU/MPS 실제 시간·메모리 측정은 하지 않았다.

## 모듈별 최소 설계

### 고주파 skip — C01/C03/C05/C06/C10/C11/C12

MAUNet의 frequency-aware skip 아이디어에서 출발한다. 최초 적용 위치는 U-Net의 얕은 1~2개 skip으로 한정한다. 특징 E에 대해 `H = E − upsample(lowpass(E))`라는 residual을 만들고 기존 skip과 H를 함께 사용한다. H만 전달해 저주파 의미 정보를 모두 버리는 설계는 첫 후보에서 피한다.

이 수식은 **우리 제안**이며 MAUNet의 정확한 필터 재현으로 주장하지 않는다. lowpass는 가벼운 평균 필터를 첫 선택으로 두되 kernel·추가 convolution은 후보를 실제 준비할 때 고정한다. 고주파가 촬영 noise도 강조할 수 있어 thin recall과 precision을 함께 본다.

### Cross-attention — C02/C03/C12

ResNet layer3/layer4, 즉 낮은 해상도의 인접 stage부터 사용한다. 채널 projection 후 높은 해상도 stage를 Q, 더 깊은 stage를 K/V로 두고 attention 출력을 residual로 더한다. 공간 좌표를 유지하며 U-Net decoder로 전달한다.

원 SegFormer 논문의 모든 stage를 그대로 복제하는 실험은 아니다. 가능한 경우 K/V의 낮은 공간 해상도를 유지해 비용을 줄인다. 같은 위치의 단순 concat+projection도 비교해, attention의 이점과 추가 용량의 효과를 구분한다.

### Adaptive multi-dilation fusion — C04/C05

decoder 중간 한 stage에서 dilation 1/2/3의 병렬 3×3 branch를 두고, 특징에서 얻은 softmax branch weight로 결합한다. residual 경로로 원 특징을 보존한다. 해당 위치의 기존 scSE는 교체해 비슷한 attention을 중첩시키지 않는다. 나머지 위치의 scSE는 기준을 유지한다.

이는 CMAA·dilated U-Net의 원리에 근거한 **단순화한 파생 설계**다. CMAA의 APAM/LSTM 전체 재현은 아니다. 고정 가중 fusion과 비교해 적응적 스케일 선택의 효과를 확인한다.

### 가중 skip fusion — C06

shallow skip과 upsampled deep 특징의 채널 수를 맞추고 학습 가중으로 결합한다. W-PSB의 가중 shallow/deep fusion을 참고하되 정확한 논문 모듈 이식과 구분한다. 고주파 H는 별도 residual 보조 경로로 넣는다.

가중치는 첫 설계에서 학습 scalar로 시작하며, 입력별 attention map까지 동시에 추가하지 않는다. 비교할 때 fusion 출력 폭을 맞춰 concat 대비 용량 차이를 줄인다.

### Pyramid/nested decoder — C07/C08

C07은 encoder를 유지하고 UPerNet의 PPM+FPN 융합을 사용한다. DySample은 최종 출력 복원에 한정하며, 같은 decoder의 일반 업샘플링과 비교한다. 표준 UPerNet을 바꾼 파생 모델이다.

C08은 nested dense skip을 가진 U-Net++로 decoder를 바꾼다. scSE와 DySample의 사용 위치를 먼저 정해 대조군과 맞춘다. 모든 nested upsampling을 한꺼번에 바꾸지 않고 최종 복원부터 검토한다. nested skip의 효과와 추가 decoder 용량의 효과는 완전히 같지 않다.

### CNN + Transformer/Mamba — C09/C10

C09는 CS-UNet의 병렬 encoder 원리를 차용한다. CNN/Swin stage의 공간·채널 정렬 후 U-Net decoder에 fusion한다. 원 CS-UNet의 Swin decoder까지 그대로 사용하는 실험과 다르다. 단순 concat을 먼저 대조군으로 두고 gated fusion 여부를 분리한다.

C10은 전체 MFF-UMamba 교체 대신 bottleneck에 소수 VSS block을 넣는다. Mamba selective-scan의 MPS 지원 또는 가능한 실행 장치를 먼저 확인해야 하므로 즉시 실행 가능한 후보로 확정하지 않는다. 최종 성능만으로 Mamba와 고주파 skip 중 어떤 요소가 기여했는지 판단하지 않는다.

### 도메인 사전학습 + 분할 구조 — C11/C12

C11은 공식 MicroNet v1.0 ResNet101 가중치 호환성을 확인한 뒤 초기화한다. decoder는 동일하고 고주파 skip만 추가한다. ImageNet→MicroNet과 MicroNet-only 중 어떤 가중치를 사용하는지 고정한다. 공개 가중치의 추가 사전학습 비용과 분할 학습 비용은 구분한다.

C12는 Train 영상만으로 ResNet101 stage1~4 특징을 GAP→gate→concat→projection하여 SSL을 수행한 뒤 encoder만 전이한다. gate/projection은 분할 모델에 남기지 않는다. 이는 현재 ResNet34 단일-stage SSL과 다른 계획이며 현재 실험 결과를 그대로 대체하지 않는다. C03과 SSL 초기화 단독이 유력할 때의 후속 결합 후보로 둔다.

## 우선 검토할 조합과 후속 결합

| 사용자 선택 목적 | 우선 검토 후보 | 판단 이유 |
| --- | --- | --- |
| 작은 변경으로 경계 문제를 겨냥 | C01 | 기존 encoder·decoder를 대부분 유지 |
| 크기와 문맥에 따른 혼동을 겨냥 | C02 또는 C04 | 문맥 참조와 스케일 선택을 별도 가설로 비교 |
| 성능을 노리는 첫 복합 조합 | **C03** | 경계 정보 전달과 문맥 참조의 역할이 다름 |
| backbone 유지하며 decoder를 바꾸기 | C07 | 팀원 최고 ResNet101을 활용하는 대안 |
| 사전학습과 구조를 함께 활용 | C11 | 같은 encoder에서 초기화와 경계 경로를 결합 |
| 큰 구조 변경의 가능성을 확인 | C09 | CNN+Transformer의 상호 보완 검토 |
| 성공 요소를 모은 후속 후보 | C05/C12 | 구성 요소별 개선을 확인한 뒤 결합 |

C03은 `B / +고주파 / +cross-attention / +둘 다` 네 모델로 비교하면 각각의 효과와 결합 효과를 볼 수 있다. 나머지 복합 후보도 표의 단독 모듈 대조가 필요하다. 표의 12개를 전부 실행하자는 의미는 아니다.

## 성능을 어떻게 판단할 것인가

- 전체 mIoU와 4개 클래스별 IoU를 함께 본다. Eutectic Si IoU·precision·recall·가는 조직 recall을 핵심 진단으로 둔다.
- 같은 이미지의 전후 차이와 혼동행렬로 희귀 상 개선이 다른 상의 오검출 증가와 맞바뀌는지 확인한다.
- latency·메모리·실제 학습 시간을 기록한다. 200 epoch여도 구조별 계산량은 같지 않다.
- Valid checkpoint 선택과 D4 규칙을 고정하고, Test는 후보 선택에 사용하지 않는다.
- 작은 단일 seed 개선을 확정적 우열로 표현하지 않는다. 선택된 유력 후보에만 반복 seed 검증을 고려한다.

구체적인 개선폭과 성공 임계치는 현재 근거로 예측하지 않는다.

## 첫 조합 목록에서 제외한 방향

SAMM은 4상 출력·fusion·loss의 세부 구현이 더 필요하다. MatSAM/SAM-I-Am은 주로 prompt·mask 후처리이며, Cellpose-SAM은 instance 라벨·flow 출력이 필요해 현재 semantic segmentation의 작은 변경으로 비교하기 어렵다. 따라서 위 목록에서는 기존 4상 분할과 바로 연결되는 조합을 우선했다.

## 원리의 출처

- [MAUNet: frequency-aware skip](https://pmc.ncbi.nlm.nih.gov/articles/PMC7796035/)
- [스케일 간 cross-attention](https://jicrs.icros.org/_PR/view/?aidx=47232&bidx=4265)
- [CMAA: 적응 다중 스케일 attention](https://doi.org/10.1016/j.eswa.2025.126925)
- [MFF-UMamba: VSS·가중 fusion·보조 학습](https://doi.org/10.1093/cdm/wqag018)
- [UPerNet: PPM+FPN](https://arxiv.org/html/1807.10221v1)
- [U-Net++: nested skip](https://arxiv.org/abs/1807.10165)
- [CS-UNet: 병렬 CNN/Swin](https://arxiv.org/html/2308.13917)
- [MicroNet 공개 encoder](https://github.com/nasa/pretrained-microscopy-models)
- [MatSSL 다중 stage 사전학습](https://arxiv.org/html/2507.18184v1)
