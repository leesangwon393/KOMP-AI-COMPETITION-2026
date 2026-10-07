> 2026-10-07 18:53 사용자 전달 보고서 원문. 아래 계획은 당시 제안이며, 최신 구현·실행 방법은 ../../experiments/transformer_hybrid_lab_20261007/README_KO.md를 따른다. 서버 원자료 전체32회는 미확보 상태다.

# KoMaP baseline 결론 및 Transformer 하이브리드 실험 제안

작성 기준: 2026-10-07 18:53 KST

## 1. 결론

- 기존 층별·공간 attention suite는 `32/32` 모두 완료됐다.
- 지금까지 검증한 recurrent channel/spatial gate 변형 중에서는 기존 baseline `B`가 가장 높다.
- `CS_CBAM`은 3-seed 평균이 baseline과 사실상 동률이지만, 평균 향상은 관측되지 않았다.
- 따라서 기존 decoder gate를 더 복잡하게 만드는 탐색은 우선 중단하는 것이 합리적이다.
- 다음 가설로는 **ImageNet 사전학습 Transformer를 ResNet101과 병렬로 연결하고, 공간 feature map을 채널 방향으로 융합하는 하이브리드 encoder**가 적절하다.
- 첫 실험에서는 순수 Transformer encoder 교체보다 기존 CNN의 얇은 경계 표현을 보존하는 `ResNet101 + Swin-T` 구성을 권장한다.

## 2. 기존 suite 최종 결과

평가 기준은 Single 검증으로 선택한 동일 checkpoint의 D4 8-view TTA mIoU다. 아래 차이는 같은 seed의 B와 짝지어 계산한 percentage point 차이다.

| 구조 | Seed 수 | D4 mIoU 평균 | B 대비 평균 Δ | Seed별 Δ 표준편차 | 결론 |
|---|---:|---:|---:|---:|---|
| B | 3 | **79.8202%** | 기준 | 0.0000%p | 최종 1위 |
| CS_CBAM | 3 | 79.8190% | -0.0012%p | 0.1482%p | 사실상 동률, 향상 아님 |
| C_ALL | 3 | 79.7879% | -0.0323%p | 0.0576%p | 향상 없음 |
| M_UNSHARED | 3 | 79.7760% | -0.0443%p | 0.0660%p | 향상 없음 |
| C0 | 3 | 79.7485% | -0.0717%p | 0.1232%p | 향상 없음 |
| C3 | 3 | 79.7224% | -0.0978%p | 0.1725%p | 향상 없음 |
| S_CBAM | 3 | 79.5659% | -0.2543%p | 0.3505%p | 저하 |

`CS_CBAM`의 seed별 D4 차이는 다음과 같다.

| Seed | B D4 | CS_CBAM D4 | Δ |
|---:|---:|---:|---:|
| 42 | 80.0013% | 79.8545% | -0.1467%p |
| 43 | 79.6690% | 79.8186% | +0.1496%p |
| 44 | 79.7905% | 79.7840% | -0.0064%p |

seed 43에서는 상승했지만 seed 42와 44에서 반복되지 않았다. seed 44 이미지 bootstrap 95% 구간도 `[-0.4060, +0.3984]%p`로 0을 포함한다. 따라서 `CS_CBAM`을 baseline보다 우수하다고 주장할 근거는 없다.

## 3. 결과의 의미

현재 결과가 의미하는 것은 “attention이 항상 불필요하다”가 아니다.

- 기존 ResNet101-scSE baseline이 이미 강하다.
- decoder 내부에 작은 gate를 추가한 효과가 seed 변동보다 작았다.
- Train70/Valid20 규모에서는 추가 branch가 유효한 표현을 학습하기보다 과적합하거나 baseline 표현을 미세하게 흔들 가능성이 있다.
- 장거리 문맥을 encoder 단계에서 모델링하는 Transformer 가설은 아직 시험하지 않았다.

따라서 다음 실험은 decoder gate 추가가 아니라 encoder 표현 자체를 확장하는 독립적인 가설로 취급한다.

## 4. 현재 baseline 구조와 데이터 조건

### Baseline B

- 입력: grayscale을 RGB 3채널로 반복
- encoder: ImageNet V2 사전학습 ResNet101
- encoder 출력: `1/2, 1/4, 1/8, 1/16, 1/32`
- decoder: U-Net형 skip decoder
- 각 decoder ConvBlock: scSE
- 파라미터: `51,611,179`
- peak CUDA memory: 약 `4.49GB`
- 학습 patch: context `448×448`, 중앙 `224×224` 감독

### 데이터와 평가

- Train: 70개 이미지
- Valid: 20개 이미지
- epoch당 420개 patch
- 150 epoch, batch 4, FP32
- Single checkpoint 선택 후 Single/D4 평가
- Test는 구조 선택에 사용하지 않음

독립 이미지 수가 많지 않으므로 Transformer를 scratch로 학습하지 않고 반드시 사전학습 가중치를 사용한다.

## 5. 권장 Transformer

첫 후보는 `torchvision.models.swin_t`와 `Swin_T_Weights.IMAGENET1K_V1`이다.

선정 이유:

- 현재 환경의 torchvision에 이미 포함되어 추가 프레임워크 설치가 필요 없다.
- 계층형 feature를 `1/4, 1/8, 1/16, 1/32` 해상도로 제공한다.
- window attention으로 고해상도 dense prediction의 계산량을 제한한다.
- 기존 ResNet feature와 해상도별 융합이 쉽다.
- 448 입력은 32로 나누어져 최종 feature가 `14×14`로 정렬된다.

후속 후보는 MiT/SegFormer encoder다. MiT는 segmentation용 다중 스케일 feature가 장점이지만, 첫 실험은 현재 환경에서 직접 사용할 수 있는 Swin-T로 제한해 구현 변수와 디스크 사용량을 줄인다.

## 6. 권장 구조

### 전체 흐름

```text
grayscale input
      │
      └─ RGB 3채널 반복
             │
             ├─ ResNet101 ── R1, R2, R3, R4 ── 로컬 경계·질감
             │
             └─ Swin-T    ── T1, T2, T3, T4 ── 장거리·형태 문맥
                                      │
                     channel projection + concat
                                      │
                         zero-init residual fusion
                                      │
                           기존 scSE U-Net decoder
                                      │
                              4-class segmentation
```

Transformer에서 pooled CLS/global vector를 꺼내 모든 픽셀에 반복해서 붙이지 않는다. 반드시 `H×W` 공간 token map을 feature map으로 변환해 사용한다. 그래야 얇은 Eutectic 구조의 위치 정보가 유지된다.

### HT_BOT: bottleneck 단일 융합

첫 번째 권장 구조다.

입력 `448×448` 기준:

- ResNet `R4`: `2048×14×14`
- Swin `T4`: `768×14×14`
- `R4` projection: 2048 → 512 channels
- `T4` projection: 768 → 256 channels
- concat: 768 channels
- fusion bottleneck: 768 → 512
- 최종 projection: 512 → 2048, zero initialization

권장 수식:

```text
r = ProjectR(R4)
t = ProjectT(T4)
delta = ZeroProject(Fuse(Concat(r, t)))
F4 = R4 + 0.1 × delta
```

`ZeroProject`를 0으로 초기화하면 학습 시작 시 segmentation 출력이 baseline B와 동일하다. 이후 학습으로 Transformer 정보가 필요할 때만 residual이 활성화된다.

### HT_PYR: 다중 스케일 융합

두 번째 비교 구조다.

- `1/8`: ResNet R2 + Swin T2
- `1/16`: ResNet R3 + Swin T3
- `1/32`: ResNet R4 + Swin T4
- 각 위치에서 projection, concat, zero-init residual fusion 사용
- `1/4` 융합은 첫 실험에서 제외

저해상도 문맥만 쓰는 `HT_BOT`과 중간 해상도 문맥까지 쓰는 `HT_PYR`를 비교하면 Transformer 효과가 전역 문맥에서 오는지, 다중 스케일 skip 보강에서 오는지 구분할 수 있다.

### 첫 단계에서 제외할 구조

- vanilla ViT의 CLS token만 decoder 채널에 반복 연결
- 사전학습 없는 Transformer
- ResNet을 제거한 Transformer-only encoder
- Transformer와 새로운 decoder를 동시에 교체하는 실험
- 기존 attention gate까지 동시에 추가하는 복합 실험

이 구조들은 실패했을 때 원인을 분리하기 어렵다. Transformer-only encoder는 하이브리드 결과가 긍정적일 때 후속 실험으로 둔다.

## 7. 실험 목록

기존 B 3-seed 결과는 재사용한다.

| ID | Encoder/Fusion | Seed | 신규 학습 수 |
|---|---|---|---:|
| B | ResNet101 baseline | 42/43/44 | 0, 기존 결과 재사용 |
| HT_BOT | ResNet101 + Swin-T, 1/32 융합 | 42/43/44 | 3 |
| HT_PYR | ResNet101 + Swin-T, 1/8·1/16·1/32 융합 | 42/43/44 | 3 |

총 신규 학습은 6회다. 현재 실험에서 seed 42 단독 순위와 3-seed 평균이 여러 번 달라졌으므로 seed 42 결과만 보고 후보를 제거하지 않는다.

## 8. 학습 조건

비교 가능성을 위해 다음 조건은 baseline과 동일하게 유지한다.

- 동일 Train70/Valid20 split과 manifest
- context 448, target 224, stride 112
- epoch당 420개 patch
- batch 4, FP32, 150 epoch
- 동일 rare-class sampling 및 증강
- CE 0.5 + Dice 0.25 + Lovasz 0.25
- 동일 class weight
- 동일 seed 42/43/44
- Single 기준 checkpoint 선택
- 동일 checkpoint의 D4 8-view TTA 평가

Transformer 안정화를 위한 optimizer group만 명시적으로 분리한다.

| 파라미터 그룹 | 초기 LR | 비고 |
|---|---:|---|
| ResNet101 | `3e-4` | baseline과 동일 |
| 기존 decoder | `3e-4` | baseline과 동일 |
| 신규 fusion | `3e-4` | scratch 학습 |
| Swin-T | `3e-5` | pretrained feature 보존 |

권장 초기 전략:

- epoch 1~5: Swin-T 동결, fusion과 기존 네트워크 학습
- epoch 6~150: Swin-T 해제, LR `3e-5`
- weight decay와 cosine schedule은 기존 설정 유지
- gradient와 fusion residual 크기를 history에 기록

이 optimizer 차이는 문서에 명시하고, Transformer 효과와 학습 recipe 효과를 구분해야 한다. 결과가 경계선이면 동일 LR 또는 동결 없는 보조 실험을 추가한다.

## 9. 구현 검증

GPU 장시간 학습 전에 다음을 통과해야 한다.

1. `HT_BOT`, `HT_PYR`의 합성 forward/backward
2. 입력 448, batch 4에서 CUDA preflight
3. Swin feature 해상도와 채널 shape 확인
4. zero-init 상태에서 HT 출력이 B 출력과 tensor 수준에서 일치하는지 확인
5. fusion branch에 gradient가 전달되는지 확인
6. grayscale RGB 반복과 ImageNet normalization 확인
7. D4 회전·반전에서 shape 및 positional 처리 확인
8. checkpoint 중단·재개 후 optimizer group과 RNG 복구 확인
9. 파라미터 수, peak VRAM, 초당 처리량 기록

## 10. 성공 판정 기준

주 평가지표는 기존과 동일한 D4 mIoU다.

Transformer 구조를 실질적 개선으로 인정하려면 다음을 함께 만족하는 것을 목표로 한다.

- 3-seed paired D4 평균이 B보다 높음
- 평균 향상 `+0.10%p` 이상
- 최소 2/3 seed에서 같은 seed B보다 높음
- seed별 Δ 표준편차가 평균 상승보다 지나치게 크지 않음
- 이미지 bootstrap 95% 구간 보고
- Eutectic Si IoU와 thin skeleton recall이 악화되지 않음
- Al3Ni↔Eutectic confusion과 contact error가 악화되지 않음
- 파라미터, peak VRAM, Single/D4 평가시간 증가를 함께 보고

향상이 `+0.02%p` 수준이고 seed 방향이 섞이면 CS_CBAM과 마찬가지로 사실상 동률로 판단한다.

## 11. 실행 순서

1. 기존 완료 결과를 영구 저장소로 백업
2. 새 output `runs/transformer_fusion_v1` 생성
3. Swin-T pretrained checkpoint를 `/dev/shm` cache에 준비
4. 합성 CPU/CUDA architecture 검증
5. batch 4, 448 FP32 메모리 preflight
6. `HT_BOT` seed 42/43/44 실행
7. `HT_PYR` seed 42/43/44 실행
8. 같은 seed B와 paired 집계
9. 두 구조 중 하나가 성공 기준을 충족하면 구조를 고정
10. 구조 고정 후에만 독립 Test를 1회 평가

## 12. 자원 및 저장공간

현재 상태:

- 루트 디스크 여유: 약 `5.7GB`
- `/dev/shm` 여유: 약 `81GB`
- 기존 32-run 산출물: 약 `31GB`
- baseline 1회 산출물: 약 `991MB`
- baseline `last.pt`: 약 `790MB`

Swin-T를 포함하면 optimizer checkpoint가 커지므로 6회에 약 `8~12GB` 이상의 추가 공간이 필요할 수 있다. 실제 값은 preflight run으로 확인한다.

현재 `/dev/shm` 용량은 실험에 충분하지만 휘발성이다. 서버 재부팅 시 기존 32개 결과와 신규 Transformer 결과가 모두 사라질 수 있으므로, 새 실험 전에 영구 저장 위치를 확보하는 것이 우선이다.

## 13. 해석상 주의사항

- B가 “모든 가능한 구조 중 최고”라는 뜻은 아니다. 이번 suite에서 시험한 gate 변형 중 최고라는 뜻이다.
- 동일 Valid20을 구조 선택에 반복 사용하므로 최종 Test 전까지 일반화 우위를 주장하지 않는다.
- ImageNet 사전학습 Transformer가 금속 미세조직에 반드시 유리하다는 보장은 없다.
- 단순 파라미터 증가 효과를 구분하기 위해 파라미터 수와 추론 시간을 함께 보고한다.
- Transformer가 전체 mIoU를 높이면서 얇은 Eutectic 구조를 흐릴 수 있으므로 class/형태 지표를 반드시 함께 본다.

## 14. 참고 자료

- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation](https://arxiv.org/abs/2102.04306)
- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)

## 15. 관련 결과 파일

- 기존 최종 결과: [`runs/layer_search/reports/RESULTS_KO.md`](runs/layer_search/reports/RESULTS_KO.md)
- seed 집계: [`runs/layer_search/reports/seed_aggregates.json`](runs/layer_search/reports/seed_aggregates.json)
- 상세 paired 비교: [`runs/layer_search/reports/comparison.csv`](runs/layer_search/reports/comparison.csv)
- 전체 suite 상태: [`runs/layer_search/suite_status.json`](runs/layer_search/suite_status.json)
- 기존 중간 브리핑: [`INTERIM_BRIEFING_KO_v3.md`](INTERIM_BRIEFING_KO_v3.md)

## 16. 현재 권고

기존 gate 탐색은 baseline 우위로 종료한다. 다음 실험은 `HT_BOT`, `HT_PYR` 두 구조만 3-seed로 실행한다. 첫 구현은 기존 코드와 dependency 호환성이 가장 좋은 `torchvision Swin-T`를 사용하고, baseline과 동일 출력으로 시작하는 zero-init residual fusion을 적용한다.
