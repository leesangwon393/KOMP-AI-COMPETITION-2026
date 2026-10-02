# KoMaP MIMU-Net local experiment

추가된 논문 `1-s2.0-S0927025624004208-main.pdf`의 MIMU-Net을 KoMaP 데이터에서 학습한다.
SAM2/SAMM 모델이 아니다. 사전학습 가중치를 사용하지 않는 무작위 초기화 학습이다.

## 근거와 재현 범위

- Majumdar et al., *Metallographic image segmentation using feature pyramid based recurrent residual U-Net*, Computational Materials Science 244 (2024), 113199. DOI: https://doi.org/10.1016/j.commatsci.2024.113199
- 논문에 직접 연결된 저자 코드: https://github.com/sammajum706/Feature-Pyramid-Recurrent-Residual-U-Net-for-Metallographic-Image-Segmentation
- 확인한 commit: `b2d21ef39cf6b513c8f3eb6d2df2dd76d59924ac`.
- 모델은 `fpn_r2unet.py`의 **실제로 출력에 연결된 계산 그래프**를 PyTorch로 이식했다.
- 입력 1채널, 18개 RRCNN 블록, 7개 scSE, 4단계 max-pooling, 14/28/56/112/224 채널, 최종 210채널 피라미드 결합을 유지한다. 5클래스 출력을 KoMaP의 4클래스로 변경했다.
- 학습 가능한 파라미터는 **6,574,713개**이다. 논문 표 7의 6.46M과 차이가 있다. 아래의 공개 코드/논문 차이 때문에 이 구현을 논문 수치와 완전히 동일한 재현이라고 주장하지 않는다.

### 공개 코드와 본문의 차이 — 숨기지 않고 유지/기록한 사항

1. **Recurrent block:** 저자 코드의 반복문은 이전 출력으로 상태를 갱신하지 않고 매번 `x+x`에 새 convolution을 적용한다. 첫 반복의 출력은 연결되지 않아 Keras 모델에서 제거된다. 이식 코드도 살아 있는 두 convolution 및 `x+x`를 유지한다. 통상적인 가중치 공유 R2U-Net으로 임의 교체하지 않았다.
2. **scSE:** 저자 코드의 sSE는 1채널이 아닌 C채널 gate이고, cSE와 sSE 출력을 **합산**한다. 이 동작을 유지한다. 논문 설명의 곱셈/표준 scSE와 차이가 있다.
3. **손실함수:** 본문 식 (4)는 `Focal + 2 × Dice`, 저자 `training.py`는 `Dice + 2 × Focal`이다. 기본 실험은 **본문 식 (4)**를 따른다. MetalDAM 전용 5개 class weight는 KoMaP에 의미가 없어 균등 가중치를 사용한다. focal alpha=.25, gamma=2는 저자가 사용한 라이브러리 기본값을 따른 가정이며 논문 표에 명시된 값은 아니다.
4. **증강:** 원본 + 상하/좌우/양축 flip + 시계 방향 90°/180° rotation = 원본당 6개. 저자 코드에서 양축 flip과 180°는 중복인데, 초기 재현 실험은 이 중복까지 유지한다. 원본 파일을 수정하지 않고 메모리에서 변환한다.
5. **프레임워크:** TensorFlow/Keras에서 PyTorch로 이식했다. BN epsilon=.001, momentum=.01(Keras .99 대응), Adam epsilon=1e-7을 사용한다. 초기화 난수와 BN running variance 처리 등의 차이로 프레임워크 간 bitwise 동일성은 보장하지 않는다. Keras와 수치 대조 실행은 하지 않았다.

## 데이터/학습 설정

| 항목 | 설정 |
|---|---|
| Train / Valid / Test | 제공된 70 / 20 / 10 원본 split 유지 |
| 입력 | grayscale, bilinear resize 224×224, /255 |
| 정답 | RGB 팔레트 → ID, nearest resize |
| Train 증강 | 원본당 6개, 총 420 samples/epoch |
| Batch | 4 (저자 main.py 기본값) |
| Optimizer | Adam, lr=0.001, weight_decay=0 |
| Epochs | 150 (논문 표 6), early stopping 없음 |
| Seed | 42 |
| Device | Apple MPS / CUDA / CPU 자동 선택 지원 |
| 최적 모델 선택 | 원본 해상도 검증 이미지별 mIoU의 평균 |

논문의 MetalDAM 6-fold 검증을 그대로 사용하지 않고 대회 공식 split을 사용한다.
증강은 train에만 적용한다. 검증 20장은 checkpoint 선택용이므로 이 점수를 독립적인 test 성능이라고 표현하면 안 된다. Test에는 정답이 없어 점수를 계산하지 않는다.

| ID | 클래스 | RGB |
|---|---|---|
| 0 | Primary Si | 76,178,76 |
| 1 | Al3Ni | 25,76,153 |
| 2 | Eutectic Si | 204,204,204 |
| 3 | Al | 153,127,76 |

### 평가지표

원본 이미지 → 224×224 모델 출력 → softmax 확률을 원본 해상도로 bilinear 복원 → argmax.
이미지마다 4개 클래스의 IoU를 평균한 다음 이미지별 값을 평균한다.
224×224 정답에 대한 점수도 따로 저장하여 축소 평가와 혼동하지 않는다.

정답/예측 모두에 없는 클래스는 공식 세부 규칙이 확인되지 않았다. 기본 `--absent-policy one`은 IoU=1로 처리한다. 0 처리와 해당 클래스 제외 점수도 함께 기록한다. 따라서 이 점수는 **명시된 absent-class 가정에 따른 검증 점수**다. Al도 평가 클래스에 포함한다.

## 실행

작업 폴더에서 실행한다. 현재 로컬 Python에 torch/numpy/Pillow가 설치되어 있어 환경을 변경하지 않고 실행했다.

```bash
python3 -m unittest discover -s tests -v
python3 -u train.py --device mps --epochs 150 --output runs/mimu_paper
```

다른 환경에서는 가상환경에 `pip install -r requirements.txt`로 의존성을 설치한다.
macOS 샌드박스에서는 MPS가 감지되지 않을 수 있어 GPU 학습은 일반 터미널 또는 승인된 샌드박스 외부 실행이 필요하다.

저장 지점에서 이어 학습:

```bash
python3 -u train.py --device mps --epochs 150 --output runs/mimu_paper150 --resume runs/mimu_paper/last.pt
```

터미널과 분리해 로그를 남기며 실행:

```bash
python3 run_local.py start --output runs/mimu_paper150 --resume runs/mimu_paper/last.pt --epochs 150
python3 run_local.py status --output runs/mimu_paper150
```

`--epochs`는 이어 학습할 추가 횟수가 아니라 **전체 목표 epoch**다. checkpoint에 모델·optimizer·학습 RNG·이력·입력 파일 hash를 함께 저장한다. 기존 결과가 있으면 명시적인 resume 없이 덮어쓰지 않는다.

저장 모델 평가 및 원본 크기 mask 생성:

```bash
python3 -u train.py --device mps --evaluate runs/mimu_paper150/best.pt --output runs/mimu_evaluation
```

## 결과 파일

- `config.json`: 적용 설정과 파라미터 수
- `data_manifest.json`: 입력 파일 SHA-256, 데이터 변경 검사
- `history.json`, `history.csv`: epoch별 loss, 원본/축소 mIoU, 클래스별 IoU
- `last.pt`: 재개용 최신 checkpoint
- `best.pt`: 검증 mIoU 최고 checkpoint (optimizer 포함)
- `status.json`: 진행 상태, 완료 epoch, 최고 점수
- `training.log`, `pid.txt`: 분리 실행 시 로그와 PID
- `validation_metrics.json`: 종료 후 최고 checkpoint의 검증 상세
- `validation_preview.jpg`: 검증 하위 3장/상위 3장의 입력·정답·예측 비교
- `valid_masks/`, `test_masks/`: 종료 후 최고 checkpoint의 원본 크기 RGB 마스크 20/10장

`runs/mimu_paper`는 초기 2-epoch 동작 확인 결과이며, 성능 결론에 사용할 학습 완료 모델이 아니다.
`runs/mimu_paper150`는 해당 checkpoint에서 이어 학습하는 150-epoch 본 실험이다.
최종 파일은 `status.json`의 `state=completed`를 확인한 뒤 사용한다.
