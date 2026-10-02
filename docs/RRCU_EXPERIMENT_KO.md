# C13 RRCU decoder 비교 실험

2026-10-02에 실제 로컬 Apple MPS GPU에서 150 epoch 학습을 시작했다. C13이 완료되면 동일 코드의 B 기준 모델을 순차 실행한다. 이 문서의 실행 상태는 시작 시점 기록이며, 완료 성능은 아직 없다.

## 설계

ResNet101 encoder와 U-Net의 가장 깊은 decoder block(3072→256, Context448에서 28×28)을 비교한다. 나머지 decoder와 scSE, 입력/감독448/224, sampling, loss, seed42, batch4, FP32, 150 epoch는 공통이다.

C13의 입력을 1×1 Conv-BN-ReLU로 투영해 r을 만들고, h₀=0에서 hₜ=ReLU(BN(Conv₃×₃(r+hₜ₋₁)))를 t=1,2에 적용한다. 두 단계는 Conv와 BN 가중치를 공유한다. 출력은 scSE(r+h₂)다. 기존 scSE의 초기 가중치를 유지한다. [R2U-Net](https://arxiv.org/abs/1802.06955)의 recurrent/residual 아이디어를 적용한 파생 블록이며 논문 전체 재현은 아니다.

## 실행 전 확인

- 기준 B와 C13에서 공통 state tensor 706개가 정확히 같고, 모델 생성 뒤 CPU RNG가 일치했다. 첫 4개 crop도 동일했다.
- 실제 Train 데이터 batch4(입력4×1×448×448, target4×224×224)에서 MPS loss, backward, AdamW step과 eval 출력448을 확인했다. 이 단일 step은 본 학습 성능이 아닌 실행 확인이다.
- 데이터는 Train70/Valid20이며 공개 데이터/체크포인트 업로드는 하지 않는다.
- B 51,611,179개 대비 C13은 45,319,723개 parameter다. 이 비교는 반복 연산뿐 아니라 1×1 투영·residual과 parameter 감소를 포함한 블록 교체의 효과다. 반복 연산만의 인과 효과로 해석하지 않는다.

## 실행 기록과 해석

실행 폴더는 `experiments/server_architecture_lab/runs/rrcu_mps_20261002/`다. `launch.json`과 `source_snapshot/`에 실제 실행 설정·소스를 보존한다. `C13.training.log`, `C13/status.json`, `C13/history.json`, `C13/best.pt`, `C13/last.pt`에 학습 상태와 결과를 저장한다. B는 순차 실행으로 GPU를 공유하지 않는다. launcher는 macOS의 caffeinate로 작업 중 유휴 절전을 막는다.

로컬 PyTorch2.8.0/MPS 결과는 기존 PyTorch2.7.1/CUDA 서버의 R022 기록과 환경이 다르다. B와 C13의 동일 코드·환경 비교를 우선한다. 각 run은 single-view Valid로 checkpoint를 고르고, 완료 후 선택된 best의 D4/strict 점수와 클래스별·이미지별 변화를 평가한다.
