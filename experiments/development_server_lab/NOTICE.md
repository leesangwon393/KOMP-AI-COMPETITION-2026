# 구현 및 결과 출처

이 패키지의 B/D03는 이전 phase_discrimination_lab의 데이터/loss/evaluation 및 학습 recipe를 유지한다.
R01–R03은 rrcu_scse_lab의 블록을 사용한다. 새 encoder 후보는 torchvision 구현을 사용한다.
관련 논문의 전체 구조 재현을 주장하지 않는다.

- RRCU 아이디어: R2U-Net, https://arxiv.org/abs/1802.06955
- scSE 아이디어: https://arxiv.org/abs/1808.08127
- PyTorch/torchvision: https://github.com/pytorch/vision
- 역사적 팀 보고와 이번 D계열 서버 결과: docs/의 보고서를 참조한다.

ConvNeXt 후보는 기존 R024 원본 소스가 없는 상태에서 만든 새 grayscale skip 설계다.
이 패키지는 학습된 모델 가중치 또는 역사적 서버 실행 artifact를 포함하지 않는다.
