# 검증 범위

이 폴더의 loss/숫자는 합성 입력 검사 결과이며 KoMaP 학습 성능이 아니다.

- reference_baseline.json: 동일 로컬 CPU/seed42에서 기존 R022 harness와 새 B의 state tensor718개와 inference logits가 정확히 일치.
- synthetic_cpu.json: B/D01/D02/D03/D04 각각의 forward/backward/AdamW2step, 실제 Data Train/Valid 쌍·palette·중복 검사. 64×64 합성 입력 사용.
- fullsize_cpu.json: 결합 D04의 실제 입력448×448, batch4, 중앙224 감독, 대조학습/BEM을 함께 켠 backward/AdamW2step 검사.
- unit_tests.json: 목적함수, 경계 wraparound 방지, 잘못 섞인 coarse GT 제외, 초기 가중치/RNG 보존, checkpoint/RNG resume, tiled inference/PNG export/comparison 검증.

로컬 PyTorch2.8.0 CPU에서 수행했다. fullsize 기록의 전체 code hash는 이후 suite PID/lock 표시를 정리하기 전 시점이고,
검사한 model/objective 구현은 최종 코드와 같다. 서버의 CUDA/PyTorch2.7.1 실행·메모리 검증과 실제150 epoch 결과는 아직 없다.
서버 README의 CUDA check를 먼저 실행한다. ImageNet weights와 실제 모델 학습 checkpoint는 포함하지 않았다.
