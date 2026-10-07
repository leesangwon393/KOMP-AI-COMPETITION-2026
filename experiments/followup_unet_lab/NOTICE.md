# 구현 및 출처

B/D03의 데이터·학습·평가는 development_server_lab와 phase_discrimination_lab에서 이어왔다.
후속 모듈은 독립 작성한 PyTorch 구현이며 논문 전체 재현이 아니다.

- scSE: https://arxiv.org/abs/1808.08127
- FreqFusion 아이디어: https://arxiv.org/abs/2408.12879
- filtering 연산 확인에 사용한 저자 공개 reference: https://github.com/Linwei-Chen/FreqFusion/blob/main/FreqFusion.py
- 방향별 연결 감독 아이디어: https://arxiv.org/abs/2304.00145
- PyTorch/torchvision: https://github.com/pytorch/vision

FreqFusion reference 파일을 배포 코드에 복사하지 않았다.
X01_FREQ는 offset·iterative compressed initialization을 제외한 축소 파생 구성이다.
X02_CONN은 단순 연결 보조 head이며 DconnNet의 방향 module/voting 전체를 구현하지 않는다.
학습 데이터와 모델 checkpoint는 Git 저장소에 포함하지 않는다.
