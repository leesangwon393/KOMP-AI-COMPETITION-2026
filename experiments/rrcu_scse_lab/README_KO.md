# B + RRCU/scSE 세 가지 비교

2026-10-03. 사용자의 요청으로 B의 구조·recipe에 RRCU를 넣는 위치 세 가지를 따로 비교한다.
초기화는 ImageNet V2이며 과거 B best checkpoint로 fine-tuning하는 실험은 아니다.
관측 최고 R031의 per-tile Dice는 seed43에서 우위가 재현되지 않았으므로, 안정 기준 recipe R022를 B로 사용한다.
과거 R022 D4 mIoU0.7989694265는 참고 기록이고, 비교 상대는 같은 환경에서 새로 학습한 B다.

| ID | 변경 | 위치 |
|---|---|---|
| B | 없음: ResNet101 U-Net/scSE | 원래 decoder |
| R01 | 기존 두 convolution 블록을 압축+shared recurrent residual2step으로 교체, 기존 scSE 유지 | decoder block3, 입력448에서224×224, 출력64채널 |
| R02 | 기존 sSE logit에 작은 RRCU branch를 더한 뒤 sigmoid | 동일 block3의 공간 gate |
| R03 | GAP→작은1×1 recurrent residual branch를 cSE logit에 더한 뒤 sigmoid | 동일 block3의 채널 gate |

R01은 이전 C13의 저해상도28×28 블록과 달리 얕은224×224 블록에서 검증한다.
R01의 식은 r=Conv1×1/BN/ReLU(x), h0=0, ht=ReLU(BN(sharedConv3×3(r+h[t−1]))), t=1,2,
출력=scSE(r+h2)다. 1×1 압축·BN 공유·parameter 감소도 포함해 반복 연산만의 인과 효과로 해석하지 않는다.

R02의 branch는 hidden16, shared3×3 convolution + GroupNorm4의2step, outer residual r+h2,
1×1→1채널 출력을 쓴다. `s=sigmoid(original_s_logit+0.1×branch(x))`이며 cSE는 유지한다.
R03은 GAP 이후 hidden4로 압축하고 shared1×1 convolution/ReLU를2회 적용한다.
`c=sigmoid(original_c_logit+0.1×branch(GAP(x)))`이며 sSE는 유지한다.
GAP 뒤에는 공간이1×1이므로3×3 공간 convolution을 쓰지 않는다. R03은 채널 recurrence 파생 설계다.
두 gate branch의 출력은0으로 초기화해 B와 첫 출력이 같으며 첫 optimizer step 이후 branch를 학습한다.
모든 변형은 기존 gate 가중치를 보존하고 별도 CPU RNG로 새 모듈을 초기화한다. t=2 가중치 공유를 실제로 구현했다.

## 공통 조건 및 실행 순서

Context448/center224/stride112, reflect halo, batch4 FP32, seed42,150epochs,420draws/epoch,
Al3Ni/Eutectic-rich10% sampling50%, 약한 photometric/D4 augmentation,
AdamW3e-4/wd1e-4/cosine150, CE.5+weightedDice.25+present-classLovasz.25, classweights[1,1,2,1].
대조학습·BEM·접촉부 loss는 이 실험에서 비활성화한다. Single-view Valid best 선택 후 해당 best의D4/strict 평가.
같은 데이터·code hash·seed·환경에서 mIoU, 양쪽 IoU, 양방향 혼동률, 얇은 skeleton recall, 시간/메모리를 비교한다.

현재 Mac의 MPS GPU에서 R01→R02→R03→B를 한 번에 하나씩 실행하도록 준비했다.
source snapshot을 보존해 학습 도중 원본 코드 변경이 진행 중 run에 반영되지 않는다.
새 변형을 먼저 확인하려고 B는 마지막에 실행한다. B가 완료되기 전에는 구조의 성능 우위를 판단하지 않는다.
기존 중단된 C13을 재개하지 않고 네 run 모두 새로 시작한다.

```bash
python -m unittest discover -s tests -v
python run.py check --device cpu --size 64 --report verification/synthetic_cpu.json
python preflight.py --data /path/to/Data --cache /path/to/torch_cache --device mps --report verification/mps.json
python launch_experiments.py start --output runs/rrcu_scse_mps_20261003 --data /path/to/Data --cache /path/to/torch_cache --device mps --preflight verification/mps.json
```

preflight는 같은 첫4개crop/초기가중치/RNG를 확인하고 실제 Train batch4에서2 optimizer step씩 실행한다.
이때의 가중치는 버린다. 실제150epoch 학습은 별도로fresh 시작한다. preflight의 loss는 성능 결과가 아니다.

```bash
cat runs/rrcu_scse_mps_20261003/trials/suite_status.json
cat runs/rrcu_scse_mps_20261003/trials/R01/status.json
tail -n 5 runs/rrcu_scse_mps_20261003/trials/logs/R01.log
python launch_experiments.py stop --output runs/rrcu_scse_mps_20261003
```

중단하면 나머지 큐도 취소하고 마지막 완료 epoch checkpoint를 보존한다.
학습과 큐 재개는 source_snapshot 폴더에서 같은 run_suite.py에 `--resume`를 넣어 실행한다.
결과 비교 시 순서는 B를 맨 앞에 둔다.

```bash
python run.py compare --runs runs/rrcu_scse_mps_20261003/trials/B runs/rrcu_scse_mps_20261003/trials/R01 runs/rrcu_scse_mps_20261003/trials/R02 runs/rrcu_scse_mps_20261003/trials/R03 --output comparisons/rrcu_scse_seed42
```

서버에서도 동일 코드로 `run_suite.py --device cuda --data /path/to/Data --output runs/server_rrcu_scse`를 실행할 수 있다.
CUDA 설치는 requirements-cuda.txt(PyTorch2.7.1/torchvision0.22.1)와 requirements.txt를 사용한다.
MPS/PyTorch2.8.0 결과와 CUDA/PyTorch2.7.1의 과거 점수를 직접 비교하지 않는다.

참고: [R2U-Net](https://arxiv.org/abs/1802.06955), [scSE](https://arxiv.org/abs/1808.08127).
원 논문의 전체 구조 재현이나 검증된 KoMaP 개선을 주장하지 않는 파생 비교 실험이다.
