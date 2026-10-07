# U²-Net / RSU 결합 실험 우선순위

목표는 RSU의 다중 해상도 표현을 기존 B 모델에 연결했을 때 Al3Ni–Eutectic 혼동과 가는 조직 분할이 개선되는지 확인하는 것이다. 기존 B의 scSE와 D03 접촉부 loss를 우선 비교한다. RRCU는 현재까지 성공이 확정된 구성요소가 아니므로 후순위 독립 비교로 둔다.

## 1. 먼저 core 5개 / seed42 / 각150epoch

| 비교 | 확인할 핵심 | 채택 판단 |
|---|---|---|
| U0 vs B | RSU encoder와 adapter를 포함한 시스템의 가능성 | Single/D4, 상별 IoU, 비용을 함께 비교 |
| U0 vs U1 | RSU 조합에서도 scSE가 필요한가 | 접촉부/가는 조직 개선이 반복되는가 |
| B_D03 vs B | 이번 sampler·환경에서 D03 이득이 있는가 | 기존 보고를 대조군에서 다시 확인 |
| U2 vs U0 | RSU 조합에서 접촉부 가중이 도움이 되는가 | 접촉부 감소와 Eutectic 손실을 함께 확인 |
| U2 vs B_D03 | 기존 최선 후보보다 결합 구조가 좋은가 | 작은 상승은 추가 seed로 확인 |

기본 실행은 총5run이다. 과거 서버의 B 점수를 대입하지 않는다. 모든 비교는 이번 코드/환경/Train70·Valid20에서 동일 seed·예산으로 다시 얻은 결과를 쓴다. U0−B에는 ImageNet 사전학습 차이와 adapter가 포함되므로 순수 RSU 효과로 부르지 않는다.

## 2. 유망 후보 재현 / seeds43·44

`reproduce`는 B/B_D03/U0/U2 네 종류를 seeds43·44에서 비교한다. 총8run이다. 기본 core seed42와 합치면 후보당3seeds가 된다. `--ids U0`로 U2를 제외하면 필수 대조군 B/B_D03/U0만 실행한다. U1 또는 특정 RRCU가 유망하면 해당 stage에서 대조군과 같은 추가 seed로 비교한다.

Single/D4의 동일 seed 차이, 접촉부 오류, Eutectic IoU·precision·recall·skeleton recall을 함께 확인한다. 이미지 bootstrap 95% 구간은 이미지 변동을 보여주며 학습 seed 변동을 대신하지 않는다. 여러 후보를 같은 Valid20에서 선택하기 때문에 이 점수는 탐색 결과다.

## 3. 초기화와 decoder 영향 분리

`scratch`: B_SCRATCH/U0 두 run. RSU가 낮을 때 ImageNet 부재가 큰 요인인지 탐색한다. 두 모델을 scratch로 맞춰도 구조와 adapter의 차이는 남는다.

`full`: B/F0/F1/F0_D03/F2 다섯 run. 원형 full U²-Net은 원본 해상도 RSU decoder skip까지 사용한다. U0가 낮다고 full 모델까지 효과가 없다고 결론 내리지 않는다. F1−F0는 scSE, F0_D03−F0는 접촉부 loss, F2−F1는 scSE 구조에서 접촉부 loss 효과다. Full끼리는 scratch 초기화를 통일한다.

## 4. 반복 정제와 추가 감독

`rrcu`: U0/U3_R01/U3_R02/U3_R03 네 run. U0가 유망하거나 특정 혼동·누락이 남을 때 적용한다. RSU는 이미 내부 다중 해상도 정제를 수행하므로 RRCU 중복이 도움이 될지는 실험해야 한다. 모든 gate에 한꺼번에 recurrence를 넣지 않고 얕은 decoder 한 지점부터 비교한다.

`deep`: F1/F3 두 run. F1이 유망하거나 학습 불안정이 보일 때 six-side 직접 감독을 비교한다. 분모1.6으로 loss 총 스케일을 맞추고 fused checkpoint 선택 규칙을 유지한다. Side head 증가 자체는 F1에도 있으므로 F3−F1는 auxiliary 감독 유무의 비교다.

## 보류할 결합

RSU + scSE + D03 + RRCU + deep supervision을 동시에 켜는 실험은 포함하지 않았다. 개별 효과를 확인한 다음 유망한 요소만 결합한다. BEM·상 구별 대조학습도 이번 서버 보고에서 이득이 명확하지 않아 제외했다. 현재 14개 설정은 선택지 목록이며 기본값으로14개 전체를 실행하지 않는다.

## 결과 회신 기준

`seed_summary.json`의 `ablation_contrasts`와 각 run의 `summary.json`을 중심으로 회신한다. 초기 loss가 감소하는지만 보지 말고 best epoch, single/D4, strict mIoU, 상별 손익과 peak memory/시간까지 비교한다. 접촉부 오류 감소를 두 상의 전반적 품질 향상과 동일하게 해석하지 않는다.
