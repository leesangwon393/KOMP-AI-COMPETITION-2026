# 완료 실험 결과 요약

원자료: 팀 제공 `README_KOMAP_ZIP_ALL_KO.md` (2026-10-02). 아래 점수는 모두 로컬 Valid 20장의 이미지 macro mIoU이며 공식 Test 결과가 아니다. D4는 checkpoint 선택 이후 적용한 8-view TTA다. absent=1은 union=0 클래스를 IoU 1로, strict는 IoU 0으로 처리한다. Δ는 각 행의 문서상 부모 recipe 대비 percentage point다.

R012–R044는 완료 32개와 중단 R015 1개를 포함한다. ZIP-REF는 앞선 baseline이다. Source report의 linked server files/checkpoints는 이 repository에 포함되지 않았다.

| ID | 주요 변경 | Epoch / seed | Best epoch | D4 absent=1 | D4 strict | Δ pp | 판단 |
|---|---|---:|---:|---:|---:|---:|---|
| ZIP-REF | ImageNet ResNet34 U-Net+scSE, 원본 Context448 recipe | 100/100 · 42 | 98 | 0.7849573723 | 0.7724573723 | — | 원본 기준 |
| R012 | ZIP-REF: MatSSL ResNet50로 encoder 교체; ZIP의 100e 문맥·loss·sampling 유지 | 100/100 · 42 | 83 | 0.7739475229 | 0.7614475229 | -1.1010 | 미채택 |
| R013 | R012: MatSSL ResNet50 유지, gray 단일채널 정규화→RGB 반복/ImageNet 채널별 정규화 | 100/100 · 42 | 63 | 0.7732681795 | 0.7607681795 | -0.0679 | 미채택 |
| R014 | ZIP-REF: ResNet34 유지; Al3Ni/Eutectic 균형 sampling + 약한 광도 증강 + 100→150e | 150/150 · 42 | 137 | 0.7930962830 | 0.7805962830 | +0.8139 | 기본 recipe에 반영 |
| R015 | R014 + boundary; 동일-epoch 지시로 중단 | 120/180 · 42 (중단/계획seed) | — | — | — | — | 중단·완료 비교 제외 |
| R016 | R014: train-only 4클래스 boundary head/보조 loss, 가중치 0.1 | 150/150 · 42 | 129 | 0.7928485676 | 0.7803485676 | -0.0248 | 미채택 |
| R017 | R014: Eutectic Si clDice 보조 loss, 가중치 0.05 | 150/150 · 42 | 131 | 0.7912932426 | 0.7787932426 | -0.1803 | 미채택 |
| R018 | R014: full-resolution grayscale detail skip 추가 | 150/150 · 42 | 137 | 0.7930963258 | 0.7805963258 | +0.0000 | 표시자리 동률·개선 미채택 |
| R019 | R014: ImageNet V2 ResNet50 + RGB 반복/ImageNet 정규화 | 150/150 · 42 | 139 | 0.7946244054 | 0.7821244054 | +0.1528 | 기본 recipe에 반영 |
| R020 | R019: Dice 0.5 중 절반을 per-image weighted Lovasz로 교체 | 150/150 · 42 | 88 | 0.7969365058 | 0.7844365058 | +0.2312 | 기본 recipe에 반영 |
| R021 | R020: Al3Ni/Eutectic-rich tile 최소 비율 10%→3% | 150/150 · 42 | 118 | 0.7940841594 | 0.7815841594 | -0.2852 | 미채택 |
| R022 | R020: encoder ImageNet V2 ResNet50→ResNet101 | 150/150 · 42 | 123 | 0.7989694265 | 0.7864694265 | +0.2033 | 기본 recipe에 반영 |
| R023 | R022: encoder ResNet101→ImageNet V2 ResNet152 | 150/150 · 42 | 103 | 0.7966134210 | 0.7841134210 | -0.2356 | 미채택 |
| R024 | R022: ImageNet V1 ConvNeXt-Tiny + grayscale detail skips; loss/sampler 유지 | 150/150 · 42 | 126 | 0.7978432398 | 0.7853432398 | -0.1126 | 미채택 |
| R025 | R024: ConvNeXt-Tiny→ConvNeXt-Small | 150/150 · 42 | 133 | 0.7923599793 | 0.7798599793 | -0.5483 | 미채택 |
| R026 | R022: encoder LR 3e-4→1e-4; decoder LR 3e-4 유지 | 150/150 · 42 | 137 | 0.7887243826 | 0.7762243826 | -1.0245 | 미채택 |
| R027 | R022: Eutectic Si의 Dice/Lovasz 가중치 2→1; CE는 그대로 | 150/150 · 42 | 123 | 0.7960386857 | 0.7835386857 | -0.2931 | 미채택 |
| R028 | R022: 밝기/대비·감마·노이즈 광도 증강만 제거; fresh attempt2 | 150/150 · 42 | 121 | 0.7927958114 | 0.7802958114 | -0.6174 | 미채택 |
| R029 | R022: 마지막 bilinear 업샘플링만 DySample LP x2/groups4로 교체 | 150/150 · 42 | 95 | 0.7975628892 | 0.7850628892 | -0.1407 | 미채택 |
| R030 | R022: 감독 영역 192/224/256 균등 배율 jitter; 문맥은 2배 후448/224로 resize; 추론448 고정 | 150/150 · 42 | 131 | 0.7916569615 | 0.7791569615 | -0.7312 | 미채택 |
| R031 | R022: Dice batch-pooling → 타일별 spatial(2,3) 계산 후 평균; CE/Lovasz 유지 | 150/150 · 42 | 118 | 0.7993317515 | 0.7868317515 | +0.0362 | 관측 최고 checkpoint 보존; 우위 미채택 |
| R032 | R022 기본 recipe의 seed43 fresh 재검증 (seed42와 다른 초기화/RNG) | 150/150 · 43 | 99 | 0.7975259082 | 0.7850259082 | -0.1444 | 기준 seed43 재현 대조 |
| R033 | R031 타일별 Dice의 seed43 fresh 재검증; same-seed R032와 대조 | 150/150 · 43 | 92 | 0.7948138677 | 0.7823138677 | -0.2712 | seed43 이득 재현 실패·미채택 |
| R034 | R022: decoder BN10개 → GroupNorm8; encoder BN 유지 | 150/150 · 42 | 117 | 0.7924690619 | 0.7799690619 | -0.6500 | 미채택 |
| R035 | R022: encoder BN104개 pretrained 통계 고정, affine 학습; decoder BN 유지 | 150/150 · 42 | 137 | 0.7978640491 | 0.7853640491 | -0.1105 | 미채택 |
| R036 | R022: CE만 타일별 손실 상위25% 평균 (k=max(1,floor(HW/4)), threshold 없음) | 150/150 · 42 | 108 | 0.7964402627 | 0.7839402627 | -0.2529 | 미채택 |
| R037 | R022: native crop 원점 dy/dx ±56px; clamp, targeted 희귀상10% 미달이면 원래 위치 fallback | 150/150 · 42 | 90 | 0.7962283235 | 0.7837283235 | -0.2741 | 미채택 |
| R038 | R022: RGB HRNet-W18 ms_in1k + stride4 merge/head; architecture+pretraining 시스템 대조 | 150/150 · 42 | 137 | 0.7498299370 | 0.7373299370 | -4.9139 | 미채택 |
| R039 | R022: ResNet101 layer4 stride→dilation, OS32→16; 기존 decoder 유지 | 150/150 · 42 | 137 | 0.7980706151 | 0.7855706151 | -0.0899 | 미채택 |
| R040 | R022: 감독224 안112-square image-mask CutMix, p=.5/batch, nonself donor; 별도 mix RNG | 150/150 · 42 | 119 | 0.7940353882 | 0.7815353882 | -0.4934 | 미채택 |
| R041 | R022: CE만 GT radius2 경계2/내부1 가중, 타일별 sumweight 정규화 | 150/150 · 42 | 118 | 0.7972385272 | 0.7847385272 | -0.1731 | 미채택 |
| R042 | R022: ImageNet V2 ResNeXt101-32x8d encoder; 동일 skip폭/decoder, architecture+pretraining 대조 | 150/150 · 42 | 101 | 0.7949337633 | 0.7824337633 | -0.4036 | 미채택 |
| R043 | R022: 기본 fresh 모델 생성 후 decoder0–4 scSE→Identity; 나머지 초기 state/RNG 유지 | 150/150 · 42 | 118 | 0.7955732038 | 0.7830732038 | -0.3396 | 미채택 |
| R044 | R022: CE.5+Dice.25+Lovasz.25 → CE.5+Dice0+Lovasz.5; classweight/present-class 정책 유지 | 150/150 · 42 | 90 | 0.7963787580 | 0.7838787580 | -0.2591 | 미채택 |

## 해석

- 유지 기준 recipe: **R022**, D4 absent=1 0.7989694265 / strict 0.7864694265.
- 관측 최고 checkpoint: **R031**, 0.7993317515 / 0.7868317515. R022보다 +0.0362 pp였으나 seed 43 대조에서는 R031−R022가 −0.2712 pp이고 두-seed 평균도 −0.1175 pp다. R031은 관측 최고로 보존하되 일반 우위로 채택하지 않는다.
- R022의 전체 설정 및 R012–R044 변경 해설은 원 제공 report를 참조한다. R045는 source report 작성 때 3/150 epoch 상태였고 완료 점수는 없다. 그 이후 상태는 여기서 재조회하지 않았다.
- RRCU decoder ablation C13 was added after these reported runs. A 150-epoch local MPS run started on 2026-10-02, followed by a queued same-code B baseline. Final C13 results are pending; see [experiment details](RRCU_EXPERIMENT_KO.md).
- 수치는 Valid를 반복 사용한 실험 기록이므로 독립 test 또는 held-out 일반화 성능으로 표현하지 않는다.
