# 데이터와 재현 안내

이 저장소에는 대회 데이터, 체크포인트, 예측 마스크를 올리지 않는다. 권한이 있는 로컬/서버 경로에 다음과 같이 데이터를 준비한 뒤 학습 코드에 `--data`로 경로를 전달한다.

```text
Data/
  train/images/*_image.png
  train/masks/*_mask.png
  valid/images/*_image.png
  valid/masks/*_mask.png
  test/images/*_image.png
```

기준 split은 Train 70쌍, Valid 20쌍, Test 10장(라벨 없음)이다. 실험 결과는 Valid 20장만을 이용하며 official Test 성능으로 해석하지 않는다. 체크포인트 선택은 single-view Valid 기준이고 D4는 선택된 checkpoint의 후속 평가다.

R022 수치는 기존 팀 run 기록에서 가져온 것이고 이 저장소의 재현 스크립트를 실행해 새로 얻은 값이 아니다. 환경/가중치/데이터 해시를 포함한 서버 run artifact가 없으면 bitwise 재현으로 주장하지 않는다. 공개된 result table은 원 제공 README의 기록을 요약했으며 서버 파일을 이 저장소에서 재조회하지 않았다.
