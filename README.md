
시즌 끝나면, models 폴더 안에 있는 두개의 .pkl 파일 지우기! 




베팅은 XGB 승률과 최종 앙상블 모델 승률이 모두 50% 이상인 경우만 고려하며, 기본적으로 Edge가 Non-Market-Flip이면 ≥3%(단 오즈가 +200 이상이면 ≥5%), Market-Flip이면 ≥6%(단 +200 이상이면 ≥8%)일 때만 허용하고, 추가로 Non-Market-Flip 상황에서 XGB≥50%·Model≥50%·|American odds|≤250·|Edge|≥1%를 동시에 만족하면 자동으로 YES 처리하며, 이외의 경우는 모두 배제한다.


