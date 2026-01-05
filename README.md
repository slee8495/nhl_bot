
이 봇의 목표는, 작은 엣지를 여러개 찾는 것이기 때문에. 

nba_bot 보다 1/10 으로 작게 들어가는게 목표인 봇. 왜냐면, 시장이 틀리는만큼 나도 틀릴 가능성이 그만큼 더 높지만, 수많은 베팅중에 엣지 찾는게 목표이기 때문에. 

즉, 하루에 nba 게임이 8개이면, 게임의 수를 9개로 가정하고, nba 게임 하나값만큼, nhl 전체에 들어가는 구조임. 

--



- nba_bot.yml (nhl_bot.yml)
- config.py (config.py)
- email_management.py (email_management.py)
- nba_advanced_stats.py ()
- nba_api_client.py (nhl_api_client.py)
- nba_data.py (nhl_data.py)
- nba_edge.py (nhl_edge.py)
- nba_feature_engineering.py (nhl_feature_engineering.py)
- nba_features.py (nhl_features.py)
- nba_injury_lineup.py ()
- nba_margin_model.py (nhl_goal_diff_model.py)
- nba_performance.py ()
- nba_quant_bot.py ()
- nba_quant_main_next_day.py ()
- nba_quant_main.py ()
- nba_quant_model.py (nhl_quant_model.py)
- nba_strategy_tuning.py (nhl_strategy_tuning.py)
- nba_team_abbr.py (nhl_team_abbr.py)
- nba_train_quant.py (nhl_train_quant.py)
- odds_client.py (odds_client.py)




	3.	nhl_performance.py

	•	bet log 저장/정산/ROI/월별/누적
	•	strategy tuning이 읽을 nhl_bet_log.csv 포맷을 여기서 고정



	5.	nhl_advanced_stats.py

	•	지금 피처 엔지니어링이 비교적 단순(롤링+Elo+rest)이라면,
	•	나중에 xG, shot attempt, special teams 같은 고급 스탯 소스 붙일 때 여기서 확장

	6.	nhl_injury_lineup.py

	•	NHL은 라인/골리(Goalie)가 핵심이라 “선발 골리” 붙이는 순간 파워가 커짐
	•	다만 데이터 소스/매칭이 귀찮아서 파이프라인 완성 후에 붙이는 게 맞음

