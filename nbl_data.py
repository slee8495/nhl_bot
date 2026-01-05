# nhl_data.py

import pandas as pd
from config import NHL_TRAIN_PATH, NHL_TODAY_PATH


def load_training_data() -> pd.DataFrame:
    """
    nhl_train.csv 형식 (예시):
      - game_id
      - date
      - season
      - home_team
      - away_team
      - feature_1, feature_2, ..., feature_k
      - target   # 1 = home 팀 승리, 0 = 패배

    필수:
      - 'target'
    """
    df = pd.read_csv(NHL_TRAIN_PATH)
    if "target" not in df.columns:
        raise ValueError("nhl_train.csv must contain a 'target' column.")
    return df


def load_today_games() -> pd.DataFrame:
    """
    nhl_today.csv 형식 (예시):

      game_id
      date
      home_team
      away_team

      # model input features (훈련 때와 동일 스키마)
      feature_1 ...
      feature_k

      # betting odds
      side              'home' 또는 'away'
      american_odds     예: -150, +120

    최소 requirements:
      - game_id
    """
    df = pd.read_csv(NHL_TODAY_PATH)
    if "game_id" not in df.columns:
        raise ValueError("nhl_today.csv must contain 'game_id'.")
    return df