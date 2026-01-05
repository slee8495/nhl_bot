# nhl_features.py

from __future__ import annotations
import numpy as np
import pandas as pd


def build_training_df(raw_games: pd.DataFrame):
    """
    raw_games 예시 컬럼 구조:

      - game_id
      - date
      - season
      - home_team, away_team
      - home_score, away_score

      - home_xxx, away_xxx (예: elo, rest_days, rolling goals for/against 등)
      - 기타 숫자형 피처들

    output:
      - df_model
      - feature_cols
    """
    df = raw_games.copy()

    # target: home win
    if {"home_score", "away_score"}.issubset(df.columns):
        df["target"] = (df["home_score"] > df["away_score"]).astype(int)
    elif "target" in df.columns:
        pass
    else:
        raise ValueError("raw_games must have either (home_score, away_score) or 'target'.")

    # date normalize
    if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # numeric features
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    drop_for_features = {
        "game_id",
        "season",
        "home_score",
        "away_score",
        "target",
    }
    feature_cols = [c for c in num_cols if c not in drop_for_features]

    keep_cols = ["game_id", "date", "season", "home_team", "away_team", "target"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = np.nan  # upstream에서 채우는 게 베스트지만, 일단 안전하게

    df_model = df[keep_cols + feature_cols].dropna(subset=feature_cols + ["target"])
    return df_model, feature_cols


def build_today_feature_df(today_games: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    오늘 경기(today_games)를 학습 feature_cols 기준으로 정리.
    side == 'home' => model_p = p_home_win
    side == 'away' => model_p = 1 - p_home_win
    """
    df = today_games.copy()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"build_today_feature_df: missing features {missing} in today_games")
    return df