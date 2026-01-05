# nhl_train_quant.py
from __future__ import annotations

from datetime import date

from nhl_api_client import NHLDataClient
from nhl_feature_engineering import build_training_dataset
from nhl_quant_model import NHLQuantModel
from nhl_goal_diff_model import NHLGoalDiffModel


def get_training_seasons_nhl(n_seasons: int = 3) -> list[int]:
    """
    최근 n_seasons개의 '완료된 시즌'만 학습에 사용.

    NHL 시즌 기준:
      - 보통 10월 시작 ~ 다음해 6월 종료
      - 여기서는 10월(>=10)이면 "그 해 시즌 진행중"으로 보고,
        last_completed = current_season - 1 로 정의 (NBA 로직과 동일)

    예)
      - 2026-01: current_season=2025, last_completed=2024 -> [2022, 2023, 2024] (n=3)
      - 2026-11: current_season=2026, last_completed=2025 -> [2023, 2024, 2025]
    """
    today = date.today()
    current_season = today.year if today.month >= 10 else today.year - 1
    last_completed = current_season - 1

    start = last_completed - (n_seasons - 1)
    return list(range(start, last_completed + 1))


def main():
    seasons = get_training_seasons_nhl(n_seasons=3)
    print(f"[NHL TRAIN] Fetching games for seasons: {seasons}")

    client = NHLDataClient()
    raw_games = client.fetch_games_by_season(seasons)

    if raw_games is None or len(raw_games) == 0:
        print("[NHL TRAIN] No games fetched. Check API or seasons list.")
        return

    print(f"[NHL TRAIN] Raw games fetched: {len(raw_games)}")

    train_df, feature_cols = build_training_dataset(raw_games)
    print(f"[NHL TRAIN] Training rows: {len(train_df)}, features: {len(feature_cols)}")

    if train_df is None or train_df.empty:
        print("[NHL TRAIN] build_training_dataset returned empty. Abort.")
        return

    # =========================
    # A) Win model (NO time-weight)
    # =========================
    win_model = NHLQuantModel()
    win_eval = win_model.train_walkforward(
        train_df=train_df,
        feature_cols=feature_cols,
        n_splits=5,
        half_life_days=None,  # 처음은 OFF
        save=True,
    )

    print(
        f"[NHL TRAIN] Win model saved. n_games={win_eval['n_games']}, "
        f"mean_auc={win_eval['mean_auc']:.4f}, mean_logloss={win_eval['mean_logloss']:.4f}"
    )

    # =========================
    # B) GoalDiff model (NO time-weight)
    # =========================
    gd_model = NHLGoalDiffModel()
    gd_eval = gd_model.train_walkforward(
        train_df=train_df,
        feature_cols=feature_cols,
        n_splits=5,
        half_life_days=None,
        save=True,
    )

    print(
        f"[NHL TRAIN] GoalDiff model saved. n_games={gd_eval['n_games']}, "
        f"mean_mae={gd_eval['mean_mae']:.3f}, mean_rmse={gd_eval['mean_rmse']:.3f}, "
        f"residual_std={gd_eval['residual_std']:.3f}"
    )


if __name__ == "__main__":
    main()