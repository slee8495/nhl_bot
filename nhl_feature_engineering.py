# nhl_feature_engineering.py
# -------------------------------------------------------------------
# NHL feature engineering (MVP) for moneyline win prediction.
#
# Design goals:
# - Mirror NBA pipeline style you already have
# - Pre-game features only (NO leakage): all rolling/season stats are lagged by 1 game
# - Robust to balldontlie schema differences (away_team vs visitor_team, score key variants)
# - Produces:
#   - build_training_dataset(raw_games) -> (train_df, feature_cols)
#   - build_today_dataset(raw_games, today_games, today) -> (today_df, feature_cols)
#
# Expected raw_games columns (after nhl_api_client normalization):
#   game_id, date, season,
#   home_team, away_team,
#   home_team_abbr, away_team_abbr,
#   home_score, away_score
#
# Expected today_games columns (from nhl_api_client fetch_games_by_date_df):
#   game_id (or id), date, season,
#   home_team, away_team (or visitor_team),
#   home_team_abbr, away_team_abbr (if present)
# -------------------------------------------------------------------

from __future__ import annotations

from typing import Tuple
from datetime import date as _date

import numpy as np
import pandas as pd

from nhl_team_abbr import to_abbr


# =========================================================
# Elo (optional but useful and cheap)
# =========================================================
def _add_elo_to_games(
    raw_games: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 20.0,
) -> pd.DataFrame:
    """
    Add pre-game Elo for home/away per game.
    Updates only when scores exist.
    Produces:
      - elo_home, elo_away columns (pre-game)
    """
    df = raw_games.copy()

    if "date" not in df.columns:
        raise ValueError("raw_games missing 'date'")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    elo_home_list: list[float] = []
    elo_away_list: list[float] = []

    for _, row in df.iterrows():
        home = str(row.get("home_team", "") or "")
        away = str(row.get("away_team", "") or "")

        Rh = ratings.get(home, base_rating)
        Ra = ratings.get(away, base_rating)

        elo_home_list.append(Rh)
        elo_away_list.append(Ra)

        hs = row.get("home_score", np.nan)
        aws = row.get("away_score", np.nan)
        if pd.isna(hs) or pd.isna(aws):
            continue

        if hs > aws:
            Sh, Sa = 1.0, 0.0
        elif hs < aws:
            Sh, Sa = 0.0, 1.0
        else:
            Sh, Sa = 0.5, 0.5  # shootout/tie-like guard (rare; depends on provider)

        Eh = 1.0 / (1.0 + 10.0 ** ((Ra - Rh) / 400.0))
        Sa_expect = 1.0 - Eh

        ratings[home] = Rh + k_factor * (Sh - Eh)
        ratings[away] = Ra + k_factor * (Sa - Sa_expect)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    return df


# =========================================================
# Team panel: team × game rows
# =========================================================
def _build_team_level_panel(raw_games: pd.DataFrame) -> pd.DataFrame:
    """
    Build a team-game panel with pre-game rolling & season-to-date stats.
    NO leakage: everything is shifted by 1 game.
    Output key columns include:
      team, date, season, game_id, is_home,
      win, goal_diff, goals_for, goals_against,
      rolling_* (5/10), season_* , rest_days, is_back_to_back,
      elo_like (if available)
    """
    df = raw_games.copy()
    if "date" not in df.columns:
        raise ValueError("raw_games missing 'date'")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # training panel requires scores
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    has_elo = ("elo_home" in df.columns) and ("elo_away" in df.columns)

    # ---- home rows
    cols_home = ["game_id", "date", "season", "home_team", "home_score", "away_score"]
    if has_elo:
        cols_home.append("elo_home")
    home_df = df[cols_home].copy()
    home_df.rename(
        columns={
            "home_team": "team",
            "home_score": "goals_for",
            "away_score": "goals_against",
        },
        inplace=True,
    )
    home_df["is_home"] = 1
    if has_elo:
        home_df["elo_like"] = home_df["elo_home"]
        home_df.drop(columns=["elo_home"], inplace=True)

    # ---- away rows
    cols_away = ["game_id", "date", "season", "away_team", "away_score", "home_score"]
    if has_elo:
        cols_away.append("elo_away")
    away_df = df[cols_away].copy()
    away_df.rename(
        columns={
            "away_team": "team",
            "away_score": "goals_for",
            "home_score": "goals_against",
        },
        inplace=True,
    )
    away_df["is_home"] = 0
    if has_elo:
        away_df["elo_like"] = away_df["elo_away"]
        away_df.drop(columns=["elo_away"], inplace=True)

    team_panel = pd.concat([home_df, away_df], ignore_index=True)
    team_panel = team_panel.sort_values(["team", "date"]).reset_index(drop=True)

    team_panel["win"] = (team_panel["goals_for"] > team_panel["goals_against"]).astype(int)
    team_panel["goal_diff"] = team_panel["goals_for"] - team_panel["goals_against"]

    # ---- rolling + season-to-date (PRE-GAME = shift 1)
    def _add_rolling(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()

        win_l1 = g["win"].shift(1)
        gd_l1 = g["goal_diff"].shift(1)
        gf_l1 = g["goals_for"].shift(1)
        ga_l1 = g["goals_against"].shift(1)
        home_l1 = g["is_home"].shift(1)

        g["rolling_win_rate_10"] = win_l1.rolling(10, min_periods=1).mean()
        g["rolling_goal_diff_10"] = gd_l1.rolling(10, min_periods=1).mean()
        g["rolling_goals_for_10"] = gf_l1.rolling(10, min_periods=1).mean()
        g["rolling_goals_against_10"] = ga_l1.rolling(10, min_periods=1).mean()
        g["rolling_home_ratio_10"] = home_l1.rolling(10, min_periods=1).mean()

        g["rolling_win_rate_5"] = win_l1.rolling(5, min_periods=1).mean()
        g["rolling_goal_diff_5"] = gd_l1.rolling(5, min_periods=1).mean()
        g["rolling_goals_for_5"] = gf_l1.rolling(5, min_periods=1).mean()
        g["rolling_goals_against_5"] = ga_l1.rolling(5, min_periods=1).mean()
        g["rolling_home_ratio_5"] = home_l1.rolling(5, min_periods=1).mean()

        # season-to-date counts exclude current game
        g["season_games"] = g.groupby("season").cumcount()
        g["season_wins"] = g.groupby("season")["win"].shift(1).fillna(0).groupby(g["season"]).cumsum()
        g["season_win_rate"] = np.where(g["season_games"] > 0, g["season_wins"] / g["season_games"], np.nan)

        g["season_goal_diff_cum"] = g.groupby("season")["goal_diff"].shift(1).fillna(0).groupby(g["season"]).cumsum()
        g["season_goal_diff_avg"] = np.where(g["season_games"] > 0, g["season_goal_diff_cum"] / g["season_games"], np.nan)

        if "elo_like" in g.columns:
            elo_l1 = g["elo_like"].shift(1)
            g["rolling_elo_10"] = elo_l1.rolling(10, min_periods=1).mean()
            g["rolling_elo_5"] = elo_l1.rolling(5, min_periods=1).mean()

        return g

    team_panel = team_panel.groupby("team", group_keys=False).apply(_add_rolling)

    # ---- rest days / b2b
    team_panel["prev_date"] = team_panel.groupby("team")["date"].shift(1)
    team_panel["rest_days"] = (team_panel["date"] - team_panel["prev_date"]).dt.days
    team_panel["is_back_to_back"] = (team_panel["rest_days"] == 1).astype(int)

    return team_panel


# =========================================================
# Public: Training dataset
# =========================================================
def build_training_dataset(raw_games: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    raw_games -> game-level training df (home perspective)
    target: 1 if home wins else 0

    Returns:
      train_df, feature_cols
    """
    if raw_games is None or raw_games.empty:
        return pd.DataFrame(), []

    rg = raw_games.copy()

    # normalize date
    rg["date"] = pd.to_datetime(rg["date"], errors="coerce")

    # drop rows without scores (can't form target)
    rg = rg.dropna(subset=["home_score", "away_score"]).copy()

    # Elo
    rg = _add_elo_to_games(rg)

    # team panel
    team_panel = _build_team_level_panel(rg)

    # --- merge home stats
    home_stats = team_panel.add_suffix("_home").rename(
        columns={
            "team_home": "home_team",
            "date_home": "date",
            "game_id_home": "game_id",
            "season_home": "season",
        }
    )

    # --- merge away stats
    away_stats = team_panel.add_suffix("_away").rename(
        columns={
            "team_away": "away_team",
            "date_away": "date",
            "game_id_away": "game_id",
            "season_away": "season",
        }
    )

    df = rg.copy()

    # columns we want (existence-checked)
    home_cols_base = [
        "game_id", "home_team", "date",
        "rolling_win_rate_10_home",
        "rolling_goal_diff_10_home",
        "rolling_goals_for_10_home",
        "rolling_goals_against_10_home",
        "rolling_home_ratio_10_home",
        "season_win_rate_home",
        "season_goal_diff_avg_home",
        "season_games_home",
        "rolling_win_rate_5_home",
        "rolling_goal_diff_5_home",
        "rolling_goals_for_5_home",
        "rolling_goals_against_5_home",
        "rolling_home_ratio_5_home",
        "rest_days_home",
        "is_back_to_back_home",
        "elo_like_home",
        "rolling_elo_10_home",
        "rolling_elo_5_home",
    ]
    away_cols_base = [
        "game_id", "away_team", "date",
        "rolling_win_rate_10_away",
        "rolling_goal_diff_10_away",
        "rolling_goals_for_10_away",
        "rolling_goals_against_10_away",
        "rolling_home_ratio_10_away",
        "season_win_rate_away",
        "season_goal_diff_avg_away",
        "season_games_away",
        "rolling_win_rate_5_away",
        "rolling_goal_diff_5_away",
        "rolling_goals_for_5_away",
        "rolling_goals_against_5_away",
        "rolling_home_ratio_5_away",
        "rest_days_away",
        "is_back_to_back_away",
        "elo_like_away",
        "rolling_elo_10_away",
        "rolling_elo_5_away",
    ]

    home_cols = [c for c in home_cols_base if c in home_stats.columns]
    away_cols = [c for c in away_cols_base if c in away_stats.columns]

    df = df.merge(home_stats[home_cols], on=["game_id", "home_team", "date"], how="left")
    df = df.merge(away_stats[away_cols], on=["game_id", "away_team", "date"], how="left")

    # target
    df["target"] = (df["home_score"] > df["away_score"]).astype(int)

    # diffs (existence guarded)
    def _safe_diff(a: str, b: str, out: str):
        if (a in df.columns) and (b in df.columns):
            df[out] = pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b], errors="coerce")

    _safe_diff("rolling_win_rate_10_home", "rolling_win_rate_10_away", "diff_rolling_win_rate_10")
    _safe_diff("rolling_goal_diff_10_home", "rolling_goal_diff_10_away", "diff_rolling_goal_diff_10")
    _safe_diff("season_win_rate_home", "season_win_rate_away", "diff_season_win_rate")
    _safe_diff("season_goal_diff_avg_home", "season_goal_diff_avg_away", "diff_season_goal_diff_avg")
    _safe_diff("elo_like_home", "elo_like_away", "diff_elo_like")
    _safe_diff("rolling_win_rate_5_home", "rolling_win_rate_5_away", "diff_rolling_win_rate_5")
    _safe_diff("rolling_goal_diff_5_home", "rolling_goal_diff_5_away", "diff_rolling_goal_diff_5")

    # feature cols list (keep it stable)
    feature_cols: list[str] = [
        # home/away base
        "rolling_win_rate_10_home",
        "rolling_win_rate_10_away",
        "rolling_goal_diff_10_home",
        "rolling_goal_diff_10_away",
        "rolling_goals_for_10_home",
        "rolling_goals_for_10_away",
        "rolling_goals_against_10_home",
        "rolling_goals_against_10_away",
        "rolling_home_ratio_10_home",
        "rolling_home_ratio_10_away",
        "season_win_rate_home",
        "season_win_rate_away",
        "season_goal_diff_avg_home",
        "season_goal_diff_avg_away",
        "rest_days_home",
        "rest_days_away",
        "is_back_to_back_home",
        "is_back_to_back_away",
        "elo_like_home",
        "elo_like_away",
        "rolling_elo_10_home",
        "rolling_elo_10_away",
        "rolling_elo_5_home",
        "rolling_elo_5_away",
        "rolling_win_rate_5_home",
        "rolling_win_rate_5_away",
        "rolling_goal_diff_5_home",
        "rolling_goal_diff_5_away",
        "rolling_goals_for_5_home",
        "rolling_goals_for_5_away",
        "rolling_goals_against_5_home",
        "rolling_goals_against_5_away",
        "rolling_home_ratio_5_home",
        "rolling_home_ratio_5_away",
        # diffs
        "diff_rolling_win_rate_10",
        "diff_rolling_goal_diff_10",
        "diff_season_win_rate",
        "diff_season_goal_diff_avg",
        "diff_elo_like",
        "diff_rolling_win_rate_5",
        "diff_rolling_goal_diff_5",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # final train df
    base_cols = [
        "game_id", "date", "season", "home_team", "away_team",
        "home_score", "away_score", "target",
    ]
    # keep season games if present (useful for diagnostics)
    for c in ["season_games_home", "season_games_away"]:
        if c in df.columns:
            base_cols.append(c)

    train_df = df[base_cols + feature_cols].copy()

    # numeric cast + impute (mean) to avoid massive dropna
    for c in feature_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")

    fill_map = {c: float(train_df[c].mean()) for c in feature_cols}
    train_df = train_df.fillna(fill_map).fillna(0.0)

    # date to python date
    train_df["date"] = pd.to_datetime(train_df["date"], errors="coerce").dt.date

    return train_df, feature_cols


# =========================================================
# Public: Today's dataset
# =========================================================
def build_today_dataset(
    raw_games: pd.DataFrame,
    today_games: pd.DataFrame,
    today: _date,
) -> Tuple[pd.DataFrame, list[str]]:
    """
    Build today feature rows for model input.

    today_games: from NHLDataClient.fetch_games_by_date_df(today)
    We will:
      - normalize team fields
      - compute snapshot features from last available team_panel rows (pre-game)
      - merge for home/away, create diffs
      - return today_df + feature_cols (same list as training, intersection on columns)
    """
    if raw_games is None or raw_games.empty or today_games is None or today_games.empty:
        return pd.DataFrame(), []

    # --- normalize today_games minimal fields
    tg = today_games.copy()

    if "game_id" not in tg.columns and "id" in tg.columns:
        tg["game_id"] = tg["id"]

    if "date" in tg.columns:
        tg["date"] = pd.to_datetime(tg["date"], errors="coerce").dt.date
    else:
        tg["date"] = today

    # handle dict team objects (if any)
    def _extract_full_name(x):
        if isinstance(x, dict):
            return x.get("full_name")
        return x

    def _extract_abbr(x):
        if isinstance(x, dict):
            return x.get("abbreviation")
        return None

    # away may be away_team or visitor_team
    if "home_team" in tg.columns:
        tg["home_team"] = tg["home_team"].apply(_extract_full_name)
    if "away_team" in tg.columns:
        tg["away_team"] = tg["away_team"].apply(_extract_full_name)
    elif "visitor_team" in tg.columns:
        tg["away_team"] = tg["visitor_team"].apply(_extract_full_name)

    if "home_team_abbr" not in tg.columns and "home_team" in today_games.columns:
        tg["home_team_abbr"] = today_games["home_team"].apply(_extract_abbr)
    if "away_team_abbr" not in tg.columns:
        if "away_team" in today_games.columns:
            tg["away_team_abbr"] = today_games["away_team"].apply(_extract_abbr)
        elif "visitor_team" in today_games.columns:
            tg["away_team_abbr"] = today_games["visitor_team"].apply(_extract_abbr)
        else:
            tg["away_team_abbr"] = None

    # standardize team abbr (canonical)
    tg["home_team_abbr_std"] = tg.get("home_team_abbr", None)
    tg["away_team_abbr_std"] = tg.get("away_team_abbr", None)

    tg["home_team_abbr_std"] = tg["home_team_abbr_std"].apply(to_abbr)
    tg["away_team_abbr_std"] = tg["away_team_abbr_std"].apply(to_abbr)

    # if abbr missing, try from full name
    tg.loc[tg["home_team_abbr_std"].isna(), "home_team_abbr_std"] = tg.loc[
        tg["home_team_abbr_std"].isna(), "home_team"
    ].apply(to_abbr)
    tg.loc[tg["away_team_abbr_std"].isna(), "away_team_abbr_std"] = tg.loc[
        tg["away_team_abbr_std"].isna(), "away_team"
    ].apply(to_abbr)

    # --- raw games normalization for snapshot panel
    rg = raw_games.copy()
    rg["date"] = pd.to_datetime(rg["date"], errors="coerce")

    # past games must have scores
    rg = rg.dropna(subset=["home_score", "away_score"]).copy()

    # Elo
    rg = _add_elo_to_games(rg)

    # season fix (today games season might be missing or > latest)
    if "season" not in tg.columns:
        tg["season"] = rg["season"].max()
    latest_season = rg["season"].max()
    tg["season"] = tg["season"].apply(lambda s: latest_season if (pd.notna(s) and s > latest_season) else s)
    tg["season"] = tg["season"].fillna(latest_season)

    # team panel
    team_panel = _build_team_level_panel(rg)

    # snapshot = last row per team
    team_panel = team_panel.sort_values(["team", "date"])
    last_rows = team_panel.groupby("team").tail(1).copy()
    last_rows.rename(columns={"date": "last_game_date"}, inplace=True)

    # rest days vs today (more accurate than panel's last rest)
    last_rows["rest_days_today"] = (pd.to_datetime(today) - pd.to_datetime(last_rows["last_game_date"])).dt.days
    last_rows["is_back_to_back_today"] = (last_rows["rest_days_today"] == 1).astype(int)

    # keep snapshot columns
    base_keep = [
        "team",
        "season",
        "rolling_win_rate_10",
        "rolling_goal_diff_10",
        "rolling_goals_for_10",
        "rolling_goals_against_10",
        "rolling_home_ratio_10",
        "season_win_rate",
        "season_goal_diff_avg",
        "season_games",
        "rolling_win_rate_5",
        "rolling_goal_diff_5",
        "rolling_goals_for_5",
        "rolling_goals_against_5",
        "rolling_home_ratio_5",
        "elo_like",
        "rolling_elo_10",
        "rolling_elo_5",
        "rest_days_today",
        "is_back_to_back_today",
    ]
    keep_cols = [c for c in base_keep if c in last_rows.columns]
    snap = last_rows[keep_cols].copy()

    home_stats = snap.add_suffix("_home").rename(columns={"team_home": "home_team", "season_home": "season"})
    away_stats = snap.add_suffix("_away").rename(columns={"team_away": "away_team", "season_away": "season"})

    # merge on team full name (team_panel uses full names from API normalization)
    df = tg.copy()
    df = df.merge(home_stats, how="left", on=["home_team", "season"])
    df = df.merge(away_stats, how="left", on=["away_team", "season"], suffixes=("_home", "_away"))

    # rename rest/b2b fields to match training feature names
    df.rename(
        columns={
            "rest_days_today_home": "rest_days_home",
            "rest_days_today_away": "rest_days_away",
            "is_back_to_back_today_home": "is_back_to_back_home",
            "is_back_to_back_today_away": "is_back_to_back_away",
        },
        inplace=True,
    )

    # diffs
    def _safe_diff_today(a: str, b: str, out: str):
        if (a in df.columns) and (b in df.columns):
            df[out] = pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b], errors="coerce")

    _safe_diff_today("rolling_win_rate_10_home", "rolling_win_rate_10_away", "diff_rolling_win_rate_10")
    _safe_diff_today("rolling_goal_diff_10_home", "rolling_goal_diff_10_away", "diff_rolling_goal_diff_10")
    _safe_diff_today("season_win_rate_home", "season_win_rate_away", "diff_season_win_rate")
    _safe_diff_today("season_goal_diff_avg_home", "season_goal_diff_avg_away", "diff_season_goal_diff_avg")
    _safe_diff_today("elo_like_home", "elo_like_away", "diff_elo_like")
    _safe_diff_today("rolling_win_rate_5_home", "rolling_win_rate_5_away", "diff_rolling_win_rate_5")
    _safe_diff_today("rolling_goal_diff_5_home", "rolling_goal_diff_5_away", "diff_rolling_goal_diff_5")

    # feature cols (intersection with df)
    feature_cols: list[str] = [
        "rolling_win_rate_10_home",
        "rolling_win_rate_10_away",
        "rolling_goal_diff_10_home",
        "rolling_goal_diff_10_away",
        "rolling_goals_for_10_home",
        "rolling_goals_for_10_away",
        "rolling_goals_against_10_home",
        "rolling_goals_against_10_away",
        "rolling_home_ratio_10_home",
        "rolling_home_ratio_10_away",
        "season_win_rate_home",
        "season_win_rate_away",
        "season_goal_diff_avg_home",
        "season_goal_diff_avg_away",
        "rest_days_home",
        "rest_days_away",
        "is_back_to_back_home",
        "is_back_to_back_away",
        "elo_like_home",
        "elo_like_away",
        "rolling_elo_10_home",
        "rolling_elo_10_away",
        "rolling_elo_5_home",
        "rolling_elo_5_away",
        "rolling_win_rate_5_home",
        "rolling_win_rate_5_away",
        "rolling_goal_diff_5_home",
        "rolling_goal_diff_5_away",
        "rolling_goals_for_5_home",
        "rolling_goals_for_5_away",
        "rolling_goals_against_5_home",
        "rolling_goals_against_5_away",
        "rolling_home_ratio_5_home",
        "rolling_home_ratio_5_away",
        "diff_rolling_win_rate_10",
        "diff_rolling_goal_diff_10",
        "diff_season_win_rate",
        "diff_season_goal_diff_avg",
        "diff_elo_like",
        "diff_rolling_win_rate_5",
        "diff_rolling_goal_diff_5",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # final today df
    base_cols = ["game_id", "date", "season", "home_team", "away_team"]

    # keep abbr std to help odds merge later
    for c in ["home_team_abbr_std", "away_team_abbr_std"]:
        if c in df.columns:
            base_cols.append(c)

    today_df = df[base_cols + feature_cols].copy()

    # numeric cast + impute
    for c in feature_cols:
        today_df[c] = pd.to_numeric(today_df[c], errors="coerce")

    fill_map = {c: float(today_df[c].mean()) for c in feature_cols}
    today_df = today_df.fillna(fill_map).fillna(0.0)

    return today_df, feature_cols