# nhl_feature_engineering.py

from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
from datetime import date as _date


# ✅ NHL 버전 advanced stats (아직 없으면 stub으로라도 만들어두자)
try:
    from nhl_advanced_stats import build_team_advanced_stats_panel
except Exception:
    def build_team_advanced_stats_panel(_: pd.DataFrame) -> pd.DataFrame:
        # placeholder: later implement (e.g., xG, shot share, special teams, etc.)
        return pd.DataFrame()


# ✅ NHL 팀 abbr 표준화 (없으면 identity fallback)
try:
    from nhl_team_abbr import to_abbr
except Exception:
    def to_abbr(x):
        return x


def _add_elo_to_games(
    raw_games: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 20.0
) -> pd.DataFrame:
    """
    각 게임별 pre-game Elo rating (home/away)을 계산해서
    'elo_home', 'elo_away' 컬럼을 추가한다.

    NHL도 동일 로직으로 시작 (OT/SO는 일단 승/패만 반영)
    """
    df = raw_games.copy()

    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    ratings: dict[str, float] = {}
    elo_home_list = []
    elo_away_list = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

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
            Sh, Sa = 0.5, 0.5

        Eh = 1.0 / (1.0 + 10.0 ** ((Ra - Rh) / 400.0))
        Ea = 1.0 - Eh

        ratings[home] = Rh + k_factor * (Sh - Eh)
        ratings[away] = Ra + k_factor * (Sa - Ea)

    df["elo_home"] = elo_home_list
    df["elo_away"] = elo_away_list
    return df


def _build_team_level_panel(raw_games: pd.DataFrame) -> pd.DataFrame:
    df = raw_games.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # ✅ team_panel 만들기 전에 점수 없는 경기 제거
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    df = df.sort_values("date").reset_index(drop=True)

    has_elo = "elo_home" in df.columns and "elo_away" in df.columns

    # 1) 홈 팀 레코드
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

    # 2) 어웨이 팀 레코드
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

    # 3) team × game 패널
    team_panel = pd.concat([home_df, away_df], ignore_index=True)
    team_panel.sort_values(["team", "date"], inplace=True)

    team_panel["win"] = (team_panel["goals_for"] > team_panel["goals_against"]).astype(int)
    team_panel["goal_diff"] = team_panel["goals_for"] - team_panel["goals_against"]

    # 4) advanced stats merge (stub 가능)
    adv_panel = build_team_advanced_stats_panel(df)
    if adv_panel is not None and not adv_panel.empty:
        adv_tmp = adv_panel.copy()

        # team key normalize: 일단 to_abbr를 통해 표준화 가능하면 사용
        if "team" in adv_tmp.columns:
            adv_tmp["team"] = adv_tmp["team"].apply(to_abbr)
        if "team" in team_panel.columns:
            team_panel["team"] = team_panel["team"].apply(to_abbr)

        rating_cols = [c for c in ["off_rating", "def_rating", "net_rating"] if c in adv_tmp.columns]
        if rating_cols:
            if "game_id" in adv_tmp.columns:
                adv_tmp["game_id"] = pd.to_numeric(adv_tmp["game_id"], errors="coerce")
                team_panel["game_id"] = pd.to_numeric(team_panel["game_id"], errors="coerce")
                adv_tmp = adv_tmp[["game_id", "team"] + rating_cols].dropna(subset=["game_id", "team"]).drop_duplicates(["game_id", "team"])
                team_panel = team_panel.merge(adv_tmp, how="left", on=["game_id", "team"])
            else:
                adv_team = adv_tmp[["team"] + rating_cols].dropna(subset=["team"]).groupby("team", as_index=False)[rating_cols].mean()
                adv_team = adv_team.rename(columns={c: f"season_{c}" for c in rating_cols})
                team_panel = team_panel.merge(adv_team, how="left", on=["team"])
                for c in rating_cols:
                    if c not in team_panel.columns:
                        team_panel[c] = team_panel[f"season_{c}"]
                    else:
                        team_panel[c] = team_panel[c].fillna(team_panel[f"season_{c}"])

    # 5) rolling / season cumulative (PRE-GAME via shift(1))
    def _add_rolling(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()

        win_l1 = g["win"].shift(1)
        gdiff_l1 = g["goal_diff"].shift(1)
        gf_l1 = g["goals_for"].shift(1)
        ga_l1 = g["goals_against"].shift(1)
        is_home_l1 = g["is_home"].shift(1)

        g["rolling_win_rate_10"] = win_l1.rolling(10, min_periods=1).mean()
        g["rolling_goal_diff_10"] = gdiff_l1.rolling(10, min_periods=1).mean()
        g["rolling_goals_for_10"] = gf_l1.rolling(10, min_periods=1).mean()
        g["rolling_goals_against_10"] = ga_l1.rolling(10, min_periods=1).mean()
        g["rolling_home_ratio_10"] = is_home_l1.rolling(10, min_periods=1).mean()

        g["rolling_win_rate_5"] = win_l1.rolling(5, min_periods=1).mean()
        g["rolling_goal_diff_5"] = gdiff_l1.rolling(5, min_periods=1).mean()
        g["rolling_goals_for_5"] = gf_l1.rolling(5, min_periods=1).mean()
        g["rolling_goals_against_5"] = ga_l1.rolling(5, min_periods=1).mean()
        g["rolling_home_ratio_5"] = is_home_l1.rolling(5, min_periods=1).mean()

        g["season_games"] = g.groupby("season").cumcount()
        g["season_wins"] = g.groupby("season")["win"].shift(1).fillna(0).groupby(g["season"]).cumsum()
        g["season_win_rate"] = np.where(g["season_games"] > 0, g["season_wins"] / g["season_games"], np.nan)

        g["season_goal_diff_cum"] = g.groupby("season")["goal_diff"].shift(1).fillna(0).groupby(g["season"]).cumsum()
        g["season_goal_diff_avg"] = np.where(g["season_games"] > 0, g["season_goal_diff_cum"] / g["season_games"], np.nan)

        for col in ["off_rating", "def_rating", "net_rating"]:
            if col in g.columns:
                col_l1 = g[col].shift(1)
                g[f"rolling_{col}_10"] = col_l1.rolling(10, min_periods=1).mean()
                g[f"rolling_{col}_5"] = col_l1.rolling(5, min_periods=1).mean()

        return g

    team_panel = team_panel.groupby("team", group_keys=False).apply(_add_rolling)

    # 6) rest days, back-to-back
    team_panel["prev_date"] = team_panel.groupby("team")["date"].shift(1)
    team_panel["rest_days"] = (team_panel["date"] - team_panel["prev_date"]).dt.days
    team_panel["is_back_to_back"] = (team_panel["rest_days"] == 1).astype(int)

    # 6.5) 4-in-6 (PRE-GAME)
    def _add_4in6(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        dts = pd.to_datetime(g["date"]).values.astype("datetime64[D]")

        counts = np.zeros(len(g), dtype=int)
        j = 0
        for i in range(len(g)):
            start = dts[i] - np.timedelta64(5, "D")
            while dts[j] < start:
                j += 1
            counts[i] = i - j + 1

        g["games_in_last_6_days"] = pd.Series(counts, index=g.index).shift(1)
        g["is_4in6"] = (g["games_in_last_6_days"].fillna(0) >= 4).astype(int)
        return g

    team_panel = team_panel.groupby("team", group_keys=False).apply(_add_4in6)

    return team_panel


def build_training_dataset(raw_games: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    raw_games → game 단위 학습 데이터셋 (home 관점)
    """
    rg = raw_games.copy()
    if not np.issubdtype(rg["date"].dtype, np.datetime64):
        rg["date"] = pd.to_datetime(rg["date"], errors="coerce")

    rg = rg.dropna(subset=["home_score", "away_score"]).copy()

    rg = _add_elo_to_games(rg)
    team_panel = _build_team_level_panel(rg)

    df = rg.copy()

    # home merge
    home_stats = team_panel.add_suffix("_home").rename(
        columns={"team_home": "home_team", "date_home": "date", "game_id_home": "game_id", "season_home": "season"}
    )
    home_cols_base = [
        "game_id","home_team","date",
        "rolling_win_rate_10_home","rolling_goal_diff_10_home",
        "season_win_rate_home","season_goal_diff_avg_home",
        "rest_days_home","is_back_to_back_home","elo_like_home",
        "season_games_home",
        "rolling_win_rate_5_home","rolling_goal_diff_5_home",
        "is_4in6_home","games_in_last_6_days_home",
    ]
    # optional advanced
    for c in ["rolling_off_rating_10_home","rolling_def_rating_10_home","rolling_net_rating_10_home"]:
        if c in home_stats.columns:
            home_cols_base.append(c)

    home_cols = [c for c in home_cols_base if c in home_stats.columns]
    df = df.merge(home_stats[home_cols], on=["game_id", "home_team", "date"], how="left")

    # away merge
    away_stats = team_panel.add_suffix("_away").rename(
        columns={"team_away": "away_team", "date_away": "date", "game_id_away": "game_id", "season_away": "season"}
    )
    away_cols_base = [
        "game_id","away_team","date",
        "rolling_win_rate_10_away","rolling_goal_diff_10_away",
        "season_win_rate_away","season_goal_diff_avg_away",
        "rest_days_away","is_back_to_back_away","elo_like_away",
        "season_games_away",
        "rolling_win_rate_5_away","rolling_goal_diff_5_away",
        "is_4in6_away","games_in_last_6_days_away",
    ]
    for c in ["rolling_off_rating_10_away","rolling_def_rating_10_away","rolling_net_rating_10_away"]:
        if c in away_stats.columns:
            away_cols_base.append(c)

    away_cols = [c for c in away_cols_base if c in away_stats.columns]
    df = df.merge(away_stats[away_cols], on=["game_id", "away_team", "date"], how="left")

    # target
    df["target"] = (df["home_score"] > df["away_score"]).astype(int)

    # diffs
    df["diff_rolling_win_rate_10"] = df["rolling_win_rate_10_home"] - df["rolling_win_rate_10_away"]
    df["diff_rolling_goal_diff_10"] = df["rolling_goal_diff_10_home"] - df["rolling_goal_diff_10_away"]
    df["diff_season_win_rate"] = df["season_win_rate_home"] - df["season_win_rate_away"]
    df["diff_season_goal_diff_avg"] = df["season_goal_diff_avg_home"] - df["season_goal_diff_avg_away"]
    df["diff_elo_like"] = df["elo_like_home"] - df["elo_like_away"]
    df["diff_rolling_win_rate_5"] = df["rolling_win_rate_5_home"] - df["rolling_win_rate_5_away"]
    df["diff_rolling_goal_diff_5"] = df["rolling_goal_diff_5_home"] - df["rolling_goal_diff_5_away"]
    df["diff_is_4in6"] = df["is_4in6_home"] - df["is_4in6_away"]

    # optional advanced diffs
    if "rolling_net_rating_10_home" in df.columns and "rolling_net_rating_10_away" in df.columns:
        df["diff_rolling_net_rating_10"] = df["rolling_net_rating_10_home"] - df["rolling_net_rating_10_away"]

    # feature cols
    feature_cols = [
        "rolling_win_rate_10_home","rolling_win_rate_10_away",
        "rolling_goal_diff_10_home","rolling_goal_diff_10_away",
        "season_win_rate_home","season_win_rate_away",
        "season_goal_diff_avg_home","season_goal_diff_avg_away",
        "rest_days_home","rest_days_away",
        "is_back_to_back_home","is_back_to_back_away",
        "elo_like_home","elo_like_away",
        "diff_rolling_win_rate_10","diff_rolling_goal_diff_10",
        "diff_season_win_rate","diff_season_goal_diff_avg",
        "diff_elo_like",
        "rolling_win_rate_5_home","rolling_win_rate_5_away",
        "rolling_goal_diff_5_home","rolling_goal_diff_5_away",
        "diff_rolling_win_rate_5","diff_rolling_goal_diff_5",
        "is_4in6_home","is_4in6_away","diff_is_4in6",
    ]
    # optional advanced
    for c in ["rolling_off_rating_10_home","rolling_off_rating_10_away","rolling_def_rating_10_home","rolling_def_rating_10_away","rolling_net_rating_10_home","rolling_net_rating_10_away","diff_rolling_net_rating_10"]:
        if c in df.columns:
            feature_cols.append(c)

    feature_cols = [c for c in feature_cols if c in df.columns]

    train_df = df[
        ["game_id","date","season","home_team","away_team","home_score","away_score","target",
         "season_games_home","season_games_away"]
        + feature_cols
    ].copy()

    for c in feature_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")

    fill_map = {c: float(train_df[c].mean()) for c in feature_cols}
    train_df = train_df.fillna(fill_map)

    train_df["date"] = pd.to_datetime(train_df["date"]).dt.date
    return train_df, feature_cols


def build_today_dataset(
    raw_games: pd.DataFrame,
    today_games: pd.DataFrame,
    today: _date,
):
    if raw_games.empty or today_games.empty:
        return pd.DataFrame(), []

    tg = today_games.copy()

    if "game_id" not in tg.columns and "id" in tg.columns:
        tg["game_id"] = tg["id"]

    if "date" in tg.columns:
        tg["date"] = pd.to_datetime(tg["date"], errors="coerce").dt.date
    else:
        tg["date"] = today

    # normalize away team key (away_team vs visitor_team)
    def _extract_full_name(x):
        if isinstance(x, dict):
            return x.get("full_name")
        return x

    if "home_team" in tg.columns:
        tg["home_team"] = tg["home_team"].apply(_extract_full_name)

    if "away_team" not in tg.columns and "visitor_team" in tg.columns:
        tg["away_team"] = tg["visitor_team"].apply(_extract_full_name)
    elif "away_team" in tg.columns:
        tg["away_team"] = tg["away_team"].apply(_extract_full_name)

    rg = raw_games.copy()
    if not np.issubdtype(rg["date"].dtype, np.datetime64):
        rg["date"] = pd.to_datetime(rg["date"], errors="coerce")

    rg = rg.dropna(subset=["home_score", "away_score"]).copy()
    rg = _add_elo_to_games(rg)

    if "season" not in tg.columns:
        tg["season"] = rg["season"].max()

    latest_season = rg["season"].max()
    tg["season"] = tg["season"].apply(lambda s: latest_season if s > latest_season else s)

    team_panel = _build_team_level_panel(rg)
    team_panel = team_panel.sort_values(["team", "date"])
    last_rows = team_panel.groupby("team").tail(1).copy()
    last_rows.rename(columns={"date": "last_game_date"}, inplace=True)

    cutoff = (pd.to_datetime(today) - pd.Timedelta(days=5)).normalize()
    tmp6 = team_panel[pd.to_datetime(team_panel["date"]) >= cutoff].copy()
    games6 = tmp6.groupby("team")["game_id"].count()

    last_rows["games_in_last_6_days_today"] = last_rows["team"].map(games6).fillna(0).astype(int)
    last_rows["is_4in6_today"] = (last_rows["games_in_last_6_days_today"] >= 4).astype(int)

    last_rows["games_in_last_6_days"] = last_rows["games_in_last_6_days_today"]
    last_rows["is_4in6"] = last_rows["is_4in6_today"]

    last_rows["rest_days_today"] = (pd.to_datetime(today) - last_rows["last_game_date"]).dt.days
    last_rows["is_back_to_back_today"] = (last_rows["rest_days_today"] == 1).astype(int)

    base_keep_cols = [
        "team","season",
        "rolling_win_rate_10","rolling_goal_diff_10",
        "season_win_rate","season_goal_diff_avg",
        "elo_like",
        "rest_days_today","is_back_to_back_today",
        "rolling_win_rate_5","rolling_goal_diff_5",
        "is_4in6","games_in_last_6_days",
    ]
    # optional advanced snapshots
    for c in ["rolling_off_rating_10","rolling_def_rating_10","rolling_net_rating_10"]:
        if c in last_rows.columns:
            base_keep_cols.append(c)

    keep_cols = [c for c in base_keep_cols if c in last_rows.columns]
    snap = last_rows[keep_cols].copy()

    home_stats = snap.add_suffix("_home").rename(columns={"team_home": "home_team", "season_home": "season"})
    away_stats = snap.add_suffix("_away").rename(columns={"team_away": "away_team", "season_away": "season"})

    df = tg.copy()
    df = df.merge(home_stats, how="left", on=["home_team", "season"])
    df = df.merge(away_stats, how="left", on=["away_team", "season"], suffixes=("_home", "_away"))

    df["diff_rolling_win_rate_10"] = df["rolling_win_rate_10_home"] - df["rolling_win_rate_10_away"]
    df["diff_rolling_goal_diff_10"] = df["rolling_goal_diff_10_home"] - df["rolling_goal_diff_10_away"]
    df["diff_season_win_rate"] = df["season_win_rate_home"] - df["season_win_rate_away"]
    df["diff_season_goal_diff_avg"] = df["season_goal_diff_avg_home"] - df["season_goal_diff_avg_away"]
    df["diff_elo_like"] = df["elo_like_home"] - df["elo_like_away"]
    df["diff_rolling_win_rate_5"] = df["rolling_win_rate_5_home"] - df["rolling_win_rate_5_away"]
    df["diff_rolling_goal_diff_5"] = df["rolling_goal_diff_5_home"] - df["rolling_goal_diff_5_away"]
    df["diff_is_4in6"] = df["is_4in6_home"] - df["is_4in6_away"]

    df.rename(
        columns={
            "rest_days_today_home": "rest_days_home",
            "rest_days_today_away": "rest_days_away",
            "is_back_to_back_today_home": "is_back_to_back_home",
            "is_back_to_back_today_away": "is_back_to_back_away",
        },
        inplace=True,
    )

    feature_cols = [
        "rolling_win_rate_10_home","rolling_win_rate_10_away",
        "rolling_goal_diff_10_home","rolling_goal_diff_10_away",
        "season_win_rate_home","season_win_rate_away",
        "season_goal_diff_avg_home","season_goal_diff_avg_away",
        "rest_days_home","rest_days_away",
        "is_back_to_back_home","is_back_to_back_away",
        "elo_like_home","elo_like_away",
        "diff_rolling_win_rate_10","diff_rolling_goal_diff_10",
        "diff_season_win_rate","diff_season_goal_diff_avg",
        "diff_elo_like",
        "rolling_win_rate_5_home","rolling_win_rate_5_away",
        "rolling_goal_diff_5_home","rolling_goal_diff_5_away",
        "diff_rolling_win_rate_5","diff_rolling_goal_diff_5",
        "is_4in6_home","is_4in6_away","diff_is_4in6",
    ]
    for c in ["rolling_off_rating_10_home","rolling_off_rating_10_away","rolling_def_rating_10_home","rolling_def_rating_10_away","rolling_net_rating_10_home","rolling_net_rating_10_away"]:
        if c in df.columns:
            feature_cols.append(c)

    feature_cols = [c for c in feature_cols if c in df.columns]

    base_cols = ["game_id","date","season","home_team","away_team"]
    today_df = df[base_cols + feature_cols].copy()

    for c in feature_cols:
        today_df[c] = pd.to_numeric(today_df[c], errors="coerce")

    fill_map = {c: float(today_df[c].mean()) for c in feature_cols}
    today_df = today_df.fillna(fill_map).fillna(0.0)

    return today_df, feature_cols


