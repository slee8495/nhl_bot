# nhl_quant_main_next_day.py

from __future__ import annotations

import pandas as pd
import numpy as np

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from nhl_api_client import NHLDataClient
from nhl_feature_engineering import build_today_dataset
from nhl_quant_model import NHLQuantModel
from nhl_goal_diff_model import NHLGoalDiffModel
from odds_client import OddsAPIClient

from nhl_edge import add_edge_from_odds, filter_bets
from nhl_performance import update_bet_log_and_summary
from nhl_strategy_tuning import tune_strategy_from_log
from email_management import send_nhl_daily_report

try:
    from nhl_injury_lineup import attach_lineup_features, adjust_prob_with_lineup
    HAS_LINEUP = True
except ModuleNotFoundError:
    HAS_LINEUP = False
    attach_lineup_features = None
    adjust_prob_with_lineup = None
    print("[NHL NEXT] nhl_injury_lineup not found -> lineup/injury adjustments disabled.")






def _add_ensemble_prob(scored_games: pd.DataFrame) -> pd.DataFrame:
    df = scored_games.copy()

    if "p_home_win_adj" in df.columns:
        p_xgb = df["p_home_win_adj"].astype(float)
    else:
        p_xgb = df["p_home_win"].astype(float)

    if "p_home_win_from_margin" in df.columns:
        p_gd = df["p_home_win_from_margin"].astype(float)
    elif "p_home_win_from_goal_diff" in df.columns:
        p_gd = df["p_home_win_from_goal_diff"].astype(float)
    else:
        p_gd = p_xgb

    if "diff_elo_like" in df.columns:
        diff_elo = df["diff_elo_like"].astype(float)
        p_elo = 1.0 / (1.0 + 10.0 ** (-diff_elo / 400.0))
    else:
        p_elo = p_xgb

    w_xgb, w_gd, w_elo = 0.55, 0.30, 0.15
    total = w_xgb + w_gd + w_elo
    w_xgb, w_gd, w_elo = w_xgb / total, w_gd / total, w_elo / total

    df["p_home_win_xgb"] = p_xgb
    df["p_home_win_elo"] = p_elo
    df["p_home_win_gd"] = p_gd

    p_ens = w_xgb * p_xgb + w_gd * p_gd + w_elo * p_elo
    p_ens = np.clip(p_ens, 0.01, 0.99)

    df["p_home_win_ensemble"] = p_ens
    df["p_home_win"] = df["p_home_win_ensemble"]

    if "p_home_win_adj" in df.columns:
        df["p_home_win_adj"] = df["p_home_win"]

    return df


def _pick_datetime_col(df: pd.DataFrame) -> str | None:
    for c in ["start_time", "commence_time", "scheduled", "game_datetime", "datetime", "date"]:
        if c in df.columns:
            return c
    return None


def _filter_games_by_pt_slate_date(sched_df: pd.DataFrame, slate_pt_date: date, tz_pt: ZoneInfo) -> pd.DataFrame:
    df = sched_df.copy()
    time_col = _pick_datetime_col(df)
    if time_col is None:
        raise ValueError("[NHL NEXT] Schedule missing datetime column.")

    df["_dt_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["slate_date"] = df["_dt_utc"].dt.tz_convert(tz_pt).dt.date

    out = df[df["slate_date"] == slate_pt_date].copy()
    out.drop(columns=["_dt_utc"], inplace=True, errors="ignore")
    return out


def _filter_odds_by_pt_slate_date(odds_df_all: pd.DataFrame, slate_pt_date: date, tz_pt: ZoneInfo) -> pd.DataFrame:
    if odds_df_all is None or odds_df_all.empty:
        return pd.DataFrame()

    df = odds_df_all.copy()
    time_col = None
    for c in ["start_time", "commence_time", "date"]:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        return pd.DataFrame()

    df["_dt_utc"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df["_pt_date"] = df["_dt_utc"].dt.tz_convert(tz_pt).dt.date

    out = df[df["_pt_date"] == slate_pt_date].copy()
    out["date"] = slate_pt_date
    out.drop(columns=["_dt_utc", "_pt_date"], inplace=True, errors="ignore")
    return out


def main():
    tz_pt = ZoneInfo("America/Los_Angeles")

    # ✅ PT 기준 "내일" 슬레이트
    pt_today = datetime.now(tz_pt).date()
    slate_date_pt = pt_today + timedelta(days=1)

    print(f"[NHL NEXT] Running (PT) for slate_date={slate_date_pt}")

    client = NHLDataClient()

    # (1) Past games (최근 1 시즌)
    seasons = [pt_today.year - 1]
    raw_games = client.fetch_games_by_season(seasons)
    if raw_games is None or raw_games.empty:
        print("[NHL NEXT] No historical games fetched. Abort.")
        return

    
    raw_games["date"] = pd.to_datetime(raw_games["date"], errors="coerce").dt.normalize()
    slate_ts = pd.Timestamp(slate_date_pt)  # 00:00:00
    raw_games_past = raw_games[raw_games["date"] < slate_ts].copy()


    if raw_games_past.empty:
        print("[NHL NEXT] No past games before slate_date. Abort.")
        return

    # (2) Schedule 확정 (내일 PT)
    start_date = slate_date_pt - timedelta(days=3)
    end_date = slate_date_pt + timedelta(days=3)

    sched_df = client.fetch_games_by_date_range_df(
        start_date=start_date,
        end_date=end_date,
        per_page=200,
    )
    if sched_df is None or sched_df.empty:
        print(f"[NHL NEXT] No scheduled games for {start_date}~{end_date}. Abort.")
        return

    tomorrow_games = _filter_games_by_pt_slate_date(sched_df, slate_date_pt, tz_pt)
    if tomorrow_games.empty:
        print(f"[NHL NEXT] No games found for PT date={slate_date_pt}. Abort.")
        return

    # (3) Odds (내일 PT)
    odds_client = OddsAPIClient()
    odds_df_all = odds_client.fetch_nhl_moneyline_odds(regions="us", attach_game_id=True)
    odds_df = _filter_odds_by_pt_slate_date(odds_df_all, slate_date_pt, tz_pt)

    if odds_df.empty:
        print("[NHL NEXT] Odds empty or not matched to PT slate_date. Abort.")
        return

    # ✅ odds game_id로 slate 확정
    if "game_id" in odds_df.columns:
        odds_game_ids = odds_df["game_id"].dropna().astype(int).unique().tolist()
        if "game_id" in tomorrow_games.columns:
            tomorrow_games = tomorrow_games[tomorrow_games["game_id"].astype(int).isin(odds_game_ids)].copy()

        print(f"[SLATE FIX][NHL NEXT] odds_game_ids={len(odds_game_ids)} | schedule_rows(after odds filter)={len(tomorrow_games)}")

        if tomorrow_games.empty:
            print("[SLATE FIX][NHL NEXT] No schedule rows matched odds game_id. Expand schedule date range.")
            return
    else:
        print("[NHL NEXT] Odds missing game_id. Abort.")
        return

    print(f"[NHL NEXT] Final slate(PT)={slate_date_pt} | games={tomorrow_games['game_id'].nunique()} | odds_rows={len(odds_df)}")

    # (4) Build features
    today_df, feature_cols = build_today_dataset(
        raw_games_past=raw_games_past,
        today_games=tomorrow_games,
        today=slate_date_pt,
    )
    if today_df is None or today_df.empty:
        print("[NHL NEXT] No valid feature rows. Abort.")
        return

    # (5) Predict
    win_model = NHLQuantModel.load_from_disk()
    scored_games = win_model.predict_proba(today_df)
    scored_games["date"] = slate_date_pt

    # ✅ (5.1) Lineup/Injury features + prob adjust (DROP-IN)
    if HAS_LINEUP:
        try:
            scored_games = attach_lineup_features(scored_games, today=slate_date_pt)
            scored_games = adjust_prob_with_lineup(scored_games)
        except Exception as e:
            print(f"[NHL NEXT] lineup adjust skipped: {e}")





    try:
        gd_model = NHLGoalDiffModel.load_from_disk()
        gd_df = gd_model.predict_goal_diff(today_df)

        merge_cols = ["game_id", "goal_diff_pred", "p_home_win_from_goal_diff", "p_home_win_from_margin"]
        merge_cols = [c for c in merge_cols if c in gd_df.columns]
        if merge_cols:
            scored_games = scored_games.merge(gd_df[merge_cols], on="game_id", how="left")
    except Exception as e:
        print(f"[NHL NEXT] GoalDiff model failed or missing: {e}")

    try:
        scored_games = _add_ensemble_prob(scored_games)
    except Exception as e:
        print(f"[NHL NEXT] Ensemble failed: {e}")

    # (6) Edge + filter
    edge_df = add_edge_from_odds(
        scored_games=scored_games,
        odds_df=odds_df,
        preferred_bookmaker=None,
        market_blend_alpha=0.7,
        odds_history_df=None,
    )
    if edge_df is None or edge_df.empty:
        print("[NHL NEXT] Could not match any odds to games. Abort.")
        return

    reco_df = filter_bets(
        edge_df,
        min_edge=0.03,
        use_bucket_rules=True,
        use_strategy_config=True,
    )

    # (7) Performance + tune (내일 슬레이트는 아직 결과가 없으니 로그 업데이트는 선택)
    # - 일반적으로 "NEXT_DAY"는 추천만 보내고, performance 업데이트는 "당일/다음날 settle 후"에 함
    perf_summary = None
    try:
        tune_strategy_from_log()
    except Exception as e:
        print(f"[NHL NEXT] strategy tuning failed: {e}")

    # (8) Email
    send_nhl_daily_report(
        reco_df=reco_df,
        full_df=edge_df,
        perf_summary=perf_summary,
        subject_prefix="[NEXT DAY]",
    )


if __name__ == "__main__":
    main()