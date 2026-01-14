# nhl_quant_main.py

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
    print("[NHL MAIN] nhl_injury_lineup not found -> lineup/injury adjustments disabled.")





# -----------------------------
# Ensemble (optional but recommended)
# -----------------------------
def _add_ensemble_prob(scored_games: pd.DataFrame) -> pd.DataFrame:
    """
    NHL 버전 앙상블:
      - XGB prob (p_home_win_adj 있으면 우선)
      - goal diff model -> p_home_win_from_margin(=from_goal_diff) 있으면 섞기
      - Elo 컬럼(diff_elo_like) 있으면 섞기
    최종은 p_home_win에 덮어쓰기
    """
    df = scored_games.copy()

    # 1) XGB
    if "p_home_win_adj" in df.columns:
        p_xgb = df["p_home_win_adj"].astype(float)
    else:
        p_xgb = df["p_home_win"].astype(float)

    # 2) goal diff 기반 prob
    if "p_home_win_from_margin" in df.columns:
        p_gd = df["p_home_win_from_margin"].astype(float)
    elif "p_home_win_from_goal_diff" in df.columns:
        p_gd = df["p_home_win_from_goal_diff"].astype(float)
    else:
        p_gd = p_xgb

    # 3) Elo
    if "diff_elo_like" in df.columns:
        diff_elo = df["diff_elo_like"].astype(float)
        p_elo = 1.0 / (1.0 + 10.0 ** (-diff_elo / 400.0))
    else:
        p_elo = p_xgb

    # weights (초기값)
    w_xgb = 0.55
    w_gd = 0.30
    w_elo = 0.15
    total = w_xgb + w_gd + w_elo
    w_xgb, w_gd, w_elo = w_xgb / total, w_gd / total, w_elo / total

    df["p_home_win_xgb"] = p_xgb
    df["p_home_win_elo"] = p_elo
    df["p_home_win_gd"] = p_gd

    p_ens = w_xgb * p_xgb + w_gd * p_gd + w_elo * p_elo
    p_ens = np.clip(p_ens, 0.01, 0.99)

    df["p_home_win_ensemble"] = p_ens
    df["p_home_win"] = df["p_home_win_ensemble"]

    # downstream이 p_home_win_adj를 쓰는 경우도 있으니 덮어쓰기
    if "p_home_win_adj" in df.columns:
        df["p_home_win_adj"] = df["p_home_win"]

    return df


def _pick_datetime_col(df: pd.DataFrame) -> str | None:
    # ✅ slate_date 계산용: "시간이 있는" 컬럼만 허용
    preferred = [
        "start_time_utc",
        "start_time",
        "commence_time",
        "scheduled",
        "game_datetime",
        "datetime",
    ]
    for c in preferred:
        if c in df.columns:
            return c
    return None


def _filter_games_by_pt_slate_date(sched_df: pd.DataFrame, slate_pt_date: date, tz_pt: ZoneInfo) -> pd.DataFrame:
    df = sched_df.copy()
    time_col = _pick_datetime_col(df)
    if time_col is None:
        raise ValueError("[NHL MAIN] Schedule missing datetime column.")

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

def _log_lineup_summary(df: pd.DataFrame, top_n: int = 50):
    if df is None or df.empty:
        return

    need = [
        "game_id","home_team_abbr","away_team_abbr",
        "p_home_win","p_home_win_adj",
        "z_lineup_depth","z_star_penalty","z_goalie_penalty","z_lineup_total",
        "goalie_proxy_name_home","goalie_proxy_name_away",
        "missing_star_names_home","missing_star_names_away",
        "out_names_preview_home","out_names_preview_away",
        "goalie_out_flag_home","goalie_out_flag_away",
    ]
    cols = [c for c in need if c in df.columns]

    print("[NHL LINEUP SUMMARY] per-game snapshot")
    for _, r in df[cols].head(top_n).iterrows():
        gid = int(r.get("game_id")) if pd.notna(r.get("game_id")) else -1
        away = r.get("away_team_abbr") or r.get("away_team") or "AWAY"
        home = r.get("home_team_abbr") or r.get("home_team") or "HOME"

        p = float(r.get("p_home_win")) if pd.notna(r.get("p_home_win")) else float("nan")
        pa = float(r.get("p_home_win_adj")) if pd.notna(r.get("p_home_win_adj")) else p

        z_d = float(r.get("z_lineup_depth", 0.0) or 0.0)
        z_s = float(r.get("z_star_penalty", 0.0) or 0.0)
        z_g = float(r.get("z_goalie_penalty", 0.0) or 0.0)
        z_t = float(r.get("z_lineup_total", 0.0) or 0.0)

        g_home = r.get("goalie_proxy_name_home")
        g_away = r.get("goalie_proxy_name_away")

        so_home = r.get("missing_star_names_home")
        so_away = r.get("missing_star_names_away")

        def _safe_int_log(x, default=0):
            v = pd.to_numeric(x, errors="coerce")
            return int(v) if pd.notna(v) else int(default)

        goh = _safe_int_log(r.get("goalie_out_flag_home", 0), 0)
        goa = _safe_int_log(r.get("goalie_out_flag_away", 0), 0)

        print(
            f"[GAME {gid}] {away}@{home} | p={p:.3f} adj={pa:.3f} | "
            f"z(d/s/g/t)=({z_d:+.3f}/{z_s:+.3f}/{z_g:+.3f}/{z_t:+.3f}) | "
            f"G(proxy) A={g_away} H={g_home} (out A/H={goa}/{goh}) | "
            f"StarOut A={so_away} H={so_home}"
        )


def main():
    tz_pt = ZoneInfo("America/Los_Angeles")
    today = datetime.now(tz_pt).date()
    print(f"[NHL MAIN] Running (PT) for date={today}")

    client = NHLDataClient()

    # =========================
    # (1) Past games (feature base)
    # =========================
    seasons = [today.year - 1]  # 운영용: 최근 1 시즌
    raw_games = client.fetch_games_by_season(seasons)
    if raw_games is None or raw_games.empty:
        print("[NHL MAIN] No historical games fetched. Abort.")
        return

    # ✅ NHL past-games 날짜 표준화 (UTC->naive normalize)
    time_col_hist = None
    for c in ["start_time_utc", "start_time", "commence_time", "scheduled", "game_datetime", "datetime", "date", "game_date"]:
        if c in raw_games.columns:
            time_col_hist = c
            break
    if time_col_hist is None:
        raise ValueError("[NHL MAIN] raw_games missing datetime column.")

    raw_games["date"] = (
        pd.to_datetime(raw_games[time_col_hist], utc=True, errors="coerce")
          .dt.tz_convert(None)
          .dt.normalize()
    )

    raw_games_past = raw_games[raw_games["date"] < pd.Timestamp(today)].copy()
    if raw_games_past.empty:
        print("[NHL MAIN] No past games before today. Abort.")
        return

    # =========================
    # ✅ SLATE 선택 (NBA MAIN 방식 그대로)
    # - schedule에서 PT '오늘' 경기만
    # - odds도 PT '오늘'로만 필터
    # - 마지막에 odds game_id로 slate 확정
    # =========================

    # (A) schedule 먼저 확정 (UTC/PT 날짜 경계 때문에 ±3일)
    start_date = today - timedelta(days=3)
    end_date   = today + timedelta(days=3)

    sched_df = client.fetch_games_by_date_range_df(
        start_date=start_date,
        end_date=end_date,
        per_page=200,
    )
    if sched_df is None or sched_df.empty:
        print(f"[NHL MAIN] No scheduled games for {start_date}~{end_date}. Abort.")
        return

    sched_df = sched_df.copy()

    # ✅ schedule에서 실제 시작시간 컬럼 찾기 (NHL은 start_time_utc가 있는 경우가 많음)
    time_col_sched = _pick_datetime_col(sched_df)
    if time_col_sched is None:
        raise ValueError("[NHL MAIN] Schedule missing datetime-with-time column (start_time_utc/start_time/etc).")

    print("[SCHED DEBUG] time_col picked:", time_col_sched)
    print("[SCHED DEBUG] columns:", list(sched_df.columns))
    print("[SCHED DEBUG] time sample:", sched_df[time_col_sched].head(3).tolist() if time_col_sched else None)

    if time_col_sched is None:
        print("[NHL MAIN] Schedule missing datetime column. Abort.")
        return

    # ✅ 핵심: schedule 시간을 UTC로 파싱 → PT로 변환 → slate_date
    sched_df["_dt_utc"] = pd.to_datetime(sched_df[time_col_sched], utc=True, errors="coerce")
    sched_df["slate_date"] = sched_df["_dt_utc"].dt.tz_convert(tz_pt).dt.date

    print("[SCHED DEBUG] slate_date counts (top):")
    print(sched_df["slate_date"].value_counts().head(5))

    today_games = sched_df[sched_df["slate_date"] == today].copy()
    if today_games.empty:
        print(f"[NHL MAIN] No games found for PT date={today}. Abort.")
        return

    # (B) odds는 "PT 오늘 하루"만 정확히 필터
    odds_client = OddsAPIClient()
    odds_df_all = odds_client.fetch_nhl_moneyline_odds(regions="us", attach_game_id=True)

    if odds_df_all is None or odds_df_all.empty:
        print("[NHL MAIN] No odds from API. Abort.")
        return

    odds_df_all = odds_df_all.copy()

    time_col_odds = None
    for c in ["start_time", "commence_time"]:  # ✅ date 제외
        if c in odds_df_all.columns:
            time_col_odds = c
            break
    if time_col_odds is None:
        raise ValueError("[NHL MAIN] Odds missing start_time/commence_time (datetime-with-time).")

    odds_df_all["_dt_utc"] = pd.to_datetime(odds_df_all[time_col_odds], utc=True, errors="coerce")
    odds_df_all["_pt_date"] = odds_df_all["_dt_utc"].dt.tz_convert(tz_pt).dt.date

    odds_df = odds_df_all[odds_df_all["_pt_date"] == today].copy()
    odds_df["date"] = today
    odds_df.drop(columns=["_dt_utc", "_pt_date"], inplace=True, errors="ignore")

    if odds_df.empty:
        print("[NHL MAIN] Odds exist but none matched PT today. Abort.")
        return

    # ✅✅✅ 핵심: odds에 붙은 game_id로 slate 최종 확정 (NBA와 동일)
    if "game_id" in odds_df.columns:
        odds_game_ids = odds_df["game_id"].dropna().astype(int).unique().tolist()

        # 오늘경기(today_games)로 확정하지 말고, odds game_id로 확정
        today_games = today_games[today_games["game_id"].astype(int).isin(odds_game_ids)].copy()

        print(f"[SLATE FIX][NHL] odds_game_ids={len(odds_game_ids)} | today_games(after odds filter)={len(today_games)}")

        if today_games.empty:
            print("[SLATE FIX][NHL] No schedule rows matched odds game_id. Expand schedule date range.")
            return
    else:
        print("[SLATE FIX][NHL] Odds missing game_id. Abort.")
        return

    print(f"[NHL MAIN] Final slate(PT)={today} | games={today_games['game_id'].nunique()} | odds_rows={len(odds_df)}")

    # =========================
    # (4) Build today features
    # =========================
    today_df, feature_cols = build_today_dataset(
        raw_games_past,
        today_games,
        today,
    )
    if today_df is None or today_df.empty:
        print("[NHL MAIN] No valid feature rows for today. Abort.")
        return



    # =========================
    # (5) Predict
    # =========================
    win_model = NHLQuantModel.load_from_disk()
    scored_games = win_model.predict_proba(today_df)

    # ✅ schedule(today_games)에서 팀 메타를 scored_games에 강제 주입
    need = ["game_id", "season", "home_team_id", "away_team_id", "home_team", "away_team", "home_team_abbr", "away_team_abbr"]
    have = [c for c in need if c in today_games.columns]

    # (A) scored_games에 같은 이름 컬럼이 이미 있으면 suffix 문제 생김 → 먼저 제거
    dup_cols = [c for c in need if c != "game_id" and c in scored_games.columns]
    if dup_cols:
        scored_games = scored_games.drop(columns=dup_cols, errors="ignore")

    # (B) merge
    if have:
        meta = today_games[have].drop_duplicates(subset=["game_id"]).copy()
        scored_games["game_id"] = pd.to_numeric(scored_games["game_id"], errors="coerce")
        meta["game_id"] = pd.to_numeric(meta["game_id"], errors="coerce")
        scored_games = scored_games.merge(meta, on="game_id", how="left")

    # (C) 최소 필수 컬럼 보장 (edge/lineup이 필요)
    if "home_team" not in scored_games.columns or "away_team" not in scored_games.columns:
        # suffix가 생긴 경우까지 커버 (혹시 남아있다면)
        for side in ["home_team", "away_team"]:
            if side not in scored_games.columns:
                if f"{side}_y" in scored_games.columns:
                    scored_games[side] = scored_games[f"{side}_y"]
                elif f"{side}_x" in scored_games.columns:
                    scored_games[side] = scored_games[f"{side}_x"]

    if "home_team" not in scored_games.columns or "away_team" not in scored_games.columns:
        raise RuntimeError("[NHL MAIN] scored_games missing home_team/away_team after schedule meta join.")
    
    





    # ✅ downstream merge 안정화: 오늘 날짜로 고정
    scored_games["date"] = today

    # ✅ (5.1) Lineup/Injury features + prob adjust
    if HAS_LINEUP:
        try:
            scored_games = attach_lineup_features(scored_games, today=today)
            scored_games = adjust_prob_with_lineup(scored_games)

            # ✅ NEW: per-game lineup summary to GitHub Actions logs
            _log_lineup_summary(scored_games)

        except Exception as e:
            print(f"[NHL MAIN] lineup adjust skipped: {e}")

    # ✅ lineup adjust 후에도 날짜 다시 고정 (안전)
    scored_games["date"] = today

    # (5.2) GoalDiff 모델 merge
    try:
        gd_model = NHLGoalDiffModel.load_from_disk()
        gd_df = gd_model.predict_goal_diff(today_df)

        merge_cols = ["game_id", "goal_diff_pred", "p_home_win_from_goal_diff", "p_home_win_from_margin"]
        merge_cols = [c for c in merge_cols if c in gd_df.columns]

        if merge_cols:
            scored_games = scored_games.merge(gd_df[merge_cols], on="game_id", how="left")
    except Exception as e:
        print(f"[NHL MAIN] GoalDiff model failed or missing: {e}")

    # (5.3) Ensemble
    try:
        scored_games = _add_ensemble_prob(scored_games)
    except Exception as e:
        print(f"[NHL MAIN] Ensemble failed: {e}")

    # =========================
    # (6) Edge + filter
    # =========================
    edge_df = add_edge_from_odds(
        scored_games=scored_games,
        odds_df=odds_df,
        preferred_bookmaker=None,
        market_blend_alpha=0.7,
        odds_history_df=None,
    )

    if edge_df is None or edge_df.empty:
        print("[NHL MAIN] Could not match any odds to games. Abort.")
        return

    reco_df = filter_bets(
        edge_df,
        min_edge=0.03,
        use_bucket_rules=True,
        use_strategy_config=True,
    )

    # =========================
    # (7) Performance log + tune
    # =========================
    try:
        _, perf_summary = update_bet_log_and_summary(
            reco_df=reco_df,
            raw_games_past=raw_games_past,
            today=today,              # ✅ 여기 today
            stake_per_bet=100.0,
        )
    except Exception as e:
        print(f"[NHL MAIN] performance update failed: {e}")
        perf_summary = None

    try:
        tune_strategy_from_log()
    except Exception as e:
        print(f"[NHL MAIN] strategy tuning failed: {e}")

    # =========================
    # (8) Email
    # =========================
    send_nhl_daily_report(
        reco_df=reco_df,
        full_df=edge_df,
        perf_summary=perf_summary,
    )


if __name__ == "__main__":
    main()
