# nhl_performance.py

from __future__ import annotations

from datetime import date as _date
import os

import numpy as np
import pandas as pd

from config import DATA_DIR

BET_LOG_PATH = os.path.join(DATA_DIR, "nhl_bet_log.csv")


def _american_payout(stake: float, odds: float) -> float:
    """
    미국식 오즈 기준, 승리했을 때 '순이익' (profit) 계산.
      - odds = +150, stake = 100 → profit = 150
      - odds = -200, stake = 100 → profit = 50
    """
    if odds > 0:
        return stake * (odds / 100.0)
    return stake * (100.0 / abs(odds))


def update_bet_log_and_summary(
    reco_df: pd.DataFrame,
    raw_games_past: pd.DataFrame,
    today: _date,
    stake_per_bet: float = 100.0,
):
    """
    - reco_df: 오늘(슬레이트) 추천 bet (edge 필터 통과) - game × side × bookmaker
    - raw_games_past: today 이전까지 완료된 모든 경기 (score 포함)
    - today: PT 기준 슬레이트 날짜 (main에서 넘겨줌)

    동작:
      1) 기존 bet log (nhl_bet_log.csv) 로드
      2) reco_df를 append (result/pnl은 아직 없음)
      3) log 중 result가 비어 있고 date < today 인 것들에 대해
         raw_games_past 를 이용해 승/패 판정 및 PnL 계산
      4) 요약 지표(summary) dict 반환
      5) 갱신된 log CSV 저장
    """
    # 1) 기존 로그 로드
    if os.path.exists(BET_LOG_PATH):
        log = pd.read_csv(BET_LOG_PATH)
    else:
        log = pd.DataFrame()

    # 2) 오늘 추천 bet 추가
    if reco_df is not None and not reco_df.empty:
        new = reco_df.copy()

        base_cols = [
            "date",
            "game_id",
            "home_team",
            "away_team",
            "side",
            "team",
            "bookmaker",
            "american_odds",
            "model_p",
            "implied_p",
            "edge",
        ]

        # 빠진 컬럼 있으면 채워 넣기
        for c in base_cols:
            if c not in new.columns:
                new[c] = np.nan

        new = new[base_cols].copy()
        new["stake"] = float(stake_per_bet)
        new["result"] = np.nan   # 'win' / 'loss'
        new["pnl"] = np.nan      # 달러 기준 손익

        log = pd.concat([log, new], ignore_index=True)

    if log.empty:
        summary = {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": None,
            "total_pnl": 0.0,
            "total_staked": 0.0,
            "roi": None,
            "avg_edge": None,
            "start_date": None,
        }
        return log, summary

    # 타입 정리
    log["date"] = pd.to_datetime(log["date"], errors="coerce").dt.date

    raw = raw_games_past.copy()
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date

    # raw에 필요한 컬럼 체크
    need_cols = {"game_id", "home_score", "away_score", "home_team", "away_team"}
    missing = [c for c in need_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"[NHL PERFORMANCE] raw_games_past missing columns: {missing}")

    # 3) 아직 result가 없고, 날짜가 today 이전인 bet들 settle
    mask_open = log["result"].isna() & (log["date"] < today)

    for idx in log[mask_open].index:
        row = log.loc[idx]
        gid = row.get("game_id")

        # (A) game_id로 매칭
        game = pd.DataFrame()
        try:
            gid_int = int(pd.to_numeric(gid, errors="coerce"))
            game = raw[pd.to_numeric(raw["game_id"], errors="coerce") == gid_int]
        except Exception:
            game = raw[raw["game_id"] == gid]

        # (B) backup: date + home + away
        if game.empty:
            if ("date" in raw.columns) and pd.notna(row.get("date")):
                game = raw[
                    (raw["date"] == row["date"])
                    & (raw["home_team"] == row.get("home_team"))
                    & (raw["away_team"] == row.get("away_team"))
                ]

        if game.empty:
            # 아직 raw 데이터가 덜 들어왔거나 매칭 실패 → 다음 실행 때 재시도
            continue

        g = game.iloc[0]

        side = str(row.get("side", "")).lower().strip()
        if side == "home":
            won = float(g["home_score"]) > float(g["away_score"])
        elif side == "away":
            won = float(g["away_score"]) > float(g["home_score"])
        else:
            continue

        try:
            odds = float(row["american_odds"])
        except Exception:
            continue

        stake = float(row.get("stake", stake_per_bet))
        profit_if_win = _american_payout(stake, odds)
        pnl = profit_if_win if won else -stake

        log.loc[idx, "result"] = "win" if won else "loss"
        log.loc[idx, "pnl"] = pnl

    # 4) 누적 PnL / summary 계산
    log["stake"] = pd.to_numeric(log["stake"], errors="coerce")
    log["pnl"] = pd.to_numeric(log["pnl"], errors="coerce")

    # 누적 PnL (NaN은 0으로 간주해서 누적)
    log["cum_pnl"] = log["pnl"].fillna(0).cumsum()

    settled = log[log["result"].isin(["win", "loss"])].copy()
    total_bets = len(settled)
    wins = int((settled["result"] == "win").sum())
    losses = int((settled["result"] == "loss").sum())

    total_staked = float(settled["stake"].sum()) if total_bets > 0 else 0.0
    total_pnl = float(settled["pnl"].sum()) if total_bets > 0 else 0.0
    win_rate = (wins / total_bets) if total_bets > 0 else None
    roi = (total_pnl / total_staked) if total_staked > 0 else None
    avg_edge = float(settled["edge"].mean()) if total_bets > 0 else None

    start_date = log["date"].min()
    start_date_str = start_date.isoformat() if pd.notnull(start_date) else None

    summary = {
        "total_bets": int(total_bets),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "total_staked": total_staked,
        "roi": roi,
        "avg_edge": avg_edge,
        "start_date": start_date_str,
    }

    # 5) 로그 저장
    os.makedirs(DATA_DIR, exist_ok=True)
    log.to_csv(BET_LOG_PATH, index=False)

    return log, summary