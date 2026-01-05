# nhl_advanced_stats.py

from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional
import math
import time

import requests
import pandas as pd
import numpy as np

from config import BALLDONTLIE_API_KEY, BALLDONTLIE_BASE_URL


def _chunked(iterable: Iterable, n: int) -> Iterable[list]:
    """간단한 리스트 chunk 유틸."""
    iterable = list(iterable)
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


class NHLStatsClient:
    """
    balldontlie /v1/stats (박스스코어)용 간단 클라이언트.
    - NHLDataClient와 같은 방식으로 인증/URL 처리
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or BALLDONTLIE_API_KEY
        self.base_url = (base_url or BALLDONTLIE_BASE_URL).rstrip("/")
        if not self.api_key:
            raise ValueError("BALLDONTLIE_API_KEY is missing. Set it in your .env.")

    def _get(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers: Dict[str, str] = {
            # v1 스타일: Bearer 없이 키 그대로
            "Authorization": self.api_key,
            "Accept": "application/json",
        }

        params = params or {}

        for i in range(max_retries):
            resp = requests.get(url, params=params, headers=headers, timeout=30)

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait_sec = int(retry_after)
                    except ValueError:
                        wait_sec = 10
                else:
                    wait_sec = 5 * (2 ** i)

                print(
                    f"[NHL STATS] 429 Too Many Requests. "
                    f"Sleep {wait_sec}s and retry ({i+1}/{max_retries})..."
                )
                time.sleep(wait_sec)
                continue

            resp.raise_for_status()
            return resp.json()

        resp.raise_for_status()

    def fetch_player_stats_for_games(
        self,
        game_ids: List[int],
        per_page: int = 100,
    ) -> pd.DataFrame:
        """
        /v1/stats 엔드포인트에서 특정 game_ids에 대한
        플레이어 레벨 박스스코어를 모두 가져와서 DataFrame으로 반환.
        """
        if not game_ids:
            return pd.DataFrame()

        all_rows: List[Dict[str, Any]] = []

        # balldontlie 는 game_ids[]= ... 배열로 받음.
        for chunk_ids in _chunked(game_ids, 25):
            cursor: Any = None

            while True:
                params: Dict[str, Any] = {"per_page": per_page}
                for gid in chunk_ids:
                    params.setdefault("game_ids[]", []).append(int(gid))
                if cursor is not None:
                    params["cursor"] = cursor

                data = self._get("/v1/stats", params=params)

                stats = data.get("data", [])
                if not stats:
                    break

                for s in stats:
                    game = s.get("game", {}) or {}
                    team = s.get("team", {}) or {}

                    all_rows.append(
                        {
                            "game_id": game.get("id"),
                            "team_id": team.get("id"),
                            "team_name": team.get("full_name"),
                            "season": game.get("season"),
                            "pts": s.get("pts"),
                            "fga": s.get("fga"),
                            "fgm": s.get("fgm"),
                            "fg3a": s.get("fg3a"),
                            "fg3m": s.get("fg3m"),
                            "fta": s.get("fta"),
                            "oreb": s.get("oreb"),
                            "dreb": s.get("dreb"),
                            "tov": s.get("turnover"),
                            "min": s.get("min"),
                        }
                    )

                meta = data.get("meta", {}) or {}
                next_cursor = meta.get("next_cursor")
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # -------------------------
        # ✅ numeric casting (stability)
        # -------------------------
        num_cols = ["pts", "fga", "fgm", "fg3a", "fg3m", "fta", "oreb", "dreb", "tov"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # ids
        for c in ["game_id", "team_id", "season"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        return df


def build_team_advanced_stats_panel(raw_games: pd.DataFrame) -> pd.DataFrame:
    """
    입력: raw_games (nhl_api_client.fetch_games_by_season에서 받아온 DF)
      - 최소 필요 컬럼: (game_id or id), date, season
    출력: game × team 레벨 advanced metrics:
      - game_id, team_id, team_name, season, date
      - off_rating, def_rating, net_rating
    """
    if raw_games is None or raw_games.empty:
        return pd.DataFrame()

    # -------------------------
    # ✅ normalize raw_games columns
    # -------------------------
    rg = raw_games.copy()
    if "game_id" not in rg.columns and "id" in rg.columns:
        rg["game_id"] = rg["id"]

    if "game_id" not in rg.columns:
        return pd.DataFrame()

    game_ids = rg["game_id"].dropna().astype(int).unique().tolist()
    if not game_ids:
        return pd.DataFrame()

    stats_client = NHLStatsClient()
    player_stats = stats_client.fetch_player_stats_for_games(game_ids)

    if player_stats is None or player_stats.empty:
        return pd.DataFrame()

    # 팀 합산 (게임×팀)
    team_box = (
        player_stats.groupby(["game_id", "team_id", "team_name"], as_index=False)
        .agg(
            pts=("pts", "sum"),
            fga=("fga", "sum"),
            fgm=("fgm", "sum"),
            fg3a=("fg3a", "sum"),
            fg3m=("fg3m", "sum"),
            fta=("fta", "sum"),
            oreb=("oreb", "sum"),
            dreb=("dreb", "sum"),
            tov=("tov", "sum"),
        )
    )

    # meta 붙이기
    gm_meta = rg[["game_id", "date", "season"]].drop_duplicates()
    team_box = team_box.merge(gm_meta, how="left", on="game_id")

    # 상대 팀 정보 붙이기 (same game_id, 다른 team_id)
    opp = team_box[
        ["game_id", "team_id", "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "oreb", "dreb", "tov"]
    ].copy()
    opp = opp.rename(
        columns={
            "team_id": "opp_team_id",
            "pts": "opp_pts",
            "fga": "opp_fga",
            "fgm": "opp_fgm",
            "fg3a": "opp_fg3a",
            "fg3m": "opp_fg3m",
            "fta": "opp_fta",
            "oreb": "opp_oreb",
            "dreb": "opp_dreb",
            "tov": "opp_tov",
        }
    )

    merged = team_box.merge(opp, how="inner", on="game_id")

    # 본인 팀과 동일한 row 제거 → 각 경기당 2행
    merged = merged[merged["team_id"] != merged["opp_team_id"]].copy()

    # -------------------------
    # ✅ possessions + ratings (stable: use avg possessions)
    # -------------------------
    def _safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
        a = pd.to_numeric(a, errors="coerce").astype(float)
        b = pd.to_numeric(b, errors="coerce").astype(float)
        out = a / b
        out = out.where((b != 0) & (~b.isna()), np.nan)
        return out

    team_poss = merged["fga"] + 0.44 * merged["fta"] - merged["oreb"] + merged["tov"]
    opp_poss = merged["opp_fga"] + 0.44 * merged["opp_fta"] - merged["opp_oreb"] + merged["opp_tov"]
    merged["poss"] = 0.5 * (team_poss + opp_poss)

    merged["off_rating"] = 100.0 * _safe_div_series(merged["pts"], merged["poss"])
    merged["def_rating"] = 100.0 * _safe_div_series(merged["opp_pts"], merged["poss"])
    merged["net_rating"] = merged["off_rating"] - merged["def_rating"]

    # 필요 컬럼만
    adv = merged[
        ["game_id", "team_id", "team_name", "season", "date", "off_rating", "def_rating", "net_rating"]
    ].copy()

    # 팀 이름 정규화 (우리 패널이 full_name을 team으로 사용)
    adv["team"] = adv["team_name"]

    return adv