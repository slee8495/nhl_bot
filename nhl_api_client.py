# nhl_api_client.py

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime, timedelta
import time

import requests
import pandas as pd

from config import BALLDONTLIE_API_KEY, NHL_BASE_URL


class NHLDataClient:
    """
    balldontlie NHL API wrapper

    - fetch_games_by_date_range_df: pull all games in [start_date, end_date]
    - fetch_games_by_date / fetch_games_by_date_df: pull games for a single date

    Assumptions:
      - NHL base URL is league-specific, e.g. f"{BALLDONTLIE_BASE_URL}/nhl/v1"
      - endpoints are like:
          GET {NHL_BASE_URL}/games
          GET {NHL_BASE_URL}/teams   (used elsewhere)
      - away team key may be "away_team" or "visitor_team" (defensive)
      - score keys may differ (defensive)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or BALLDONTLIE_API_KEY
        self.base_url = (base_url or NHL_BASE_URL).rstrip("/")
        if not self.api_key:
            raise ValueError("BALLDONTLIE_API_KEY is missing. Set it in your .env.")

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Common GET wrapper
        - Authorization header: API key (no Bearer)
        - retries for 429 with backoff
        """
        url = f"{self.base_url}{path}"
        headers: Dict[str, str] = {
            "Authorization": self.api_key,
            "Accept": "application/json",
        }
        params = params or {}

        last_exc: Optional[Exception] = None

        for i in range(max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=90)

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
                        f"[NHL API] 429 Too Many Requests. "
                        f"Sleep {wait_sec}s and retry ({i+1}/{max_retries})..."
                    )
                    time.sleep(wait_sec)
                    continue

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_exc = e
                if i < max_retries - 1:
                    time.sleep(1.5 * (2 ** i))
                else:
                    break

        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown error in _get()")

    @staticmethod
    def _normalize_game_row(g: Dict[str, Any]) -> Dict[str, Any]:
        home = g.get("home_team") or {}
        away = g.get("away_team") or g.get("visitor_team") or {}

        home_score = g.get("home_team_score", g.get("home_score"))
        away_score = g.get("away_team_score", g.get("visitor_team_score", g.get("away_score")))

        game_date = g.get("game_date")  # "YYYY-MM-DD"
        start_time_utc = g.get("start_time_utc")  # "YYYY-MM-DDTHH:MM:SSZ"

        return {
            "game_id": g.get("id"),

            # ✅ 핵심: slate 계산용은 start_time_utc
            "start_time_utc": start_time_utc,
            # ✅ 보조: 날짜만 필요한 곳(디버그/표시/필터)용
            "game_date": game_date,

            # (기존 호환용) date는 start_time_utc 우선으로 둬도 됨
            "date": start_time_utc or game_date,

            "season": g.get("season"),

            "home_team_id": home.get("id"),
            "away_team_id": away.get("id"),

            "home_team_abbr": (home.get("tricode") or home.get("abbreviation")),
            "away_team_abbr": (away.get("tricode") or away.get("abbreviation")),

            "home_team": home.get("full_name"),
            "away_team": away.get("full_name"),

            "home_score": home_score,
            "away_score": away_score,

            "status": g.get("game_state") or g.get("status"),
            "postseason": g.get("postseason"),
            "period": g.get("period"),
            "time": g.get("time"),
        }





    # nhl_api_client.py 안의 NHLDataClient 클래스에 넣기/교체

    

    def fetch_games_by_date_range_df(
        self,
        start_date: date,
        end_date: date,
        per_page: int = 100,
    ) -> pd.DataFrame:
        """
        Fetch all games in [start_date, end_date] using dates[] (balldontlie NHL spec).
        """
        rows = []
        d = start_date
        while d <= end_date:
            # ✅ balldontlie NHL: dates[]=YYYY-MM-DD 가 정식
            day_rows = self.fetch_games_by_date(d=d, per_page=per_page)
            for g in day_rows:
                rows.append(g)
            d += timedelta(days=1)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        return df


    def fetch_games_by_season(self, seasons: List[int], per_page: int = 100) -> pd.DataFrame:
        """
        Fetch games by seasons[] (multi-season)
        """
        all_rows: List[Dict[str, Any]] = []

        # ✅ per_page 캡
        try:
            per_page = int(per_page)
        except Exception:
            per_page = 100
        per_page = max(1, min(per_page, 100))

        for season in seasons:
            cursor = None

            while True:
                params = {"seasons[]": season, "per_page": per_page}
                if cursor is not None:
                    params["cursor"] = cursor

                data = self._get("/games", params=params)
                games = data.get("data", []) or []
                if not games:
                    break

                for g in games:
                    all_rows.append(self._normalize_game_row(g))

                meta = data.get("meta", {}) or {}
                next_cursor = meta.get("next_cursor")

                if not next_cursor or next_cursor == cursor:
                    break

                cursor = next_cursor


        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
        return df

    def fetch_games_by_date(self, d, per_page: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch games for a date using dates[]=YYYY-MM-DD
        """
        if isinstance(d, (datetime, date)):
            d_str = d.strftime("%Y-%m-%d")
        else:
            d_str = str(d)

        rows: List[Dict[str, Any]] = []

        # ✅ per_page 안전 캡 (API 스펙/서버 검증 강화 대응)
        try:
            per_page = int(per_page)
        except Exception:
            per_page = 100
        per_page = max(1, min(per_page, 100))

        cursor = None

        while True:
            params = {"dates[]": d_str, "per_page": per_page}
            if cursor is not None:
                params["cursor"] = cursor

            data = self._get("/games", params=params)
            games = data.get("data", []) or []
            if not games:
                break

            for g in games:
                rows.append(self._normalize_game_row(g))

            meta = data.get("meta", {}) or {}
            next_cursor = meta.get("next_cursor")

            # ✅ cursor가 없으면 끝 (page 기반 fallback 삭제)
            if not next_cursor or next_cursor == cursor:
                break

            cursor = next_cursor

        return rows


    def fetch_games_by_date_df(
        self,
        d: Union[str, date, datetime],
        per_page: int = 100,
    ) -> pd.DataFrame:
        rows = self.fetch_games_by_date(d=d, per_page=per_page)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
        return df