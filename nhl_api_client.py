# nhl_api_client.py

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime
import time

import requests
import pandas as pd

from config import BALLDONTLIE_API_KEY, NHL_BASE_URL


class NHLDataClient:
    """
    balldontlie NHL API 래퍼
      - 여러 시즌 게임 히스토리 가져오기
      - 특정 날짜 게임 가져오기 (today_games 용)

    NOTE:
      - balldontlie 응답에서 away 팀 키가 "away_team" 또는 "visitor_team"일 수 있어서 둘 다 지원
      - base_url/BALLDONTLIE_BASE_URL 세팅에 따라 path가 /v1/games 또는 /nhl/v1/games 형태일 수 있음
        (1차 포팅은 /v1/games 유지)
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or BALLDONTLIE_API_KEY
        self.base_url = (base_url or NHL_BASE_URL).rstrip("/")
        if not self.api_key:
            raise ValueError("BALLDONTLIE_API_KEY is missing. Set it in your .env.")

    def fetch_games_by_date_range_df(
        self,
        start_date: date,
        end_date: date,
        per_page: int = 100,
    ) -> pd.DataFrame:
        """
        [start_date, end_date] 범위 게임을 전부 가져온다
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        rows = []
        cursor = None
        page = 1

        while True:
            params = {"start_date": start_str, "end_date": end_str, "per_page": per_page}
            if cursor is not None:
                params["cursor"] = cursor
            else:
                params["page"] = page

            data = self._get("/games", params=params)
            games = data.get("data", []) or []
            if not games:
                break

            for g in games:
                rows.append(self._normalize_game_row(g))

            meta = data.get("meta", {}) or {}
            next_cursor = meta.get("next_cursor", None)

            if next_cursor is not None:
                if next_cursor == cursor:
                    break
                cursor = next_cursor
                continue

            total_pages = meta.get("total_pages", page)
            if page >= total_pages:
                break
            page += 1

        df = pd.DataFrame(rows)
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def _get(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        공통 GET 래퍼
        - v1 엔드포인트: Authorization 헤더에 Bearer 없이 API 키 그대로
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
        """
        balldontlie game dict -> downstream에서 쓰기 좋은 표준 row로 변환

        NHL/NBA 차이 방어:
          - home_team: 동일
          - away 팀: away_team 또는 visitor_team
          - score 키도 리그별로 다를 수 있어 fallback 처리
        """
        home = g.get("home_team") or {}

        # ✅ NHL 가능성: away_team
        away = g.get("away_team") or g.get("visitor_team") or {}

        # ✅ 점수 키도 방어 (NBA: home_team_score / visitor_team_score)
        home_score = g.get("home_team_score", g.get("home_score"))
        away_score = g.get("away_team_score", g.get("visitor_team_score", g.get("away_score")))

        return {
            "game_id": g.get("id"),
            "date": g.get("date"),
            "season": g.get("season"),

            "home_team_id": home.get("id"),
            "away_team_id": away.get("id"),
            "home_team_abbr": home.get("abbreviation"),
            "away_team_abbr": away.get("abbreviation"),

            "home_team": home.get("full_name"),
            "away_team": away.get("full_name"),

            "home_score": home_score,
            "away_score": away_score,

            "status": g.get("status"),
            "postseason": g.get("postseason"),
            "period": g.get("period"),
            "time": g.get("time"),
        }

    def fetch_games_by_season(self, seasons: List[int], per_page: int = 100) -> pd.DataFrame:
        all_rows: List[Dict[str, Any]] = []

        for season in seasons:
            cursor = None
            page = 1

            while True:
                params = {"seasons[]": season, "per_page": per_page}
                if cursor is not None:
                    params["cursor"] = cursor
                else:
                    params["page"] = page

                data = self._get("/games", params=params)
                games = data.get("data", []) or []
                if not games:
                    break

                for g in games:
                    all_rows.append(self._normalize_game_row(g))

                meta = data.get("meta", {}) or {}
                next_cursor = meta.get("next_cursor", None)

                if next_cursor is not None:
                    if next_cursor == cursor:
                        break
                    cursor = next_cursor
                    continue

                total_pages = meta.get("total_pages", page)
                if page >= total_pages:
                    break
                page += 1

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df

    def fetch_games_by_date(self, d, per_page: int = 100) -> List[Dict[str, Any]]:
        if isinstance(d, (datetime, date)):
            d_str = d.strftime("%Y-%m-%d")
        else:
            d_str = str(d)

        rows: List[Dict[str, Any]] = []
        cursor = None
        page = 1

        while True:
            params = {"dates[]": d_str, "per_page": per_page}
            if cursor is not None:
                params["cursor"] = cursor
            else:
                params["page"] = page

            data = self._get("/games", params=params)
            games = data.get("data", []) or []
            if not games:
                break

            for g in games:
                rows.append(self._normalize_game_row(g))

            meta = data.get("meta", {}) or {}
            next_cursor = meta.get("next_cursor", None)

            if next_cursor is not None:
                if next_cursor == cursor:
                    break
                cursor = next_cursor
                continue

            total_pages = meta.get("total_pages", page)
            if page >= total_pages:
                break
            page += 1

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
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df