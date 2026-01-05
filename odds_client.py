# odds_client.py

from __future__ import annotations
from typing import Dict, Any, Optional
import re
import requests
import pandas as pd

from config import (
    ODDS_API_KEY,
    ODDS_API_BASE_URL,
    BALLDONTLIE_API_KEY,
    NHL_BASE_URL,
)
from nhl_team_abbr import to_abbr as to_nhl_abbr


def _norm_team_name(s: str) -> str:
    """
    팀명 문자열 정규화 (Odds API / balldontlie 간 표기 차이 흡수)
    """
    s = (s or "").lower().strip()
    s = s.replace(".", " ")
    s = re.sub(r"[-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # 흔한 별칭 처리
    s = s.replace("l a ", "la ")
    s = s.replace("los angeles", "la")
    s = s.replace("new york", "ny")

    return s


class OddsAPIClient:
    """
    The Odds API 래퍼 (NHL moneyline odds)
    + balldontlie teams를 이용해 team_abbr 매핑까지 생성
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or ODDS_API_KEY
        self.base_url = (base_url or ODDS_API_BASE_URL).rstrip("/")
        if not self.api_key:
            raise ValueError("ODDS_API_KEY is missing. Set it in your .env.")

        # balldontlie (팀/경기 매핑용)
        self.bdl_key = BALLDONTLIE_API_KEY
        self.bdl_base = (BALLDONTLIE_BASE_URL or "").rstrip("/")
        if not self.bdl_key or not self.bdl_base:
            raise ValueError("BALLDONTLIE_API_KEY / BALLDONTLIE_BASE_URL missing (needed for team mapping).")

        self._team_map_norm_to_abbr: Optional[Dict[str, str]] = None

    # -------------------------
    # Odds API
    # -------------------------
    def _get_odds(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        params = params.copy()
        params["apiKey"] = self.api_key
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # -------------------------
    # balldontlie
    # -------------------------
    def _get_bdl(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.bdl_base}{path}"
        headers = {"Authorization": self.bdl_key, "Accept": "application/json"}
        resp = requests.get(url, params=(params or {}), headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def _build_team_map(self) -> Dict[str, str]:
        """
        returns: { normalized_team_name: "BOS" } 같은 형태
        NOTE: NHL에서도 /v1/teams 가 동일 구조로 내려온다는 전제 (balldontlie 공용)
        """
        if self._team_map_norm_to_abbr is not None:
            return self._team_map_norm_to_abbr

        data = self._get_bdl("/v1/teams", params={"per_page": 200})
        teams = data.get("data", []) or []

        m: Dict[str, str] = {}
        for t in teams:
            full_name = t.get("full_name") or ""
            abbr = t.get("abbreviation") or ""
            if not abbr:
                continue

            m[_norm_team_name(full_name)] = abbr

            city = t.get("city") or ""
            if city:
                nickname = full_name.replace(city, "").strip()
                if nickname:
                    m[_norm_team_name(f"{city} {nickname}")] = abbr

        # ✅ NBA에 있던 LA/NY 하드코딩은 NHL엔 불필요 (유지하면 오염될 수 있음)
        self._team_map_norm_to_abbr = m
        return m

    def _to_abbr(self, team_name: str) -> str:
        """
        Odds API 팀명 -> ABBR
        매칭 실패하면 바로 에러로 터뜨려서 '조용히 누락'되는 상황을 방지
        """
        m = self._build_team_map()
        key = _norm_team_name(team_name)
        abbr = m.get(key)
        if abbr:
            return abbr
        raise KeyError(f"[ODDS TEAM MAP FAIL] Unknown team name: '{team_name}' (norm='{key}')")

    # -------------------------
    # public API
    # -------------------------
    def fetch_nhl_moneyline_odds(self, regions: str = "us", attach_game_id: bool = True) -> pd.DataFrame:
        """
        NHL moneyline odds 가져오기 (The Odds API)
        - sport key: icehockey_nhl
        - markets: h2h (moneyline)
        """
        data = self._get_odds(
            "/sports/icehockey_nhl/odds",
            params={
                "regions": regions,
                "markets": "h2h",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )

        rows = []
        for event in data:
            event_id = event["id"]
            home_team = event["home_team"]
            away_team = event["away_team"]
            start_time = event.get("commence_time")  # ISO string

            home_abbr = self._to_abbr(home_team)
            away_abbr = self._to_abbr(away_team)

            for bk in event.get("bookmakers", []):
                bookmaker = bk.get("title")
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != "h2h":
                        continue
                    for outcome in mkt.get("outcomes", []):
                        team = outcome.get("name")
                        american = outcome.get("price")
                        if team is None or american is None:
                            continue

                        team_abbr = self._to_abbr(team)

                        rows.append(
                            {
                                "event_id": event_id,
                                "start_time": start_time,
                                "home_team": home_team,
                                "away_team": away_team,
                                "home_team_abbr": home_abbr,
                                "away_team_abbr": away_abbr,
                                "bookmaker": bookmaker,
                                "team": team,
                                "team_abbr": team_abbr,
                                "american_odds": american,
                            }
                        )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["start_time"] = pd.to_datetime(df["start_time"], utc=True, errors="coerce")

        if attach_game_id:
            df = self._attach_game_id(df, league="nhl")

        return df

    # -------------------------
    # attach game_id via balldontlie games
    # -------------------------
    def _commence_to_utc_date_str(self, commence_time) -> str:
        if commence_time is None or (isinstance(commence_time, float) and pd.isna(commence_time)):
            raise ValueError("commence_time missing")
        dt = pd.to_datetime(commence_time, utc=True, errors="raise")
        return dt.date().isoformat()

    def _fetch_bdl_games_by_dates(self, dates: list[str], league: str) -> pd.DataFrame:
        """
        balldontlie games를 날짜 리스트로 가져와서 DataFrame 반환

        league:
          - "nba" or "nhl"
        """
        all_rows = []
        for d in sorted(set(dates)):
            # balldontlie: /v1/games?dates[]=YYYY-MM-DD&per_page=100
            # NOTE: NHL에서도 동일 경로라고 가정. 만약 리그별 prefix가 필요하면 여기서만 수정하면 됨.
            data = self._get_bdl("/v1/games", params={"dates[]": d, "per_page": 200})
            games = data.get("data", []) or []

            for g in games:
                # --- NHL/NBA 구조 차이 방어 ---
                # NBA: home_team / visitor_team
                # NHL: home_team / away_team (가능성 높음)
                home = (g.get("home_team") or {})
                away = (g.get("away_team") or g.get("visitor_team") or {})

                all_rows.append(
                    {
                        "game_id": g.get("id"),
                        "date": d,
                        "home_team_abbr": (home.get("abbreviation") or "").upper(),
                        "away_team_abbr": (away.get("abbreviation") or "").upper(),
                    }
                )

        out = pd.DataFrame(all_rows)
        if not out.empty:
            out["home_team_abbr"] = out["home_team_abbr"].astype(str).str.upper()
            out["away_team_abbr"] = out["away_team_abbr"].astype(str).str.upper()
        return out

    def _attach_game_id(self, odds_rows: pd.DataFrame, league: str = "nhl") -> pd.DataFrame:
        """
        odds_rows (event_id 단위) -> balldontlie 기준 game_id 매핑
        매핑 우선순위:
          1) (date_utc, home_abbr, away_abbr)
          2) (date_utc +/- 1일, home_abbr, away_abbr)  # 날짜 경계/타임존 방어
        """
        df = odds_rows.copy()
        if df.empty:
            df["game_id"] = None
            return df

        if "start_time" not in df.columns:
            raise ValueError("odds_rows missing start_time")

        df["date_utc"] = df["start_time"].apply(self._commence_to_utc_date_str)

        dates = df["date_utc"].dropna().unique().tolist()
        dates_plus = set(dates)
        for d in dates:
            dt = pd.to_datetime(d)
            dates_plus.add((dt - pd.Timedelta(days=1)).date().isoformat())
            dates_plus.add((dt + pd.Timedelta(days=1)).date().isoformat())

        bdl_games = self._fetch_bdl_games_by_dates(sorted(dates_plus), league=league)
        if bdl_games.empty:
            df["game_id"] = None
            print("[DEBUG][ODDS][BDL_GAMES] empty - cannot attach game_id")
            return df

        df["home_team_abbr"] = df["home_team_abbr"].astype(str).str.upper()
        df["away_team_abbr"] = df["away_team_abbr"].astype(str).str.upper()

        key_cols = ["date_utc", "home_team_abbr", "away_team_abbr"]
        bdl_map = bdl_games.rename(columns={"date": "date_utc"})[
            ["date_utc", "home_team_abbr", "away_team_abbr", "game_id"]
        ]

        df = df.merge(bdl_map, how="left", on=key_cols)

        miss = df["game_id"].isna()
        if miss.any():
            tmp = df.loc[miss, key_cols].copy()
            tmp["date_dt"] = pd.to_datetime(tmp["date_utc"])
            candidates = []
            for shift in (-1, 1):
                t2 = tmp.copy()
                t2["date_utc"] = (t2["date_dt"] + pd.Timedelta(days=shift)).dt.date.astype(str)
                candidates.append(t2[key_cols])
            retry_keys = pd.concat(candidates, ignore_index=True).drop_duplicates()

            retry = retry_keys.merge(bdl_map, how="left", on=key_cols).dropna(subset=["game_id"])
            if not retry.empty:
                retry["k"] = retry["date_utc"] + "|" + retry["home_team_abbr"] + "|" + retry["away_team_abbr"]
                retry_map = dict(zip(retry["k"], retry["game_id"]))

                def _lookup_game_id(row):
                    base_date = pd.to_datetime(row["date_utc"])
                    for shift in (0, -1, 1):
                        d = (base_date + pd.Timedelta(days=shift)).date().isoformat()
                        k = f"{d}|{row['home_team_abbr']}|{row['away_team_abbr']}"
                        if k in retry_map:
                            return retry_map[k]
                    return None

                df.loc[miss, "game_id"] = df.loc[miss].apply(_lookup_game_id, axis=1)

        null_rate = df["game_id"].isna().mean()
        print(f"[DEBUG][ODDS][ATTACH_GAME_ID] rows={len(df)} null_rate={null_rate:.3f}")

        df.drop(columns=["date_utc"], errors="ignore", inplace=True)
        return df