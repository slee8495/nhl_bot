# nhl_injury_lineup.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import date as _date
import os
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

from config import BALLDONTLIE_API_KEY, NHL_BASE_URL, DATA_DIR

print("[NHL LINEUP] LOADED nhl_injury_lineup.py (tricode-mapping build)")

# ==========================================================
# Utilities
# ==========================================================
def _norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return " ".join(s.lower().strip().split())


def _safe_int(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        return int(x)
    except Exception:
        return None


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clip_prob(p: float) -> float:
    return float(np.clip(p, 0.01, 0.99))

def _normalize_bdl_season(season: int) -> int:
    """
    balldontlie NHL season 파라미터는 보통 '시즌 시작 연도'(예: 2024)를 기대.
    네 봇에서 season이 20252026 같은 형태로 들어올 가능성도 있어서 안전 변환.
    """
    s = int(season)
    if s >= 10_000_000:      # e.g., 20252026
        return s // 10000    # -> 2025
    if s >= 1_000_000:       # e.g., 20242025
        return s // 10000    # -> 2024
    return s

def _normalize_tricode(x: Any) -> str:
    s = _norm(x).upper()
    return s.replace(" ", "")

def _resolve_bdl_team_ids_from_tricodes(tricodes: List[str], client: NHLLineupClient) -> List[int]:
    """
    schedule의 home_team_abbr/away_team_abbr(=tricode) -> BDL /teams의 id로 매핑
    """
    teams = client.fetch_teams()
    if teams is None or teams.empty:
        return []

    teams = teams.copy()
    teams["tricode_norm"] = teams["tricode"].apply(_normalize_tricode)

    want = {_normalize_tricode(t) for t in tricodes if t}
    hit = teams[teams["tricode_norm"].isin(want)]

    out = pd.to_numeric(hit["team_id"], errors="coerce").dropna().astype(int).unique().tolist()
    return out


def _attach_bdl_team_ids_to_games(df: pd.DataFrame, bdl_client: "NHLLineupClient") -> pd.DataFrame:
    """
    scored_games(df)의 home_team_abbr/away_team_abbr(tricode) -> BDL team_id를 붙인다.
    결과 컬럼: home_team_id_bdl, away_team_id_bdl
    """
    out = df.copy()

    if ("home_team_abbr" not in out.columns) or ("away_team_abbr" not in out.columns):
        return out

    teams = bdl_client.fetch_teams()
    if teams is None or teams.empty:
        return out

    teams = teams.copy()
    teams["tricode_norm"] = teams["tricode"].apply(_normalize_tricode)
    teams["team_id_bdl"] = pd.to_numeric(teams["team_id"], errors="coerce").astype("Int64")

    out["home_tricode_norm"] = out["home_team_abbr"].apply(_normalize_tricode)
    out["away_tricode_norm"] = out["away_team_abbr"].apply(_normalize_tricode)

    # home
    out = out.merge(
        teams[["tricode_norm", "team_id_bdl"]].rename(
            columns={"tricode_norm": "home_tricode_norm", "team_id_bdl": "home_team_id_bdl"}
        ),
        on="home_tricode_norm",
        how="left",
    )

    # away
    out = out.merge(
        teams[["tricode_norm", "team_id_bdl"]].rename(
            columns={"tricode_norm": "away_tricode_norm", "team_id_bdl": "away_team_id_bdl"}
        ),
        on="away_tricode_norm",
        how="left",
    )

    out.drop(columns=["home_tricode_norm", "away_tricode_norm"], inplace=True, errors="ignore")
    return out




# ==========================================================
# Config
# ==========================================================
@dataclass
class NHLLineupConfig:
    min_toi_rotation: float = 12.0
    top_n_stars: int = 3

    depth_clip_min: float = 0.0
    depth_clip_max: float = 1.2

    depth_weight: float = 0.15
    star_penalty: float = 0.20
    goalie_penalty: float = 0.55
    goalie_is_star: bool = True

    star_min_toi_for_rank: float = 14.0
    star_score_clip: float = 3.0
    star_weight_pts: float = 1.00
    star_weight_g: float = 1.20
    star_weight_a: float = 0.80

    goalie_min_toi_for_rank: float = 35.0
    goalie_score_clip: float = 2.5

    out_keywords: tuple[str, ...] = ("out", "doubtful", "ir", "injured reserve", "suspended")

    fail_hard_if_lineup_bad: bool = False
    min_team_coverage: float = 0.85

    # API 호출량 줄이기용 캐시
    cache_dir: str = DATA_DIR
    cache_ttl_hours: int = 24


# ==========================================================
# Client (API는 "있으면 쓰고, 없으면 CSV로도 동작")
# ==========================================================
class NHLLineupClient:
    """
    balldontlie NHL v1 (GOAT)
      - GET /teams
      - GET /players (team_ids[], seasons[])
      - GET /players/:id/season_stats?season=YYYY
      - GET /player_injuries
    Docs: https://api.balldontlie.io/nhl/v1/...  [oai_citation:3‡nhl.balldontlie.io](https://nhl.balldontlie.io/?utm_source=chatgpt.com)
    """

    TEAMS_PATH = "/teams"
    PLAYERS_PATH = "/players"
    PLAYER_INJURIES_PATH = "/player_injuries"

    def __init__(self, api_key=None, base_url=None, max_retries=3):
        self.api_key = api_key or BALLDONTLIE_API_KEY
        self.base_url = (base_url or NHL_BASE_URL).rstrip("/")
        self.max_retries = max_retries
        if not self.api_key:
            raise ValueError("BALLDONTLIE_API_KEY missing.")

    def _get(self, path, params=None):
        if not path.startswith("/"):
            path = "/" + path
        url = f"{self.base_url}{path}"
        headers = {"Authorization": self.api_key, "Accept": "application/json"}
        params = params or {}

        last_exc = None
        for i in range(self.max_retries):
            try:
                resp = requests.get(url, params=params, headers=headers, timeout=90)

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait_sec = int(retry_after) if (retry_after and retry_after.isdigit()) else 5 * (2 ** i)
                    print(f"[NHL LINEUP] 429 Too Many Requests. Sleep {wait_sec}s ({i+1}/{self.max_retries})")
                    time.sleep(wait_sec)
                    continue

                if resp.status_code in (401, 403):
                    raise RuntimeError(f"[NHL LINEUP AUTH] {resp.status_code} {url} | {resp.text[:200]}")

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_exc = e
                time.sleep(1.0 * (i + 1))

        raise last_exc

    def fetch_teams(self) -> pd.DataFrame:
        rows = []
        cursor = None
        while True:
            params = {"per_page": 100}
            if cursor is not None:
                params["cursor"] = cursor
            data = self._get(self.TEAMS_PATH, params=params)
            items = data.get("data", []) or []
            if not items:
                break

            for t in items:
                rows.append(
                    {
                        "team_id": t.get("id"),
                        "full_name": t.get("full_name"),
                        "tricode": t.get("tricode"),
                        "conference_name": t.get("conference_name"),
                        "division_name": t.get("division_name"),
                    }
                )

            meta = data.get("meta", {}) or {}
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return pd.DataFrame(rows)

    def fetch_players(self, season: int, team_ids: list[int]) -> pd.DataFrame:
        """
        GET /players?seasons[]=2024&team_ids[]=61...
        """
        rows = []
        cursor = None
        while True:
            params = {"per_page": 100, "seasons[]": int(season)}
            params["team_ids[]"] = [int(t) for t in team_ids]

            if cursor is not None:
                params["cursor"] = cursor

            data = self._get(self.PLAYERS_PATH, params=params)
            items = data.get("data", []) or []
            if not items:
                break

            rows.extend(items)
            meta = data.get("meta", {}) or {}
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return pd.DataFrame(rows)

    def fetch_player_season_stats(self, player_id: int, season: int) -> dict:
        """
        GET /players/:id/season_stats?season=2024  [oai_citation:4‡nhl.balldontlie.io](https://nhl.balldontlie.io/?utm_source=chatgpt.com)
        """
        return self._get(f"/players/{int(player_id)}/season_stats", params={"season": int(season)})

    def fetch_player_injuries(self, team_ids: list[int] | None = None) -> pd.DataFrame:
        """
        GET /player_injuries?team_ids[]=...
        """
        rows = []
        cursor = None
        while True:
            params = {"per_page": 100}
            if cursor is not None:
                params["cursor"] = cursor
            if team_ids:
                params["team_ids[]"] = [int(t) for t in team_ids]

            data = self._get(self.PLAYER_INJURIES_PATH, params=params)
            items = data.get("data", []) or []
            if not items:
                break

            rows.extend(items)
            meta = data.get("meta", {}) or {}
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        return pd.DataFrame(rows)
        

        


# ==========================================================
# Core: build team-level lineup table
# ==========================================================
def _is_out_status(status: Any, cfg: NHLLineupConfig) -> bool:
    s = _norm(status)
    if not s:
        return False
    return any(k in s for k in cfg.out_keywords)


def _compute_star_scores(team_df: pd.DataFrame, cfg: NHLLineupConfig) -> pd.DataFrame:
    """
    team_df: one-team season averages subset
    MUST have: toi, pts, g, a, is_goalie
    Returns team_df with 'star_score'
    """
    df = team_df.copy()

    df["toi"] = pd.to_numeric(df.get("toi", 0), errors="coerce").fillna(0.0)
    df["pts"] = pd.to_numeric(df.get("pts", 0), errors="coerce").fillna(0.0)
    df["g"] = pd.to_numeric(df.get("g", 0), errors="coerce").fillna(0.0)
    df["a"] = pd.to_numeric(df.get("a", 0), errors="coerce").fillna(0.0)
    df["is_goalie"] = pd.to_numeric(df.get("is_goalie", 0), errors="coerce").fillna(0).astype(int)

    # 기본적으로 goalie는 스타 랭킹에서 제외(별도 처리)
    skaters = df[df["is_goalie"] == 0].copy()
    if skaters.empty:
        df["star_score"] = 0.0
        return df

    skaters = skaters[skaters["toi"] >= float(cfg.star_min_toi_for_rank)].copy()
    if skaters.empty:
        skaters = df[df["is_goalie"] == 0].copy()

    denom = skaters["toi"].clip(lower=1.0)
    skaters["ppm_pts"] = skaters["pts"] / denom
    skaters["ppm_g"] = skaters["g"] / denom
    skaters["ppm_a"] = skaters["a"] / denom

    skaters["impact_base"] = (
        cfg.star_weight_pts * skaters["ppm_pts"]
        + cfg.star_weight_g * skaters["ppm_g"]
        + cfg.star_weight_a * skaters["ppm_a"]
    )

    # 출전시간 반영: sqrt(toi/18)
    skaters["toi_scale"] = np.sqrt(skaters["toi"].clip(lower=1.0) / 18.0)
    skaters["star_score"] = (skaters["impact_base"]) * skaters["toi_scale"]
    skaters["star_score"] = skaters["star_score"].clip(0.0, float(cfg.star_score_clip))

    out = df.merge(
        skaters[["player_name", "star_score"]],
        on="player_name",
        how="left",
    )
    out["star_score"] = out["star_score"].fillna(0.0)
    return out


def _pick_goalie(team_df: pd.DataFrame, cfg: NHLLineupConfig) -> Optional[Dict[str, Any]]:
    """
    주전 골리 proxy:
      - is_goalie==1 중에서 TOI가 가장 큰 선수
    """
    df = team_df.copy()
    if "is_goalie" not in df.columns:
        return None
    g = df[df["is_goalie"].astype(int) == 1].copy()
    if g.empty:
        return None
    g["toi"] = pd.to_numeric(g.get("toi", 0), errors="coerce").fillna(0.0)
    # goalie_min_toi_for_rank 미만이면 신뢰도 낮지만, 그래도 top1 뽑는다
    g = g.sort_values("toi", ascending=False)
    row = g.iloc[0].to_dict()
    return row


def build_lineup_table_for_today(
    season: int,
    team_ids: Optional[List[int]] = None,
    team_names: Optional[List[str]] = None,
    today: Optional[_date] = None,
    cfg: NHLLineupConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or NHLLineupConfig()
    today = today or _date.today()

    # ✅ client를 먼저 만든다 (아래에서 team_names 매핑 시 client 필요)
    client = NHLLineupClient()

    # season normalize (BDL expects start year)
    season_bdl = _normalize_bdl_season(season)

    # team_ids 없으면 team_names로 매핑 시도
    if not team_ids:
        if team_names:
            teams_df = client.fetch_teams()
            if teams_df is None or teams_df.empty:
                print("[NHL LINEUP] team_ids missing and teams fetch failed -> skip lineup.")
                return pd.DataFrame()

            teams_df = teams_df.copy()
            teams_df["name_norm"] = teams_df["full_name"].apply(_norm)

            want = set(_norm(x) for x in team_names if x)
            mapped = teams_df[teams_df["name_norm"].isin(want)]["team_id"].dropna().astype(int).tolist()

            if mapped:
                team_ids = mapped
            else:
                print("[NHL LINEUP] team_ids missing and team_names not mappable -> skip lineup.")
                return pd.DataFrame()
        else:
            print("[NHL LINEUP] team_ids missing -> skip lineup (NHL needs team_ids).")
            return pd.DataFrame()

    team_ids = [int(x) for x in team_ids if x is not None]

    # ==========================================================
    # 1) players list (team_ids + season)
    # ==========================================================
    players = client.fetch_players(season=season_bdl, team_ids=team_ids)
    if players is None or players.empty:
        print("[NHL LINEUP] players empty -> skip.")
        return pd.DataFrame()

    # ==========================================================
    # 2) player season stats (player별 호출) -> df_sa 생성
    # ==========================================================
    rows = []
    for _, p in players.iterrows():
        pid = _safe_int(p.get("id"))
        if pid is None:
            continue

        try:
            payload = client.fetch_player_season_stats(player_id=pid, season=season_bdl)
            data = payload.get("data")

            # data가 dict 또는 list일 수 있음
            if isinstance(data, dict):
                data = [data]
            st = (data[0] if (isinstance(data, list) and len(data) > 0) else {}) or {}

            team = p.get("team") if isinstance(p.get("team"), dict) else {}
            team_id = _safe_int(team.get("id")) or _safe_int(p.get("team_id"))
            team_name = team.get("full_name") or team.get("name") or p.get("team_name")

            first = p.get("first_name") or ""
            last = p.get("last_name") or ""
            player_name = (f"{first} {last}").strip() or p.get("full_name") or str(pid)

            pos = p.get("position") or ""
            is_goalie = 1 if _norm(pos) in ("g", "goalie", "goaltender") else 0

            # best-effort stat keys
            toi = st.get("toi") or st.get("time_on_ice") or st.get("avg_toi") or 0
            pts = st.get("points") or st.get("pts") or 0
            g = st.get("goals") or st.get("g") or 0
            a = st.get("assists") or st.get("a") or 0

            rows.append(
                {
                    "player_id": pid,
                    "player_name": player_name,
                    "team_id": team_id,
                    "team_name": team_name,
                    "position": pos,
                    "toi": _safe_float(toi, 0.0),
                    "pts": _safe_float(pts, 0.0),
                    "g": _safe_float(g, 0.0),
                    "a": _safe_float(a, 0.0),
                    "is_goalie": int(is_goalie),
                }
            )

        except Exception as e:
            print(f"[NHL LINEUP DEBUG] season_stats failed pid={pid} err={type(e).__name__}: {e}")
            continue

    df_sa = pd.DataFrame(rows)
    if df_sa.empty:
        print("[NHL LINEUP] season_avgs empty (players+season_stats). Skip lineup.")
        return pd.DataFrame()

    # ==========================================================
    # 3) injuries (team_ids 필터) -> inj 정규화
    # ==========================================================
    inj_raw = client.fetch_player_injuries(team_ids=team_ids)
    if inj_raw is None or inj_raw.empty:
        inj = pd.DataFrame()
    else:
        inj = inj_raw.copy()

        def _get_player_name(x: Any) -> str:
            if isinstance(x, dict):
                fn = x.get("first_name") or ""
                ln = x.get("last_name") or ""
                return (f"{fn} {ln}").strip()
            return ""

        def _get_team_id(x: Any) -> Optional[int]:
            if isinstance(x, dict):
                return _safe_int(x.get("id"))
            return None

        def _get_team_name(x: Any) -> str:
            if isinstance(x, dict):
                return x.get("full_name") or x.get("name") or ""
            return ""

        if "player_name" not in inj.columns:
            if "player" in inj.columns:
                inj["player_name"] = inj["player"].apply(_get_player_name)
            else:
                inj["player_name"] = ""

        if "team_id" not in inj.columns:
            if "team" in inj.columns:
                inj["team_id"] = inj["team"].apply(_get_team_id)
            else:
                if "player" in inj.columns:
                    inj["team_id"] = inj["player"].apply(
                        lambda d: _safe_int(d.get("team_id")) if isinstance(d, dict) else None
                    )
                else:
                    inj["team_id"] = None

        if "team_name" not in inj.columns:
            if "team" in inj.columns:
                inj["team_name"] = inj["team"].apply(_get_team_name)
            else:
                inj["team_name"] = ""

        if "status" not in inj.columns:
            inj["status"] = inj.get("injury_status", "") if "injury_status" in inj.columns else ""

        inj["team_id_int"] = pd.to_numeric(inj["team_id"], errors="coerce").fillna(-1).astype(int)
        inj["player_name_norm"] = inj["player_name"].apply(_norm)
        inj["status_out"] = inj["status"].apply(lambda x: _is_out_status(x, cfg))

    # ==========================================================
    # df_sa 컬럼/타입 정리
    # ==========================================================
    for c in ["team_id", "team_name", "player_name", "position"]:
        if c not in df_sa.columns:
            df_sa[c] = np.nan

    if "is_goalie" not in df_sa.columns:
        df_sa["is_goalie"] = df_sa["position"].apply(lambda x: 1 if _norm(x) in ("g", "goalie", "goaltender") else 0)

    for c in ["toi", "pts", "g", "a", "is_goalie"]:
        if c not in df_sa.columns:
            df_sa[c] = 0

    df_sa["team_id_int"] = pd.to_numeric(df_sa["team_id"], errors="coerce").fillna(-1).astype(int)
    df_sa["toi"] = pd.to_numeric(df_sa["toi"], errors="coerce").fillna(0.0)
    df_sa["pts"] = pd.to_numeric(df_sa["pts"], errors="coerce").fillna(0.0)
    df_sa["g"] = pd.to_numeric(df_sa["g"], errors="coerce").fillna(0.0)
    df_sa["a"] = pd.to_numeric(df_sa["a"], errors="coerce").fillna(0.0)
    df_sa["is_goalie"] = pd.to_numeric(df_sa["is_goalie"], errors="coerce").fillna(0).astype(int)

    # team filter (team_ids 기반)
    keep = set(int(x) for x in team_ids if x is not None)
    df_sa = df_sa[df_sa["team_id_int"].isin(keep)].copy()

    if df_sa.empty:
        print("[NHL LINEUP] season_avgs filtered empty. Skip lineup.")
        return pd.DataFrame()

    # ==========================================================
    # Build per-team
    # ==========================================================
    out_rows: List[Dict[str, Any]] = []
    team_keys = sorted(df_sa["team_id_int"].unique().tolist())

    for tid in team_keys:
        team_df = df_sa[df_sa["team_id_int"] == tid].copy()
        if team_df.empty:
            continue

        team_name = (
            str(team_df["team_name"].dropna().iloc[0])
            if team_df["team_name"].notna().any()
            else str(tid)
        )

        # rotation: skaters only
        skaters = team_df[team_df["is_goalie"] == 0].copy().sort_values("toi", ascending=False)
        rotation = skaters[skaters["toi"] >= float(cfg.min_toi_rotation)].copy()
        if rotation.empty:
            rotation = skaters.copy()

        baseline_total_toi = float(rotation["toi"].sum())
        if baseline_total_toi <= 0:
            baseline_total_toi = 1.0

        # star scoring
        team_df_scored = _compute_star_scores(team_df, cfg)
        stars = (
            team_df_scored[team_df_scored["is_goalie"] == 0]
            .sort_values("star_score", ascending=False)
            .head(int(cfg.top_n_stars))
            .copy()
        )
        star_names = set(stars["player_name"].dropna().astype(str).tolist())
        star_score_map = dict(zip(stars["player_name"].astype(str), stars["star_score"].astype(float)))

        # goalie pick
        goalie = _pick_goalie(team_df_scored, cfg)
        goalie_name = str(goalie.get("player_name")) if goalie else None
        goalie_score = float(cfg.goalie_score_clip) if goalie and cfg.goalie_is_star else 0.0

        # OUT list (by injuries)
        out_names: set[str] = set()
        if inj is not None and (not inj.empty):
            inj_team = inj[(inj["team_id_int"] == int(tid)) & (inj["status_out"])].copy()
            if not inj_team.empty:
                out_names = set(inj_team["player_name_norm"].dropna().astype(str).tolist())

        # availability among rotation by name matching
        rotation["player_name_norm"] = rotation["player_name"].apply(_norm)
        rotation["is_available"] = ~rotation["player_name_norm"].isin(out_names)

        active_toi = float(rotation.loc[rotation["is_available"], "toi"].sum())
        depth_score = float(np.clip(active_toi / baseline_total_toi, float(cfg.depth_clip_min), float(cfg.depth_clip_max)))

        # missing stars
        missing_star_score = 0.0
        missing_star_count = 0
        for pn in star_names:
            if _norm(pn) in out_names:
                missing_star_count += 1
                missing_star_score += float(star_score_map.get(str(pn), 0.0))

        missing_star_score = float(
            np.clip(missing_star_score, 0.0, float(cfg.star_score_clip) * float(cfg.top_n_stars))
        )

        # goalie out
        goalie_out_flag = 0
        goalie_missing_score = 0.0
        if goalie_name and (_norm(goalie_name) in out_names):
            goalie_out_flag = 1
            goalie_missing_score = float(np.clip(goalie_score, 0.0, float(cfg.goalie_score_clip)))

        out_rows.append(
            {
                "team_id": int(tid),
                "team_name": team_name,
                "season": int(season),
                "lineup_depth_score": depth_score,
                "lineup_missing_star_count": int(missing_star_count),
                "lineup_missing_star_score": float(missing_star_score),
                "goalie_out_flag": int(goalie_out_flag),
                "goalie_missing_score": float(goalie_missing_score),
            }
        )

    return pd.DataFrame(out_rows) if out_rows else pd.DataFrame()


# ==========================================================
# Attach features to scored_games (same style as NBA)
# ==========================================================
def attach_lineup_features(
    scored_games: pd.DataFrame,
    today: _date,
    cfg: NHLLineupConfig | None = None,
) -> pd.DataFrame:
    if scored_games is None or scored_games.empty:
        return scored_games

    cfg = cfg or NHLLineupConfig()
    df = scored_games.copy()

    # season 결정
    if "season" in df.columns and df["season"].notna().any():
        season = int(pd.to_numeric(df["season"], errors="coerce").dropna().max())
    else:
        season = int(today.year)

    # ✅ tricode 없으면 lineup 자체 스킵 (NHL은 이름매칭 정확도 낮음)
    if ("home_team_abbr" not in df.columns) or ("away_team_abbr" not in df.columns):
        print("[NHL LINEUP] missing home/away tricode -> skip lineup.")
        return df

    tricodes = pd.unique(pd.concat([df["home_team_abbr"], df["away_team_abbr"]]).dropna()).tolist()
    tricodes = [str(x) for x in tricodes if str(x).strip()]

    bdl_client = NHLLineupClient()

    # ✅ (1) tricodes -> BDL team_ids
    team_ids_bdl = _resolve_bdl_team_ids_from_tricodes(tricodes, bdl_client)
    if not team_ids_bdl:
        print("[NHL LINEUP] could not resolve BDL team_ids from tricodes -> skip lineup.")
        return df

    # ✅ (2) df에 home_team_id_bdl / away_team_id_bdl 붙이기 (merge 키)
    df = _attach_bdl_team_ids_to_games(df, bdl_client)

    # 혹시라도 bdl id가 안 붙으면 스킵
    if ("home_team_id_bdl" not in df.columns) or ("away_team_id_bdl" not in df.columns):
        print("[NHL LINEUP] failed to attach home/away BDL team ids -> skip lineup.")
        return df

    if df["home_team_id_bdl"].isna().any() or df["away_team_id_bdl"].isna().any():
        # 일부만 NaN이어도 일단 진행은 가능. (coverage는 아래 fail_hard에서 체크)
        print("[NHL LINEUP] WARNING: some games missing BDL team_id mapping (abbr->id).")

    # ✅ (3) lineup_team 생성 (BDL team_id 기준)
    lineup_team = build_lineup_table_for_today(season=season, team_ids=team_ids_bdl, today=today, cfg=cfg)

    if lineup_team is None or lineup_team.empty:
        print("[NHL LINEUP] lineup_team empty -> skip lineup features.")
        return df

    # dtype cleanup
    lineup_team["team_id"] = pd.to_numeric(lineup_team["team_id"], errors="coerce").astype("Int64")
    df["home_team_id_bdl"] = pd.to_numeric(df["home_team_id_bdl"], errors="coerce").astype("Int64")
    df["away_team_id_bdl"] = pd.to_numeric(df["away_team_id_bdl"], errors="coerce").astype("Int64")

    # =========================
    # ✅ MERGE (BDL id로만)
    # =========================
    # HOME
    home_feat = lineup_team.add_suffix("_home")
    df = df.merge(
        home_feat[
            [
                "team_id_home",
                "lineup_depth_score_home",
                "lineup_missing_star_count_home",
                "lineup_missing_star_score_home",
                "goalie_out_flag_home",
                "goalie_missing_score_home",
            ]
        ],
        how="left",
        left_on="home_team_id_bdl",
        right_on="team_id_home",
    )
    df.drop(columns=["team_id_home"], inplace=True, errors="ignore")

    # AWAY
    away_feat = lineup_team.add_suffix("_away")
    df = df.merge(
        away_feat[
            [
                "team_id_away",
                "lineup_depth_score_away",
                "lineup_missing_star_count_away",
                "lineup_missing_star_score_away",
                "goalie_out_flag_away",
                "goalie_missing_score_away",
            ]
        ],
        how="left",
        left_on="away_team_id_bdl",
        right_on="team_id_away",
    )
    df.drop(columns=["team_id_away"], inplace=True, errors="ignore")

    # fail-hard validation (optional)
    if cfg.fail_hard_if_lineup_bad:
        home_ok = df["lineup_depth_score_home"].notna().mean() if "lineup_depth_score_home" in df.columns else 0.0
        away_ok = df["lineup_depth_score_away"].notna().mean() if "lineup_depth_score_away" in df.columns else 0.0
        coverage = float(min(home_ok, away_ok))

        if coverage < float(cfg.min_team_coverage):
            raise RuntimeError(
                "[NHL LINEUP FAIL-HARD] Lineup feature join coverage too low.\n"
                f"- coverage(min(home,away))={coverage:.2f} (need >= {cfg.min_team_coverage:.2f})\n"
                "→ Abort to avoid trading on bad lineup signal."
            )

    return df


# ==========================================================
# Adjust probability with lineup features (same style as NBA)
# ==========================================================
def adjust_prob_with_lineup(
    scored_games: pd.DataFrame,
    cfg: NHLLineupConfig | None = None,
) -> pd.DataFrame:
    if scored_games is None or scored_games.empty:
        return scored_games

    cfg = cfg or NHLLineupConfig()
    df = scored_games.copy()

    def _logit(p: float) -> float:
        p = _clip_prob(p)
        return float(np.log(p / (1.0 - p)))

    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -50, 50))
        return float(1.0 / (1.0 + np.exp(-z)))

    adj_p = []
    z_depth_list, z_star_list, z_goalie_list, z_total_list = [], [], [], []

    for _, row in df.iterrows():
        base_p = _safe_float(row.get("p_home_win", 0.5), 0.5)
        z_base = _logit(base_p)

        depth_home = _safe_float(row.get("lineup_depth_score_home", 1.0), 1.0)
        depth_away = _safe_float(row.get("lineup_depth_score_away", 1.0), 1.0)
        depth_diff = depth_home - depth_away

        miss_score_home = _safe_float(row.get("lineup_missing_star_score_home", 0.0), 0.0)
        miss_score_away = _safe_float(row.get("lineup_missing_star_score_away", 0.0), 0.0)
        # 상대가 더 크게 빠지면 + (home 유리)
        star_diff = (miss_score_away - miss_score_home)

        goalie_miss_home = _safe_float(row.get("goalie_missing_score_home", 0.0), 0.0)
        goalie_miss_away = _safe_float(row.get("goalie_missing_score_away", 0.0), 0.0)
        goalie_diff = (goalie_miss_away - goalie_miss_home)

        z_depth = float(cfg.depth_weight) * float(depth_diff)
        z_star = float(cfg.star_penalty) * float(star_diff)
        z_goalie = float(cfg.goalie_penalty) * float(goalie_diff)

        z_adj = z_base + z_depth + z_star + z_goalie

        # 로짓 폭주 방지
        z_adj = float(np.clip(z_adj, -3.0, 3.0))

        p_adj = _sigmoid(z_adj)
        adj_p.append(_clip_prob(p_adj))

        z_depth_list.append(z_depth)
        z_star_list.append(z_star)
        z_goalie_list.append(z_goalie)
        z_total_list.append(z_adj - z_base)

    df["p_home_win_adj"] = adj_p
    df["z_lineup_depth"] = z_depth_list
    df["z_star_penalty"] = z_star_list
    df["z_goalie_penalty"] = z_goalie_list
    df["z_lineup_total"] = z_total_list

    return df
