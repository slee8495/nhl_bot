# email_management.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from datetime import datetime
import pandas as pd
import numpy as np

from config import (
    EMAIL_HOST,
    EMAIL_PORT,
    EMAIL_USER,
    EMAIL_PASSWORD,
    EMAIL_TO,
    NHL_TOTAL_DAILY_RISK,
)


def _calc_return_amount(odds: float, stake: float = 100.0) -> float:
    try:
        o = float(odds)
    except Exception:
        return stake

    if o > 0:
        profit = stake * (o / 100)
    else:
        profit = stake * (100 / abs(o))

    return stake + profit


def _fmt_american(odds) -> str:
    """Format American odds with + / - sign."""
    try:
        o = float(odds)
        return f"{o:+.0f}"
    except Exception:
        return str(odds)


def _calc_ev_amount(odds: float, model_p: float, stake: float = 100.0) -> float:
    """
    Expected value in dollars for a given bet (American odds, model win prob).
    EV = p*profit_if_win + (1-p)*(-stake)
    """
    try:
        o = float(odds)
    except Exception:
        return 0.0

    if o > 0:
        profit_if_win = stake * (o / 100.0)
    else:
        profit_if_win = stake * (100.0 / abs(o))

    ev = model_p * profit_if_win - (1.0 - model_p) * stake
    return ev


def _fmt_profit_pct(odds) -> str:
    """
    Return profit percentage as string without the plus sign, e.g.
    +310 -> "310", -286 -> "34.9"
    """
    try:
        o = float(odds)
    except Exception:
        return "0.0"

    if o > 0:
        return f"{o:.0f}"
    else:
        pct = 10000.0 / abs(o)
        return f"{pct:.1f}"


def _kelly_fraction(model_p: float, american_odds: float) -> float:
    """
    Kelly fraction f* 계산 (bankroll 대비 비율).
    반환:
      - 음수면 0으로 클립 (no bet)
      - 정상 범위 [0, 1] 내에서만 사용
    """
    try:
        p = float(model_p)
        o = float(american_odds)
    except Exception:
        return 0.0

    if p <= 0.0 or p >= 1.0:
        return 0.0

    if o > 0:
        b = o / 100.0
    else:
        b = 100.0 / abs(o)

    q = 1.0 - p
    f_star = (b * p - q) / b
    if f_star <= 0.0:
        return 0.0

    return min(f_star, 0.5)


def _apply_kelly_sizing(bet_rows: pd.DataFrame, total_daily_risk: float) -> pd.DataFrame:
    """
    균등 스테이크 배분 (Equal-weight sizing)
    - total_daily_risk 를 베팅 개수 N으로 나눠서 동일 stake 배정
    - 기존 컬럼 호환을 위해 weight/kelly_f 유지
    """
    if bet_rows is None or bet_rows.empty:
        return bet_rows

    bet_rows = bet_rows.copy()
    n_bets = len(bet_rows)
    if n_bets <= 0:
        return bet_rows

    try:
        total_daily_risk = float(total_daily_risk)
    except Exception:
        total_daily_risk = 100.0 * n_bets

    equal_stake = total_daily_risk / n_bets
    bet_rows["weight"] = 1.0 / n_bets
    bet_rows["stake"] = round(equal_stake, 2)

    # 호환/디버깅용
    bet_rows["kelly_f"] = 0.0
    return bet_rows


def send_nhl_daily_report(
    reco_df: pd.DataFrame,
    full_df: pd.DataFrame,
    perf_summary: dict | None = None,
) -> None:
    """Send daily NHL report email in ENGLISH."""
    has_full = full_df is not None and not full_df.empty


    # =========================================================
    # ✅ NHL/NBA 컬럼명 차이 흡수: 여기서 표준 컬럼을 강제로 만든다
    #   - elig 로직이 p_xgb / model_p / implied_p / american_odds 를 전제로 함
    # =========================================================
    def _ensure_col(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
        if target in df.columns:
            return df
        for c in candidates:
            if c in df.columns:
                df[target] = df[c]
                return df
        df[target] = np.nan
        return df

    def _american_to_prob(o):
        try:
            o = float(o)
        except Exception:
            return np.nan
        if o > 0:
            return 100.0 / (o + 100.0)
        return (-o) / ((-o) + 100.0)

    if has_full:
        full_df = full_df.copy()

        # --- odds / implied ---
        full_df = _ensure_col(full_df, "american_odds", ["american_odds", "odds", "moneyline_odds"])
        full_df["american_odds"] = pd.to_numeric(full_df["american_odds"], errors="coerce")

        # implied_p가 비었으면 american_odds로 계산해서 채움
        if "implied_p" not in full_df.columns:
            full_df["implied_p"] = np.nan
        if full_df["implied_p"].isna().all():
            full_df["implied_p"] = full_df["american_odds"].apply(_american_to_prob)

        # --- model probabilities ---
        full_df = _ensure_col(full_df, "p_xgb", ["p_xgb", "p_win_xgb", "win_prob_xgb", "p_win", "p_pred"])
        full_df["p_xgb"] = pd.to_numeric(full_df["p_xgb"], errors="coerce")

        # model_p가 없으면 p_xgb로 fallback (NHL은 ensemble 컬럼명이 다를 수 있음)
        full_df = _ensure_col(full_df, "model_p", ["model_p", "p_model", "p_final", "p_ensemble", "p_win"])
        full_df["model_p"] = pd.to_numeric(full_df["model_p"], errors="coerce")
        full_df["model_p"] = full_df["model_p"].fillna(full_df["p_xgb"])

    # ------------------------------
    # Helpers
    # ------------------------------
    def _fmt_prob(x) -> str:
        try:
            return f"{float(x):.2f}"
        except Exception:
            return ""

    def _to_num(s, default=np.nan):
        try:
            return pd.to_numeric(s, errors="coerce")
        except Exception:
            return default

    # ------------------------------
    # Edge/EV rule config
    # ------------------------------
    EDGE_MIN_NORMAL = 3.0
    EDGE_MIN_NORMAL_PLUS200 = 5.0

    EDGE_MIN_FLIP = 6.0
    EDGE_MIN_FLIP_PLUS200 = 8.0

    XGB_MIN = 0.50
    MODEL_MIN = 0.50

    AUTO_YES_EDGE_MIN = 1.0      # edge% (percent)  # (현재 로직에서는 -1.0 사용중)
    AUTO_YES_MAX_ABS_ODDS = 250  # -250 ~ +250
    AUTO_YES_ONLY_NON_FLIP = True

    MAX_ORDERS = 10
    UNDERDOG_ONLY = False

    USE_AGREEMENT_FILTER = False  # 합의(2/3) 필터 제거 상태 유지

    def _edge_pct(df: pd.DataFrame, edge_col: str) -> pd.Series:
        return (_to_num(df.get(edge_col)) * 100.0)

    def _market_flip_mask(df: pd.DataFrame) -> pd.Series:
        mp = _to_num(df.get("model_p"))
        ip = _to_num(df.get("implied_p"))
        return ((mp >= 0.5) & (ip < 0.5)) | ((mp <= 0.5) & (ip > 0.5))

    def _agreement_ok_with_min(df: pd.DataFrame, p_min: float) -> pd.Series:
        xgb = _to_num(df.get("p_xgb"))
        mar = _to_num(df.get("p_margin"))
        elo = _to_num(df.get("p_elo"))

        P = np.vstack([xgb.values, mar.values, elo.values]).T.astype(float)
        valid_cnt = np.sum(~np.isnan(P), axis=1)
        has_2 = valid_cnt >= 2

        agree_cnt = np.nansum((P >= 0.50).astype(float), axis=1)
        agree_ok = agree_cnt >= 2

        P2 = np.where(np.isnan(P), -1e9, P)
        second_best = np.sort(P2, axis=1)[:, -2]
        pmin_ok = second_best >= float(p_min)

        return has_2 & agree_ok & pmin_ok

    def _eligible_by_edge(df: pd.DataFrame, edge_col: str) -> pd.Series:
        edgep = _edge_pct(df, edge_col)
        flip = _market_flip_mask(df)

        odds = _to_num(df.get("american_odds"))
        mp = _to_num(df.get("model_p"))
        xgb = _to_num(df.get("p_xgb"))

        base_prob_ok = (xgb >= XGB_MIN) & (mp >= MODEL_MIN)

        is_plus200 = (odds >= 200)
        req_edge_normal = np.where(is_plus200, EDGE_MIN_NORMAL_PLUS200, EDGE_MIN_NORMAL)
        req_edge_flip = np.where(is_plus200, EDGE_MIN_FLIP_PLUS200, EDGE_MIN_FLIP)
        req_edge = np.where(flip, req_edge_flip, req_edge_normal)

        elig = (edgep >= req_edge) & base_prob_ok

        if UNDERDOG_ONLY:
            elig = elig & (odds > 0)

        auto_yes = (
            (xgb >= 0.55) &
            (mp >= 0.55) &
            (odds.abs() <= AUTO_YES_MAX_ABS_ODDS) &
            (edgep >= -1.0)
        )

        if AUTO_YES_ONLY_NON_FLIP:
            auto_yes = auto_yes & (~flip)

        elig = elig.astype(bool) | auto_yes.astype(bool)
        return elig

    # ==============================
    # 0) Top table: best book per (game_id, side)
    # ==============================
    top_games = None
    edge_col = "edge"

    if has_full:
        tmp = full_df.copy()

        if "edge_used" in tmp.columns:
            edge_col = "edge_used"
        elif "edge_with_move" in tmp.columns:
            edge_col = "edge_with_move"
        else:
            edge_col = "edge"

        tmp = tmp.sort_values(edge_col, ascending=False)
        best_by_game_side = tmp.groupby(["game_id", "side"], as_index=False).head(1)

        if "sport" in best_by_game_side.columns:
            sort_cols = ["sport", "date", "home_team", "away_team", edge_col]
            asc = [True, True, True, True, False]
        else:
            sort_cols = ["date", "home_team", "away_team", edge_col]
            asc = [True, True, True, False]

        top_games = best_by_game_side.sort_values(sort_cols, ascending=asc).reset_index(drop=True)
        top_games.insert(0, "Rank", top_games.index + 1)

        top_games["model_p_pct"] = (_to_num(top_games.get("model_p")) * 100.0).round(2)
        top_games["implied_p_pct"] = (_to_num(top_games.get("implied_p")) * 100.0).round(2)
        top_games["edge_pct"] = (_to_num(top_games.get(edge_col)) * 100.0).round(2)

        for col in ["p_xgb", "p_margin", "p_elo"]:
            if col in top_games.columns:
                top_games[col + "_str"] = top_games[col].apply(_fmt_prob)
            else:
                top_games[col] = np.nan
                top_games[col + "_str"] = ""

        if "sport" not in top_games.columns:
            top_games["sport"] = "NHL"  # ✅ 변경

        mp = _to_num(top_games.get("model_p"))
        ip = _to_num(top_games.get("implied_p"))
        top_games["disagree_flag"] = ((mp >= 0.5) & (ip < 0.5)) | ((mp <= 0.5) & (ip > 0.5))
        top_games["signal"] = np.where(top_games["disagree_flag"], "MARKET FLIP", "")

        top_games["ev_amount"] = top_games.apply(
            lambda r: _calc_ev_amount(r.get("american_odds", 0.0), r.get("model_p", 0.0), stake=100.0),
            axis=1,
        )

        top_games["eligible_order"] = _eligible_by_edge(top_games, edge_col=edge_col)
        print("[EMAIL DEBUG] eligible count:", int(top_games["eligible_order"].sum()), " / ", len(top_games))

        top_games["is_no_bet"] = ~top_games["eligible_order"].astype(bool)

    # ==============================
    # 1) Plain text body
    # ==============================
    text_lines: list[str] = []
    text_lines.append("NHL Quant Bot – Daily Report")  # ✅ 변경
    text_lines.append("")

    if perf_summary:
        text_lines.append(f"0) Performance Summary (since {perf_summary.get('start_date', 'N/A')})")
        text_lines.append("--------------------------------------------------")
        settled = perf_summary.get("total_bets", 0)
        wins = perf_summary.get("wins", 0)
        losses = perf_summary.get("losses", 0)
        win_rate = perf_summary.get("win_rate", None)
        roi = perf_summary.get("roi", None)
        total_pnl = perf_summary.get("total_pnl", 0.0)

        text_lines.append(f"Settled bets : {settled} (wins {wins} / losses {losses})")
        if win_rate is not None:
            text_lines.append(f"Win rate     : {win_rate*100:.2f}%")
        if roi is not None:
            text_lines.append(f"ROI          : {roi*100:.2f}%")
        text_lines.append(f"Total PnL    : {total_pnl:.2f} USD")
        text_lines.append("")

    text_lines.append("1) Top Games by Model Edge (best bookmaker per side)")
    text_lines.append("--------------------------------------------------")

    if top_games is None or top_games.empty:
        text_lines.append("No odds / edge data available today.")
        text_lines.append("")
    else:
        for _, row in top_games.iterrows():
            matchup = f"{row['away_team']} @ {row['home_team']}"
            side = str(row["side"]).upper()
            odds_str = _fmt_american(row.get("american_odds", ""))

            ev_str = f"{float(row.get('ev_amount', 0.0)):+.2f}"
            tag = f" [{row['signal']}]" if str(row.get("signal", "")).strip() else ""

            extra = f" | XGB={row.get('p_xgb_str','')}, M={row.get('p_margin_str','')}, Elo={row.get('p_elo_str','')}"
            elig = " ✅ELIGIBLE" if bool(row.get("eligible_order", False)) else ""

            text_lines.append(
                f"#{int(row['Rank'])} {matchup} – {row['team']} ({side}) {odds_str}, "
                f"model={float(row['model_p_pct']):.2f}% vs implied={float(row['implied_p_pct']):.2f}%, "
                f"edge={float(row['edge_pct']):.2f}%, EV=${ev_str}{tag}{extra}{elig}"
            )
        text_lines.append("")

    # ==============================
    # 2) Today's bettable equity per game (Eligible only)
    # ==============================
    text_lines.append("2) Today's bettable equity per game (Eligible only)")
    text_lines.append("-------------------------------------------------")

    if top_games is None or top_games.empty:
        text_lines.append("No bets today (no top_games).")
        text_lines.append("")
        bet_rows_final = pd.DataFrame()
    else:
        bet_base = top_games.copy()
        bet_base["eligible_order"] = _eligible_by_edge(bet_base, edge_col=edge_col)
        bet_base = bet_base[bet_base["eligible_order"]].copy()

        n_elig = int(len(bet_base))
        try:
            total_daily_risk = float(NHL_TOTAL_DAILY_RISK) if NHL_TOTAL_DAILY_RISK is not None else 0.0
        except Exception:
            total_daily_risk = 0.0

        if n_elig <= 0:
            text_lines.append("Eligible bets: 0")
            text_lines.append(f"Total daily risk: ${total_daily_risk:.2f}")
            text_lines.append("Per-game equity: N/A")
            text_lines.append("")
            bet_rows_final = pd.DataFrame()
        else:
            per_game = total_daily_risk / n_elig
            text_lines.append(f"Eligible bets: {n_elig}")
            text_lines.append(f"Total daily risk: ${total_daily_risk:.2f}")
            text_lines.append(f"Per-game bettable equity: ${per_game:.2f}")
            text_lines.append("")
            bet_rows_final = bet_base.copy()

    plain_body = "\n".join(text_lines)

    # ==============================
    # 3) HTML body
    # ==============================
    html_parts: list[str] = []
    html_parts.append("<html>")
    html_parts.append("<body style='font-family:Arial,Helvetica,sans-serif; font-size:14px; line-height:1.4;'>")
    html_parts.append("<h2>NHL Quant Bot – Daily Report</h2>")  # ✅ 변경

    if perf_summary and perf_summary.get("total_bets", 0) > 0:
        html_parts.append("<h3>0) Performance Summary</h3>")
        html_parts.append("<ul>")
        html_parts.append(f"<li>Since: {perf_summary.get('start_date', 'N/A')}</li>")
        html_parts.append(
            f"<li>Settled bets: {perf_summary.get('total_bets', 0)} "
            f"(wins {perf_summary.get('wins', 0)} / losses {perf_summary.get('losses', 0)})</li>"
        )
        win_rate = perf_summary.get("win_rate", None)
        roi = perf_summary.get("roi", None)
        total_pnl = perf_summary.get("total_pnl", 0.0)
        if win_rate is not None:
            html_parts.append(f"<li>Win rate: {win_rate*100:.2f}%</li>")
        if roi is not None:
            html_parts.append(f"<li>ROI: {roi*100:.2f}% (PnL / total stake)</li>")
        html_parts.append(f"<li>Total PnL: {total_pnl:.2f} USD</li>")
        html_parts.append("</ul>")

    html_parts.append("<h3>1) Top Games by Model Edge (best bookmaker per side)</h3>")

    if top_games is None or top_games.empty:
        html_parts.append("<p>No odds / edge data available today.</p>")
    else:
        html_parts.append("<table style='border-collapse:collapse; width:100%; font-size:12px;'>")
        html_parts.append(
            "<tr>"
            "<th style='border:1px solid #ccc; padding:4px;'>Rank</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Sport</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Matchup</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Bet</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Book</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Odds</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>XGB</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Margin</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Elo</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Model %</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Implied %</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Edge %</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>EV ($)</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Signal</th>"
            "<th style='border:1px solid #ccc; padding:4px;'>Order?</th>"
            "</tr>"
        )

        td = "border:1px solid #ccc; padding:4px;"
        for _, row in top_games.iterrows():
            matchup = f"{row['away_team']} @ {row['home_team']}"
            side = str(row["side"]).upper()
            bet_label = f"{row['team']} ({side})"
            odds_str = _fmt_american(row.get("american_odds", ""))
            signal = str(row.get("signal", ""))
            ev_amount = float(row.get("ev_amount", 0.0))
            sport = row.get("sport", "NHL")  # ✅ 변경

            xgb_s = row.get("p_xgb_str", "")
            mar_s = row.get("p_margin_str", "")
            elo_s = row.get("p_elo_str", "")

            eligible = bool(row.get("eligible_order", False))
            is_no_bet = bool(row.get("is_no_bet", False))
            is_market_flip = (signal.strip().upper() == "MARKET FLIP")

            if eligible:
                style = "color:#0a7a0a; font-weight:bold;"
            elif is_no_bet:
                style = "color:#cc0000;"
            elif is_market_flip:
                style = "color:#0066cc; font-weight:bold;"
            else:
                style = ""

            order_tag = "YES" if eligible else "no"

            html_parts.append(
                "<tr>"
                f"<td style='{td} text-align:center; {style}'>{int(row['Rank'])}</td>"
                f"<td style='{td} {style}'>{sport}</td>"
                f"<td style='{td} {style}'>{matchup}</td>"
                f"<td style='{td} {style}'>{bet_label}</td>"
                f"<td style='{td} {style}'>{row.get('bookmaker','')}</td>"
                f"<td style='{td} text-align:right; {style}'>{odds_str}</td>"
                f"<td style='{td} text-align:right; {style}'>{xgb_s}</td>"
                f"<td style='{td} text-align:right; {style}'>{mar_s}</td>"
                f"<td style='{td} text-align:right; {style}'>{elo_s}</td>"
                f"<td style='{td} text-align:right; {style}'>{float(row['model_p_pct']):.2f}</td>"
                f"<td style='{td} text-align:right; {style}'>{float(row['implied_p_pct']):.2f}</td>"
                f"<td style='{td} text-align:right; {style}'>{float(row['edge_pct']):.2f}</td>"
                f"<td style='{td} text-align:right; {style}'>{ev_amount:+.2f}</td>"
                f"<td style='{td} text-align:center; {style}'>{signal}</td>"
                f"<td style='{td} text-align:center; {style}'>{order_tag}</td>"
                "</tr>"
            )
        html_parts.append("</table>")

    html_parts.append("<h3>2) Today's bettable equity per game (Eligible only)</h3>")

    if top_games is None or top_games.empty:
        html_parts.append("<p>No bets today (no top_games).</p>")
    else:
        bet_base = top_games.copy()
        bet_base["eligible_order"] = _eligible_by_edge(bet_base, edge_col=edge_col)
        bet_base = bet_base[bet_base["eligible_order"]].copy()
        n_elig = int(len(bet_base))

        try:
            total_daily_risk = float(NHL_TOTAL_DAILY_RISK) if NHL_TOTAL_DAILY_RISK is not None else 0.0
        except Exception:
            total_daily_risk = 0.0

        if n_elig <= 0:
            html_parts.append("<p>Eligible bets: 0</p>")
            html_parts.append(f"<p>Total daily risk: ${total_daily_risk:.2f}</p>")
            html_parts.append("<p>Per-game bettable equity: N/A</p>")
        else:
            per_game = total_daily_risk / n_elig
            html_parts.append(f"<p>Eligible bets: <b>{n_elig}</b></p>")
            html_parts.append(f"<p>Total daily risk: <b>${total_daily_risk:.2f}</b></p>")
            html_parts.append(f"<p>Per-game bettable equity: <b>${per_game:.2f}</b></p>")

    html_parts.append(
        "<div style='text-align:center; color:#888888; font-size:12px; margin-top:30px;'>"
        "Sent automatically by QuantLee Bot 🤖"
        "</div>"
    )
    html_parts.append("</body></html>")
    html_body = "\n".join(html_parts)

    # ==============================
    # 4) Send email
    # ==============================
    msg = MIMEMultipart("alternative")
    today_str = datetime.now().strftime("%m-%d-%Y")
    msg["Subject"] = Header(f"NHL Quant Bot – Daily Report, {today_str}", "utf-8")  # ✅ 변경
    msg["From"] = EMAIL_USER
    msg["To"] = EMAIL_TO

    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)