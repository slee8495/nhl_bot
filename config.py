# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# Risk / Bankroll
# =========================
NHL_TOTAL_DAILY_RISK = 100


# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

NHL_TRAIN_PATH = os.path.join(DATA_DIR, "nhl_train.csv")
NHL_TODAY_PATH = os.path.join(DATA_DIR, "nhl_today.csv")

NHL_MODEL_PATH = os.path.join(MODEL_DIR, "nhl_xgb_win.pkl")
NHL_MARGIN_MODEL_PATH = os.path.join(MODEL_DIR, "nhl_xgb_margin.pkl")



# =========================
# Email
# =========================
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")


# =========================
# Odds API
# =========================
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE_URL = os.getenv("ODDS_API_BASE_URL")


# =========================
# balldontlie API 
# =========================
BALLDONTLIE_API_KEY = os.getenv("BALLDONTLIE_API_KEY", "")
BALLDONTLIE_BASE_URL = os.getenv("BALLDONTLIE_BASE_URL")

NBA_BASE_URL = f"{BALLDONTLIE_BASE_URL}/nba/v1"
NHL_BASE_URL = f"{BALLDONTLIE_BASE_URL}/nhl/v1"


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default