# nhl_goal_diff_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import NHL_MARGIN_MODEL_PATH


@dataclass
class NHLGoalDiffEvalResult:
    n_games: int
    mean_mae: float
    mean_rmse: float
    residual_std: float
    half_life_days: float | None = None


class NHLGoalDiffModel:
    """
    - target: goal_diff = home_score - away_score
    NBA margin 모델과 동일 구조:
      1) half-life 제거(호환 파라미터만)
      2) 최신 시즌만 season_games 기반 신뢰도 weight(1.0~1.2)
      3) residual_std 저장
      4) goal_diff -> win prob(sigmoid) 보조 확률 생성
    """

    def __init__(self):
        self.model: XGBRegressor | None = None
        self.feature_cols: List[str] = []
        self.residual_std: float = 2.0  # NHL은 점수 스케일이 작아서 기본도 작게

    @staticmethod
    def _season_reliability_weight(season_games_min) -> float:
        try:
            g = float(season_games_min)
        except Exception:
            return 1.0

        if g < 15:
            return 1.00
        if g < 30:
            return 1.10
        return 1.20

    @staticmethod
    def _compute_sample_weights_by_season_games(df: pd.DataFrame) -> np.ndarray:
        if df.empty or "season" not in df.columns:
            return np.ones(len(df), dtype=float)

        current_season = int(pd.to_numeric(df["season"], errors="coerce").max())
        s = pd.to_numeric(df["season"], errors="coerce").fillna(current_season).astype(int)

        if ("season_games_home" not in df.columns) or ("season_games_away" not in df.columns):
            return np.ones(len(df), dtype=float)

        g_home = pd.to_numeric(df["season_games_home"], errors="coerce").fillna(0).values
        g_away = pd.to_numeric(df["season_games_away"], errors="coerce").fillna(0).values
        g_min = np.minimum(g_home, g_away)

        w = np.ones(len(df), dtype=float)
        is_current = (s.values == current_season)
        idxs = np.where(is_current)[0]
        for i in idxs:
            w[i] = NHLGoalDiffModel._season_reliability_weight(g_min[i])
        return w

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def train_walkforward(
        self,
        df_train: pd.DataFrame,
        feature_cols: List[str],
        n_splits: int = 5,
        random_state: int = 42,
        half_life_days: float | None = None,  # 호환용
        save: bool = True,
    ) -> NHLGoalDiffEvalResult:
        df = df_train.sort_values("date").reset_index(drop=True).copy()
        df["date"] = pd.to_datetime(df["date"])

        if not {"home_score", "away_score"}.issubset(df.columns):
            raise ValueError("[NHLGoalDiffModel] df_train must contain 'home_score' and 'away_score'.")

        y = (pd.to_numeric(df["home_score"], errors="coerce") - pd.to_numeric(df["away_score"], errors="coerce")).astype(float).values
        X = df[feature_cols].apply(pd.to_numeric, errors="coerce").values

        w_rows = self._compute_sample_weights_by_season_games(df)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        maes, rmses = [], []

        preds_oof = np.full_like(y, fill_value=np.nan, dtype=float)

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X, y), start=1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            w_tr = w_rows[tr_idx]

            model = XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr, sample_weight=w_tr)

            y_val_pred = model.predict(X_val).astype(float)
            preds_oof[val_idx] = y_val_pred

            mae = mean_absolute_error(y_val, y_val_pred)
            rmse = mean_squared_error(y_val, y_val_pred, squared=False)

            maes.append(mae)
            rmses.append(rmse)
            print(f"[NHL GOALDIFF WF] Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}")

        mean_mae = float(np.mean(maes)) if maes else np.nan
        mean_rmse = float(np.mean(rmses)) if rmses else np.nan

        residuals = y - preds_oof
        residuals = residuals[~np.isnan(residuals)]
        residual_std = float(np.std(residuals)) if len(residuals) else 2.0
        if residual_std <= 1e-6:
            residual_std = 2.0

        print(
            f"[NHL GOALDIFF WF] mean MAE={mean_mae:.3f}, mean RMSE={mean_rmse:.3f}, residual_std={residual_std:.3f}"
        )

        final_model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
        final_model.fit(X, y, sample_weight=w_rows)

        self.model = final_model
        self.feature_cols = list(feature_cols)
        self.residual_std = residual_std

        if save:
            joblib.dump(
                {"model": self.model, "feature_cols": self.feature_cols, "residual_std": self.residual_std},
                NHL_MARGIN_MODEL_PATH,
            )
            print(f"[NHLGoalDiffModel] Final goal diff model saved to {NHL_MARGIN_MODEL_PATH}")

        return NHLGoalDiffEvalResult(
            n_games=len(df),
            mean_mae=mean_mae,
            mean_rmse=mean_rmse,
            residual_std=residual_std,
            half_life_days=None,
        )

    @staticmethod
    def load_from_disk() -> "NHLGoalDiffModel":
        obj = joblib.load(NHL_MARGIN_MODEL_PATH)
        m = NHLGoalDiffModel()
        m.model = obj["model"]
        m.feature_cols = obj["feature_cols"]
        m.residual_std = float(obj.get("residual_std", 2.0))
        return m

    def predict_goal_diff(self, df_today: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or not self.feature_cols:
            raise ValueError("NHLGoalDiffModel.predict_goal_diff: model not loaded/trained.")

        missing = [c for c in self.feature_cols if c not in df_today.columns]
        if missing:
            raise ValueError(f"[NHL GOALDIFF] Missing features in input: {missing}")

        X = df_today[self.feature_cols].apply(pd.to_numeric, errors="coerce").values
        gd_pred = self.model.predict(X).astype(float)

        scale = max(float(self.residual_std), 0.5)  # NHL 스케일 방어
        p_home = self._sigmoid(gd_pred / scale)
        p_home = np.clip(p_home, 0.01, 0.99)

        out = df_today.copy()
        out["goal_diff_pred"] = gd_pred
        out["p_home_win_from_goal_diff"] = p_home
        return out