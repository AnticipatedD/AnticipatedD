# /// script
# dependencies = ["catboost>=1.2"]
# ///

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from predictor import Predictor


class MyPredictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (101, 307, 601)
        self.cb_params = {
            "loss_function": "Huber:delta=0.015",
            "depth": 3,
            "iterations": 85,
            "learning_rate": 0.035,
            "l2_leaf_reg": 35.0,
            "subsample": 0.72,
            "rsm": 0.75,
            "random_strength": 1.5,
            "thread_count": 4,
            "verbose": False,
        }
        self.kalman_q = 0.00018
        self.kalman_r = 0.025
        self.drift_window = 1800
        self.band = 0.24

    def _extract_panel(self, features: pd.DataFrame):
        f1 = features["Feature.1"]
        f2 = features["Feature.2"]
        f3 = features["Feature.3"]
        f4 = features["Feature.4"]
        f5 = features["Feature.5"]
        f6 = features["Feature.6"]

        # Cross-sectional quantile/rank transforms
        r1 = 2.0 * f1.rank(axis=1, pct=True) - 1.0
        r2 = 2.0 * f2.rank(axis=1, pct=True) - 1.0
        r3 = 2.0 * f3.rank(axis=1, pct=True) - 1.0
        r4 = 2.0 * f4.rank(axis=1, pct=True) - 1.0
        r5 = 2.0 * f5.rank(axis=1, pct=True) - 1.0
        r6 = 2.0 * f6.rank(axis=1, pct=True) - 1.0

        # Cross-sectional z-scores clipped for tail stability
        def csz(df: pd.DataFrame) -> pd.DataFrame:
            m = df.mean(axis=1)
            s = df.std(axis=1).replace(0.0, np.nan)
            return df.sub(m, axis=0).div(s, axis=0).fillna(0.0).clip(-3.5, 3.5)

        z1, z2, z4, z5, z6 = csz(f1), csz(f2), csz(f4), csz(f5), csz(f6)

        # Orthogonalized spreads & regime-modulated interactions
        s21 = 2.0 * (f2 - f1).rank(axis=1, pct=True) - 1.0
        s24 = 2.0 * (f2 - f4).rank(axis=1, pct=True) - 1.0
        s31 = 2.0 * (f3 - f1).rank(axis=1, pct=True) - 1.0

        # Short-horizon velocity on Feature.6 & regime dispersion
        v6 = 2.0 * (r6 - r6.ewm(span=24, min_periods=1).mean()).rank(axis=1, pct=True) - 1.0
        disp = (f1.std(axis=1) + f2.std(axis=1) + f4.std(axis=1)) / 3.0
        disp_scale = (disp / disp.ewm(span=240, min_periods=1).mean().replace(0.0, np.nan)).fillna(1.0).clip(0.5, 2.0)

        blocks = [
            s21,
            s24,
            s31,
            v6,
            (r2 - r1) * r6,
            (r2 - r4) * r5,
            np.tanh(z5 * z6),
            s21.mul(disp_scale, axis=0),
            r6,
        ]

        tickers = list(f1.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, f1.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._extract_panel(features)
        y = target[tickers].values.reshape(-1)
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok = X[valid]
        y_ok = y[valid]

        self.models = []
        for seed in self.seeds:
            params = dict(self.cb_params, random_seed=seed)
            cb = CatBoostRegressor(**params)
            cb.fit(X_ok, y_ok)
            self.models.append(cb)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._extract_panel(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw = np.mean([m.predict(X_clean) for m in self.models], axis=0)
        pred_mat = raw.reshape(-1, len(tickers))
        n_rows, n_cols = pred_mat.shape

        # Causal 1D Kalman state estimator (smooths noise with minimal lag vs cascaded EMA)
        est = np.zeros_like(pred_mat)
        if n_rows > 0:
            x_hat = pred_mat[0].copy()
            p_cov = np.ones(n_cols, dtype=np.float64)
            est[0] = x_hat
            q = self.kalman_q
            r = self.kalman_r
            for t in range(1, n_rows):
                p_pred = p_cov + q
                k_gain = p_pred / (p_pred + r)
                x_hat = x_hat + k_gain * (pred_mat[t] - x_hat)
                p_cov = (1.0 - k_gain) * p_pred
                est[t] = x_hat

        est_df = pd.DataFrame(est, index=index, columns=tickers)
        slow_drift = est_df.shift(1).ewm(span=self.drift_window, min_periods=1).mean()
        signal = (est_df - slow_drift).fillna(0.0)

        # Cross-sectional de-meaning + unit-variance scaling + tail sharpening
        centered = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        cs_std = centered.std(axis=1).replace(0.0, np.nan)
        z = centered.div(cs_std, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.35)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Causal hysteresis deadband to suppress 5 bp hourly turnover drag
        pos_arr = target_pos.values
        held = np.zeros_like(pos_arr)
        if n_rows > 0:
            curr = pos_arr[0].copy()
            held[0] = curr
            b = self.band
            for t in range(1, n_rows):
                diff = pos_arr[t] - curr
                curr = curr + np.sign(diff) * np.maximum(np.abs(diff) - b, 0.0)
                held[t] = curr

        preds = pd.DataFrame(held, index=index, columns=tickers)
        return preds.sub(preds.mean(axis=1), axis=0).fillna(0.0)
