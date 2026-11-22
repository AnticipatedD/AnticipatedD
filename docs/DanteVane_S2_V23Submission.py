# /// script
# dependencies = [
#   "lightgbm>=4.0",
#   "numpy",
#   "pandas",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (17, 59, 131)
        self.params = {
            "objective": "huber",
            "alpha": 0.015,
            "boosting_type": "gbdt",
            "extra_trees": True,
            "num_leaves": 4,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.70,
            "min_child_samples": 950,
            "reg_alpha": 1.5,
            "reg_lambda": 20.0,
            "n_estimators": 28,
            "n_jobs": 4,
            "verbose": -1,
        }
        self.ema1_span = 360
        self.ema2_span = 240
        self.slow_span = 2400
        self.trade_band = 0.25

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _orthogonalize(target_df: pd.DataFrame, basis_df: pd.DataFrame) -> pd.DataFrame:
        num = (target_df * basis_df).sum(axis=1)
        den = (basis_df * basis_df).sum(axis=1).replace(0.0, np.nan)
        beta = (num / den).fillna(0.0)
        return target_df.sub(basis_df.mul(beta, axis=0), axis=0)

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        d24 = g[2] - g[4]
        s24 = self._csrank(d24)
        s12 = self._csrank(self._orthogonalize(g[1] - g[2], d24))
        s13 = self._csrank(self._orthogonalize(g[1] - g[3], d24))

        blocks = [
            s24,
            s12,
            s13,
            (r[2] - r[3]) * r[6],
            (r[2] - r[4]) * r[5],
            (r[1] - r[4]) * r[6],
            r[5] * r[6],
            r[5],
            r[6],
        ]

        tickers = list(s24.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, s24.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.models = []
        for s in self.seeds:
            p = dict(self.params, random_state=s)
            est = lgb.LGBMRegressor(**p)
            est.fit(X_ok, y_ok)
            self.models.append(est)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw = np.mean([m.predict(X_clean) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Second-order causal low-pass filter + full baseline de-trending
        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        # Scale-free conviction sharpening (no per-bar std division)
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        target_pos = np.sign(p) * (p.abs() ** 1.35)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Causal Hysteresis deadband scaled relative to signal RMS to cut 5 bp turnover
        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            rms = float(np.sqrt(np.mean(arr ** 2))) + 1e-12
            band = self.trade_band * rms
            curr = arr[0].copy()
            held[0] = curr
            for t in range(1, n_rows):
                diff = arr[t] - curr
                curr = curr + np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
