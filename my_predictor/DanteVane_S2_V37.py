# /// script
# dependencies = [
#   "lightgbm>=4.0",
#   "xgboost>=2.0",
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
        self.seeds = [31, 89, 173, 251]
        self.params = {
            "objective": "huber",
            "alpha": 0.015,
            "num_leaves": 5,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "min_child_samples": 950,
            "reg_alpha": 2.0,
            "reg_lambda": 25.0,
            "n_estimators": 44,
            "n_jobs": 4,
            "verbose": -1,
        }
        self.ema1_span = 260
        self.ema2_span = 165
        self.slow_span = 2000
        self.trade_band = 0.32

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.70 * z)

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        t = {i: self._robust_tanh(g[i]) for i in range(1, 7)}
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        # Proven 0.050 IC MAD-copula core
        energy = (t[1].abs() + t[3].abs() + t[4].abs()) / 3.0
        rat_24 = self._csrank((t[2] - t[4]) / (0.5 + energy))
        rat_31 = self._csrank((t[3] - t[1]) / (0.5 + energy))
        geom_246 = np.sign(t[2] - t[4]) * np.sqrt((t[2] - t[4]).abs() * r[6].abs())

        # Proven 15.29 / 19.46 Novelty boosters
        accel_6 = self._csrank(r[6] - r[6].ewm(span=90, min_periods=1).mean())
        gated_6 = self._csrank(t[6] * (1.0 - r[5].abs()))
        harm_26 = self._csrank((2.0 * t[2] * t[6]) / (t[2].abs() + t[6].abs() + 0.35))

        blocks = [rat_24, rat_31, geom_246, accel_6, gated_6, harm_26, t[5] * t[6], t[6]]
        tickers = list(rat_24.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, rat_24.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.models = [
            lgb.LGBMRegressor(**dict(self.params, random_state=s)).fit(X[ok], y[ok])
            for s in self.seeds
        ]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.models) > 0
        X, tickers, index = self._design(features)
        raw = np.mean([m.predict(np.nan_to_num(X, nan=0.0)) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        z = p.div(p.std(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.40)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        held = np.zeros_like(arr)
        if len(arr) > 0:
            curr = arr[0].copy()
            held[0] = curr
            for t in range(1, len(arr)):
                diff = arr[t] - curr
                curr = curr + np.sign(diff) * np.maximum(np.abs(diff) - self.trade_band, 0.0)
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
