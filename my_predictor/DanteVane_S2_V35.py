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
        self.seeds = (37, 109, 233)
        self.params = {
            "objective": "huber",
            "alpha": 0.015,
            "num_leaves": 4,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "min_child_samples": 1000,
            "reg_alpha": 2.5,
            "reg_lambda": 30.0,
            "n_estimators": 40,
            "n_jobs": 4,
            "verbose": -1,
        }
        self.ema1_span = 260
        self.ema2_span = 165
        self.slow_span = 2000
        self.trade_band = 0.31

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    def _design(self, features: pd.DataFrame):
        r = {i: self._csrank(features[f"Feature.{i}"]) for i in range(1, 7)}

        # 1. Term-structure butterfly / convexity & consensus (unvisited rank geometry)
        butterfly = self._csrank((r[1] + r[4]) - (r[2] + r[3]))
        consensus = self._csrank(r[1] + r[2] + r[3] + r[4])

        # 2. Temporal rank acceleration (cross-section x time-series interaction)
        accel_2 = self._csrank(r[2] - r[2].ewm(span=120, min_periods=1).mean())
        accel_6 = self._csrank(r[6] - r[6].ewm(span=90, min_periods=1).mean())

        # 3. Quantile-gated Feature.6 (suppresses F6 during extreme F5 crowding)
        calm_gate = 1.0 - r[5].abs()
        gated_6 = self._csrank(r[6] * calm_gate)

        # 4. Harmonic saturation & anti-correlated tail product
        harm_26 = self._csrank((2.0 * r[2] * r[6]) / (r[2].abs() + r[6].abs() + 0.35))
        skew_56 = self._csrank(r[6] - 0.5 * r[5])

        blocks = [butterfly, consensus, accel_2, accel_6, gated_6, harm_26, skew_56, r[6]]
        tickers = list(r[1].columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, r[1].index

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
        z = p.div(p.std(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0).clip(-2.8, 2.8)
        target_pos = np.sign(z) * (z.abs() ** 1.08)
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
