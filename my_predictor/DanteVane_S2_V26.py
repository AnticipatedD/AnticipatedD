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
        self.seeds = (31, 89, 173)
        self.params = {
            "objective": "huber",
            "alpha": 0.015,
            "boosting_type": "gbdt",
            "extra_trees": False,
            "num_leaves": 5,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.70,
            "min_child_samples": 950,
            "reg_alpha": 2.0,
            "reg_lambda": 22.0,
            "n_estimators": 42,
            "n_jobs": 4,
            "verbose": -1,
        }
        self.ema1_span = 190
        self.ema2_span = 120
        self.slow_span = 1600
        self.trade_band = 0.36

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
        t_feat = {i: self._robust_tanh(g.get(i)) for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}

        # Nonlinear MAD-copula ratios & asymmetric cross-feature products
        energy = (t_feat.get(1).abs() + t_feat.get(3).abs() + t_feat.get(4).abs()) / 3.0
        rat_24 = self._csrank((t_feat.get(2) - t_feat.get(4)) / (0.5 + energy))
        rat_31 = self._csrank((t_feat.get(3) - t_feat.get(1)) / (0.5 + energy))
        rat_42 = self._csrank((t_feat.get(4) - t_feat.get(2)) / (0.5 + energy))

        # Signed geometric interactions
        geom_246 = np.sign(t_feat.get(2) - t_feat.get(4)) * np.sqrt(
            (t_feat.get(2) - t_feat.get(4)).abs() * r.get(6).abs()
        )
        geom_315 = np.sign(t_feat.get(3) - t_feat.get(1)) * np.sqrt(
            (t_feat.get(3) - t_feat.get(1)).abs() * r.get(5).abs()
        )

        blocks = [
            rat_24,
            rat_31,
            rat_42,
            geom_246,
            geom_315,
            t_feat.get(5) * t_feat.get(6),
            t_feat.get(2) * t_feat.get(6),
            r.get(6) - r.get(5),
            t_feat.get(6),
        ]

        tickers = list(rat_24.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, rat_24.index

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

        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.35)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            curr = arr[0].copy()
            held[0] = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
