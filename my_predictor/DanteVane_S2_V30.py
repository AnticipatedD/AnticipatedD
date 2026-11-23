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
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (11, 29, 53, 107)
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.015,
            "max_depth": 2,
            "extra_trees": False,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 900,
            "reg_lambda": 35.0,
            "reg_alpha": 1.5,
            "verbosity": -1,
            "nthread": 4,
        }
        self.n_rounds = 28
        self.fast_span = 360
        self.slow_span = 2160
        self.trade_band = 0.38

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

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}

        # Orthogonalized spreads remove redundant cross-sectional collinearity
        d24 = self._csrank(g.get(2) - g.get(4))
        d12 = self._csrank(self._orthogonalize(g.get(1) - g.get(2), g.get(2) - g.get(4)))
        d13 = self._csrank(self._orthogonalize(g.get(1) - g.get(3), g.get(2) - g.get(4)))

        return {
            "d24": d24,
            "d12": d12,
            "d13": d13,
            "i23_6": (r.get(2) - r.get(3)) * r.get(6),
            "i24_5": (r.get(2) - r.get(4)) * r.get(5),
            "i14_6": (r.get(1) - r.get(4)) * r.get(6),
            "p56": r.get(5) * r.get(6),
            "r5": r.get(5),
            "r6": r.get(6),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["d24"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["d24"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)

        dtrain = xgb.DMatrix(X[ok], label=y[ok])
        self.models = []
        for s in self.seeds:
            p = dict(self.params, seed=s)
            self.models.append(
                xgb.train(p, dtrain, num_boost_round=self.n_rounds)
            )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.models) > 0
        X, tickers, index = self._design(features)
        dtest = xgb.DMatrix(np.nan_to_num(X, nan=0.0))

        raw = np.mean([m.predict(dtest) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Dual-horizon band-pass extraction
        fast = pred.ewm(span=self.fast_span, min_periods=1).mean()
        slow = fast.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (fast - slow).fillna(0.0)

        # Cross-sectional de-meaning + scale-free tail sharpening
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        out = np.sign(p) * (p.abs() ** 1.35)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
        lad_21 = self._csrank((t_feat.get(2) - t_feat.get(1)) / (0.5 + e_all))

        geom_c1 = np.sign(curv_123) * np.sqrt(curv_123.abs() * r.get(6).abs())
        geom_c2 = np.sign(curv_234) * np.sqrt(curv_234.abs() * r.get(5).abs())

        blocks = [
            curv_123,
            curv_234,
            lad_21,
            geom_c1,
            geom_c2,
            t_feat.get(3) * t_feat.get(6),
            t_feat.get(4) * t_feat.get(5),
            r.get(6) - r.get(2),
            t_feat.get(6),
        ]

        tickers = list(curv_123.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, curv_123.index

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
        target_pos = np.sign(z) * (z.abs() ** 1.45)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            curr = next(iter(arr)).copy()
            held[:1] = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
