import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V21Predictor(Predictor):
    def __init__(self):
        self.model = None
        # Combines V8's raw-target magnitude learning with V11's lower IC Std regularization
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.035,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_weight": 1000,
            "reg_lambda": 20.0,
            "reg_alpha": 2.0,
            "seed": 42,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 26
        self.smooth_span = 420
        self.tilt_span = 2200

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}
        return {
            "d24": self._csrank(g.get(2) - g.get(4)),
            "d12": self._csrank(g.get(1) - g.get(2)),
            "d13": self._csrank(g.get(1) - g.get(3)),
            "m23_6": (r.get(2) - r.get(3)) * r.get(6),
            "m24_5": (r.get(2) - r.get(4)) * r.get(5),
            "p56": r.get(5) * r.get(6),
            "r6": r.get(6),
            "r5": r.get(5),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["d24"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["d24"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        # Raw target (from V8) preserves return magnitude weighting -> higher IC (0.038+)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(
            self.params,
            xgb.DMatrix(X[ok], label=y[ok]),
            num_boost_round=self.n_rounds,
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Extended V8 trend filter -> cuts turnover drag & preserves V11's Global Novelty
        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - tilt).fillna(0.0)

        # Cross-sectional de-mean (no rank squashing -> keeps Concentration low & Sharpe up)
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

        # Cubic conviction scaling: suppresses tiny noisy weights near zero while
        # letting high-conviction alpha signals drive portfolio PnL
        scale = pred.std(axis=1).replace(0.0, np.nan)
        z = pred.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        conv = z * (z.abs() ** 0.5)
        return conv.sub(conv.mean(axis=1), axis=0).fillna(0.0)
