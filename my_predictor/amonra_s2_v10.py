# /// script
# dependencies = ["xgboost"]
# ///

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V11Predictor(Predictor):
    def __init__(self):
        self.model = None
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.06,
            "subsample": 0.70,
            "colsample_bytree": 0.55,
            "min_child_weight": 350,
            "reg_lambda": 6.0,
            "reg_alpha": 1.5,
            "seed": 42,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 65
        self.smooth_span = 45
        self.tilt_span = 450

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _cs_robust_z(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = df.sub(med, axis=0).abs().median(axis=1).replace(0.0, 1.0)
        return df.sub(med, axis=0).div(mad * 1.4826, axis=0).clip(-4.0, 4.0)

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}

        s24 = g.get(2) - g.get(4)
        s12 = g.get(1) - g.get(2)
        s13 = g.get(1) - g.get(3)

        return {
            "d24": self._csrank(s24),
            "d12": self._csrank(s12),
            "d13": self._csrank(s13),
            "m23_6": (r.get(2) - r.get(3)) * r.get(6),
            "m24_5": (r.get(2) - r.get(4)) * r.get(5),
            "p56": r.get(5) * r.get(6),
            "r6": r.get(6),
            "r5": r.get(5),
            "nl_14_6": self._csrank(g.get(1) + g.get(4)) * self._csrank(g.get(6).abs()),
            "nl_35_2": np.tanh(2.0 * r.get(3)) * (r.get(5) - r.get(2)),
            "ratio_16": self._csrank(self._cs_robust_z(g.get(1))) - r.get(6),
            "ts_delta_24": self._csrank(s24.diff(5).fillna(0.0)),
            "ts_rev_13": -self._csrank(g.get(1) - g.get(1).ewm(span=12, min_periods=1).mean()) * r.get(3),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b.get("d24").columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b.get("d24").index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(
            self.params,
            xgb.DMatrix(X[ok], label=y[ok]),
            num_boost_round=self.n_rounds,
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None, "Model must be trained before calling predict."
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - tilt).fillna(0.0)

        mean = pred.mean(axis=1)
        std = pred.std(axis=1).replace(0.0, 1.0)
        pred = pred.sub(mean, axis=0).div(std, axis=0).fillna(0.0)
        pred = pred.clip(-4.0, 4.0)

        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
