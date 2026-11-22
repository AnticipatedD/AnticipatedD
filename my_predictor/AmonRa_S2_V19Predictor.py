import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V19Predictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (42, 100, 2026)
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "min_child_weight": 800,
            "reg_lambda": 15.0,
            "reg_alpha": 2.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 35
        self.smooth_span = 450
        self.tilt_span = 2400

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}
        rs = {i: r.get(i).ewm(span=180, min_periods=1).mean() for i in range(1, 7)}

        return {
            "d24": self._csrank(g.get(2) - g.get(4)),
            "d12": self._csrank(g.get(1) - g.get(2)),
            "d13": self._csrank(g.get(1) - g.get(3)),
            "m23_6": (r.get(2) - r.get(3)) * r.get(6),
            "m24_5": (r.get(2) - r.get(4)) * r.get(5),
            "p56": r.get(5) * r.get(6),
            "r6": r.get(6),
            "r5": r.get(5),
            "sd24": self._csrank(rs.get(2) - rs.get(4)),
            "sd12": self._csrank(rs.get(1) - rs.get(2)),
            "sm23_6": (rs.get(2) - rs.get(3)) * rs.get(6),
            "sp56": rs.get(5) * rs.get(6),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["d24"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["d24"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y_smooth = target[tickers].ewm(span=12, min_periods=1).mean()
        y = self._csrank(y_smooth).values.reshape(-1)
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

        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - 0.65 * tilt).fillna(0.0)

        pred = self._csrank(pred).fillna(0.0)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
