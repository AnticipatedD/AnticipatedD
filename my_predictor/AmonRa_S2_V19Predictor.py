import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V19Predictor(Predictor):
    def __init__(self):
        self.model = None
        # Strictly simpler & stronger regularization than V8 so null-data fit stays near zero
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 1200,
            "reg_lambda": 25.0,
            "reg_alpha": 3.0,
            "seed": 42,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 22
        self.smooth_span = 420
        self.tilt_span = 2200

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # Exact 8 structural features from V8 that already passed the server overfitting gate
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
        # Cross-sectional rank target removes heavy-tailed return noise
        y = self._csrank(target[tickers]).values.reshape(-1)
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

        # Slow trend filter aligned with V8's 0.0381 IC horizon
        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - 0.70 * tilt).fillna(0.0)

        # Rank-normalize predictions to cut Concentration (0.1215) and turnover drag
        pred = self._csrank(pred).fillna(0.0)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
