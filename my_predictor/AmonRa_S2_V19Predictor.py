import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V19Predictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = [42, 1024, 2025]
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_weight": 500,
            "reg_lambda": 12.0,
            "reg_alpha": 2.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 45
        self.smooth_span = 8
        self.tilt_span = 60

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _cszscore(df: pd.DataFrame) -> pd.DataFrame:
        mean = df.mean(axis=1)
        std = df.std(axis=1).replace(0.0, np.nan)
        return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0).clip(-3.0, 3.0)

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        # Base cross-sectional spreads & interactions
        d24 = self._csrank(g[2] - g[4])
        d12 = self._csrank(g[1] - g[2])
        d13 = self._csrank(g[1] - g[3])
        d34 = self._csrank(g[3] - g[4])

        # Short-horizon temporal momentum of ranks (adds novelty & reduces lag)
        dr1 = r[1] - r[1].ewm(span=10, min_periods=1).mean()
        dr2 = r[2] - r[2].ewm(span=10, min_periods=1).mean()
        dr5 = r[5] - r[5].ewm(span=15, min_periods=1).mean()

        return {
            "d24": d24,
            "d12": d12,
            "d13": d13,
            "d34": d34,
            "m23_6": (r[2] - r[3]) * r[6],
            "m24_5": (r[2] - r[4]) * r[5],
            "m14_6": (r[1] - r[4]) * r[6],
            "p56": r[5] * r[6],
            "r5": r[5],
            "r6": r[6],
            "dr1": self._csrank(dr1),
            "dr2": self._csrank(dr2),
            "dr5": self._csrank(dr5),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["d24"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["d24"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        # Rank-normalize target cross-sectionally so outliers do not distort gradients
        y_cs = self._csrank(target[tickers]).values.reshape(-1)
        ok = np.isfinite(y_cs) & np.isfinite(X).all(axis=1)

        dtrain = xgb.DMatrix(X[ok], label=y_cs[ok])
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

        # Multi-seed ensemble to lower IC Std
        raw = np.mean([m.predict(dtest) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Light smoothing to cut turnover without destroying signal phase
        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - 0.5 * tilt).fillna(0.0)

        # Cross-sectional z-score with winsorization stabilizes Concentration & Sharpe
        return self._cszscore(pred)
