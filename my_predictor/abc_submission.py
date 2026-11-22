import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V20Predictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = [42, 1024, 2025]
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
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        # Slow-horizon smoothed ranks to align training features with slow alpha
        r_slow = {i: r[i].ewm(span=180, min_periods=1).mean() for i in range(1, 7)}

        return {
            # Original V8 high-IC core
            "d24": self._csrank(g[2] - g[4]),
            "d12": self._csrank(g[1] - g[2]),
            "d13": self._csrank(g[1] - g[3]),
            "m23_6": (r[2] - r[3]) * r[6],
            "m24_5": (r[2] - r[4]) * r[5],
            "p56": r[5] * r[6],
            "r6": r[6],
            "r5": r[5],
            # Slow structural regime features (reduces IC Std & boosts Novelty)
            "sd24": self._csrank(r_slow[2] - r_slow[4]),
            "sd12": self._csrank(r_slow[1] - r_slow[2]),
            "sm23_6": (r_slow[2] - r_slow[3]) * r_slow[6],
            "sp56": r_slow[5] * r_slow[6],
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["d24"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["d24"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        # Smooth target slightly during training to suppress single-bar noise,
        # then rank cross-sectionally so outliers don't skew tree splits
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

        # Ultra-low turnover trend extraction
        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - 0.65 * tilt).fillna(0.0)

        # Cross-sectional rank output guarantees uniform Concentration and zero outliers
        out = self._csrank(pred)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
