import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRa_S2_V26Predictor(Predictor):
    def __init__(self):
        self.xgb_models = []
        self.ridge_w = None
        self.seeds = (19, 83, 151)
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_weight": 1100,
            "reg_lambda": 25.0,
            "reg_alpha": 2.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 24
        self.ridge_alpha = 5000.0
        self.fast_span = 72
        self.slow_span = 720

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # Light pre-filtering suppresses single-bar threshold chatter in trees
        g = {
            i: features[f"Feature.{i}"].ewm(span=12, min_periods=1).mean()
            for i in range(1, 7)
        }
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        return {
            "d24": self._csrank(g[2] - g[4]),
            "d12": self._csrank(g[1] - g[2]),
            "d13": self._csrank(g[1] - g[3]),
            "m23_6": (r[2] - r[3]) * r[6],
            "m24_5": (r[2] - r[4]) * r[5],
            "p56": r[5] * r[6],
            "r6": r[6],
            "r5": r[5],
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
        X_ok = X[ok]
        y_ok = y[ok]

        # 1. Closed-form L2 Ridge model (continuous surface -> ultra-low turnover)
        XtX = X_ok.T @ X_ok + self.ridge_alpha * np.eye(X_ok.shape[1])
        self.ridge_w = np.linalg.solve(XtX, X_ok.T @ y_ok)

        # 2. Multi-seed shallow XGBoost ensemble (captures nonlinear interactions)
        dtrain = xgb.DMatrix(X_ok, label=y_ok)
        self.xgb_models = []
        for s in self.seeds:
            p = dict(self.params, seed=s)
            self.xgb_models.append(
                xgb.train(p, dtrain, num_boost_round=self.n_rounds)
            )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.xgb_models) > 0 and self.ridge_w is not None
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        # Blend continuous linear alpha (50%) + nonlinear tree alpha (50%)
        raw_ridge = X_clean @ self.ridge_w
        dtest = xgb.DMatrix(X_clean)
        raw_xgb = np.mean([m.predict(dtest) for m in self.xgb_models], axis=0)

        # Standardize scales before blending so both contribute equally
        raw_ridge = raw_ridge / (np.std(raw_ridge) + 1e-12)
        raw_xgb = raw_xgb / (np.std(raw_xgb) + 1e-12)
        raw = 0.5 * raw_ridge + 0.5 * raw_xgb

        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Faster signal extraction + partial baseline removal preserves persistent alpha
        fast = pred.ewm(span=self.fast_span, min_periods=1).mean()
        slow = fast.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (fast - 0.50 * slow).fillna(0.0)

        # Cross-sectional z-score + soft deadzone to cut low-conviction turnover drag
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        std = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(std, axis=0).fillna(0.0).clip(-3.0, 3.0)

        # Soft-threshold noise inside [-0.35, +0.35] std while boosting strong tails
        z_clean = np.sign(z) * np.maximum(z.abs() - 0.35, 0.0)
        out = np.sign(z_clean) * (z_clean.abs() ** 1.35)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
