# /// script
# dependencies = ["xgboost"]
# ///

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V12Predictor(Predictor):
    def __init__(self):
        self.model = None
        self.ridge_w = None
        self.feat_mean = None
        self.feat_std = None
        self.feature_names = []

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
        self.ridge_alpha = 5000.0
        self.xgb_weight = 0.70
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
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        s24 = g - g
        s12 = g - g
        s13 = g - g
        r_spread_24 = r - r

        return {
            # 1-3: Core ranked spreads
            "d24": self._csrank(s24),
            "d12": self._csrank(s12),
            "d13": self._csrank(s13),
            # 4-6: Multi-factor rank interactions
            "m23_6": (r - r) * r,
            "m24_5": (r - r) * r,
            "p56": r * r,
            # 7-8: Base rank anchors
            "r6": r,
            "r5": r,
            # 9-11: Orthogonalized non-linear & dispersion novelty signals
            "nl_14_6": self._csrank(g + g) * self._csrank(g.abs()),
            "nl_35_2": np.tanh(2.0 * r) * (r - r),
            "ratio_16": self._csrank(self._cs_robust_z(g)) * r,
            # 12-13: Orthogonalized temporal velocity & mean-reversion signals
            "ts_delta_24": self._csrank(r_spread_24.diff(3).fillna(0.0)),
            "ts_rev_13": -self._csrank(g - g.ewm(span=12, min_periods=1).mean()) * r,
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        self.feature_names = list(b.keys())
        tickers = list(b["d24"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["d24"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        # 1. Fit XGBoost model
        dtrain = xgb.DMatrix(X_ok, label=y_ok, feature_names=self.feature_names)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_rounds,
        )

        # 2. Fit L2-regularized Ridge model (closed-form)
        self.feat_mean = X_ok.mean(axis=0)
        self.feat_std = np.where(X_ok.std(axis=0) == 0.0, 1.0, X_ok.std(axis=0))
        X_std = (X_ok - self.feat_mean) / self.feat_std
        y_centered = y_ok - y_ok.mean()

        n_feats = X_std.shape
        gram = X_std.T @ X_std + self.ridge_alpha * np.eye(n_feats)
        self.ridge_w = np.linalg.solve(gram, X_std.T @ y_centered)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None and self.ridge_w is not None, "Call train() first."
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X)

        # Blend 70% XGBoost + 30% Ridge
        dtest = xgb.DMatrix(X_clean, feature_names=self.feature_names)
        raw_xgb = self.model.predict(dtest)

        X_std = (X_clean - self.feat_mean) / self.feat_std
        raw_ridge = X_std @ self.ridge_w

        xgb_norm = (raw_xgb - raw_xgb.mean()) / (raw_xgb.std() + 1e-8)
        ridge_norm = (raw_ridge - raw_ridge.mean()) / (raw_ridge.std() + 1e-8)
        raw = self.xgb_weight * xgb_norm + (1.0 - self.xgb_weight) * ridge_norm

        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Fast-capture EMA smoothing & structural drift removal
        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - tilt).fillna(0.0)

        # Volatility stabilization & clipping
        mean = pred.mean(axis=1)
        std = pred.std(axis=1).replace(0.0, 1.0)
        pred = pred.sub(mean, axis=0).div(std, axis=0).fillna(0.0)
        pred = pred.clip(-4.0, 4.0)

        # Mandatory cross-sectional de-meaning
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
