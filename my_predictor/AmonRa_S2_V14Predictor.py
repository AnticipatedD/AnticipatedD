import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V15Predictor(Predictor):
    def __init__(self):
        super().__init__()
        self.gbdt_model = None
        self.ridge_w = None
        self.feat_mean = None
        self.feat_std = None

        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.04,
            "subsample": 0.8,
            "colsample_bytree": 0.65,
            "min_child_weight": 500,
            "reg_lambda": 8.0,
            "reg_alpha": 1.0,
            "seed": 42,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 45
        self.ridge_alpha = 8000.0
        self.smooth_span = 300
        self.tilt_span = 1800

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _cs_robust_z(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = df.sub(med, axis=0).abs().median(axis=1)
        mad = mad.replace(0.0, 1.0).fillna(1.0)
        z = df.sub(med, axis=0).div(mad * 1.4826, axis=0)
        return z.clip(-4.0, 4.0)

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g1 = features["Feature.1"]
        g2 = features["Feature.2"]
        g3 = features["Feature.3"]
        g4 = features["Feature.4"]
        g5 = features["Feature.5"]
        g6 = features["Feature.6"]

        r1 = self._csrank(g1)
        r2 = self._csrank(g2)
        r3 = self._csrank(g3)
        r4 = self._csrank(g4)
        r5 = self._csrank(g5)
        r6 = self._csrank(g6)

        s24 = g2 - g4
        s12 = g1 - g2
        s13 = g1 - g3

        diff5_s24 = s24.diff(5).fillna(0.0)
        ema12_f1 = g1.ewm(span=12, min_periods=1).mean()

        return {
            "d24": self._csrank(s24),
            "d12": self._csrank(s12),
            "d13": self._csrank(s13),
            "m23_6": (r2 - r3) * r6,
            "m24_5": (r2 - r4) * r5,
            "p56": r5 * r6,
            "r6": r6,
            "r5": r5,
            "nl_14_6": self._csrank(g1 + g4) * self._csrank(g6.abs()),
            "nl_35_2": np.tanh(2.0 * r3) * (r5 - r2),
            "ratio_16": self._csrank(self._cs_robust_z(g1)) - r6,
            "ts_delta_24": self._csrank(diff5_s24),
            "ts_rev_13": -self._csrank(g1 - ema12_f1) * r3,
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

        X_train = X[ok]
        y_train = y[ok]

        # 1. Train 70% GBDT (V8-aligned regularization across 13 signals)
        self.gbdt_model = xgb.train(
            self.params,
            xgb.DMatrix(X_train, label=y_train),
            num_boost_round=self.n_rounds,
        )

        # 2. Train 30% Closed-Form L2 Ridge Regression
        self.feat_mean = np.mean(X_train, axis=0)
        self.feat_std = np.std(X_train, axis=0, ddof=1)
        self.feat_std = np.where(self.feat_std == 0.0, 1.0, self.feat_std)

        X_train_std = (X_train - self.feat_mean) / self.feat_std
        y_train_ctr = y_train - np.mean(y_train)

        n_features = X_train_std.shape[1]
        gram = (X_train_std.T @ X_train_std) + self.ridge_alpha * np.eye(n_features)
        self.ridge_w = np.linalg.solve(gram, X_train_std.T @ y_train_ctr)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.gbdt_model is not None and self.ridge_w is not None
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Out-of-sample GBDT & Ridge predictions
        raw_gbdt = self.gbdt_model.predict(xgb.DMatrix(X_clean))
        X_val_std = (X_clean - self.feat_mean) / self.feat_std
        raw_ridge = X_val_std @ self.ridge_w

        # Scale-match before 70/30 blend
        gbdt_norm = (raw_gbdt - np.mean(raw_gbdt)) / (np.std(raw_gbdt) + 1e-8)
        ridge_norm = (raw_ridge - np.mean(raw_ridge)) / (np.std(raw_ridge) + 1e-8)
        raw_blend = 0.70 * gbdt_norm + 0.30 * ridge_norm

        pred = pd.DataFrame(raw_blend.reshape(-1, len(tickers)), index=index, columns=tickers)

        # EMA Smoothing (span = 300) & Lagged Drift De-tilting (span = 1800)
        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - tilt).fillna(0.0)

        # Cross-sectional zero row-mean (preserving natural conviction magnitude)
        return pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
