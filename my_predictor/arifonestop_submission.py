import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRa_S2_V27Predictor(Predictor):
    def __init__(self):
        self.xgb_models = []
        self.ridge_w = None
        self.seeds = (11, 29, 53)
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.015,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 950,
            "reg_lambda": 20.0,
            "reg_alpha": 2.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 26
        self.ridge_alpha = 6000.0
        self.fast_span = 180
        self.slow_span = 2160

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
        # Light pre-smoothing eliminates single-bar tree threshold chatter
        g = {
            i: features[f"Feature.{i}"].ewm(span=8, min_periods=1).mean()
            for i in range(1, 7)
        }
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}

        s24 = self._csrank(g.get(2) - g.get(4))
        s12 = self._csrank(self._orthogonalize(g.get(1) - g.get(2), g.get(2) - g.get(4)))
        s13 = self._csrank(self._orthogonalize(g.get(1) - g.get(3), g.get(2) - g.get(4)))

        return {
            "s24": s24,
            "s12": s12,
            "s13": s13,
            "i23_6": (r.get(2) - r.get(3)) * r.get(6),
            "i24_5": (r.get(2) - r.get(4)) * r.get(5),
            "i14_6": (r.get(1) - r.get(4)) * r.get(6),
            "p56": r.get(5) * r.get(6),
            "r5": r.get(5),
            "r6": r.get(6),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["s24"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["s24"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok = X[ok]
        y_ok = y[ok]

        # Smooth L2 Ridge component eliminates step-function tree turnover
        XtX = X_ok.T @ X_ok + self.ridge_alpha * np.eye(X_ok.shape[1])
        self.ridge_w = np.linalg.solve(XtX, X_ok.T @ y_ok)

        # Robust Pseudo-Huber GBDT ensemble
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

        raw_ridge = X_clean @ self.ridge_w
        dtest = xgb.DMatrix(X_clean)
        raw_xgb = np.mean([m.predict(dtest) for m in self.xgb_models], axis=0)

        raw_ridge = raw_ridge / (np.std(raw_ridge) + 1e-12)
        raw_xgb = raw_xgb / (np.std(raw_xgb) + 1e-12)
        raw = 0.45 * raw_ridge + 0.55 * raw_xgb

        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Retain 65% of the persistent trend instead of subtracting 100% away
        fast = pred.ewm(span=self.fast_span, min_periods=1).mean()
        slow = fast.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (fast - 0.35 * slow).fillna(0.0)

        # Cross-sectional volatility stabilization (reduces portfolio return variance)
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        cs_std = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(cs_std, axis=0).fillna(0.0).clip(-2.5, 2.5)

        # Soft deadzone zeroes out noisy weak weights to slash turnover drag
        z_active = np.sign(z) * np.maximum(z.abs() - 0.40, 0.0)
        out = np.sign(z_active) * (z_active.abs() ** 1.20)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
