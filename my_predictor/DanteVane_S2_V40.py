# /// script
# dependencies = [
#   "lightgbm>=4.0",
#   "xgboost>=2.0",
#   "numpy",
#   "pandas",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.lgb_models = []
        self.xgb_models = []
        self.seeds = [31, 89, 173, 251]

        self.lgb_params = {
            "objective": "huber",
            "alpha": 0.035,
            "boosting_type": "gbdt",
            "num_leaves": 5,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.70,
            "min_child_samples": 950,
            "reg_alpha": 2.0,
            "reg_lambda": 25.0,
            "n_estimators": 45,
            "n_jobs": 4,
            "verbose": -1,
        }

        self.xgb_params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.018,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.72,
            "min_child_weight": 900,
            "reg_lambda": 15.0,
            "reg_alpha": 1.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.xgb_rounds = 25

        # Slightly deeper smoothing + 0.37 band to cut -0.2997 compression loss while keeping IC ~0.046
        self.ema1_span = 250
        self.ema2_span = 165
        self.slow_span = 2050
        self.trade_band = 0.37

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.67 * z)

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

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        t_feat = {i: self._robust_tanh(g[i]) for i in range(1, 7)}
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        # Exact 0.0474 IC MAD-copula feature set
        energy = (t_feat[1].abs() + t_feat[3].abs() + t_feat[4].abs()) / 3.0
        rat_24 = self._csrank((t_feat[2] - t_feat[4]) / (0.5 + energy))
        rat_31 = self._csrank((t_feat[3] - t_feat[1]) / (0.5 + energy))

        orth_12 = self._csrank(self._orthogonalize(g[1] - g[2], g[2] - g[4]))
        curv_123 = self._csrank((t_feat[1] - 2.0 * t_feat[2] + t_feat[3]) / (0.5 + energy))

        geom_246 = np.sign(t_feat[2] - t_feat[4]) * np.sqrt(
            (t_feat[2] - t_feat[4]).abs() * r[6].abs()
        )
        geom_315 = np.sign(t_feat[3] - t_feat[1]) * np.sqrt(
            (t_feat[3] - t_feat[1]).abs() * r[5].abs()
        )

        blocks = [
            rat_24,
            rat_31,
            orth_12,
            curv_123,
            geom_246,
            geom_315,
            t_feat[5] * t_feat[6],
            t_feat[2] * t_feat[6],
            r[6] - r[5],
            t_feat[6],
        ]

        tickers = list(rat_24.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, rat_24.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.lgb_models = []
        self.xgb_models = []
        dtrain = xgb.DMatrix(X_ok, label=y_ok)

        for s in self.seeds:
            est = lgb.LGBMRegressor(**dict(self.lgb_params, random_state=s))
            est.fit(X_ok, y_ok)
            self.lgb_models.append(est)

            bst = xgb.train(
                dict(self.xgb_params, seed=s),
                dtrain,
                num_boost_round=self.xgb_rounds,
            )
            self.xgb_models.append(bst)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.lgb_models) > 0 and len(self.xgb_models) > 0
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw_lgb = np.mean([m.predict(X_clean) for m in self.lgb_models], axis=0)
        raw_xgb = np.mean([m.predict(xgb.DMatrix(X_clean)) for m in self.xgb_models], axis=0)

        raw = 0.80 * raw_lgb + 0.20 * raw_xgb
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-2.8, 2.8)

        # Lower exponent (1.12 vs 1.35) reduces tail concentration & fee churn
        target_pos = np.sign(z) * (z.abs() ** 1.12)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Inverse-dispersion vol targeting scales down exposure in noisy regimes
        vol_reg = (scale / scale.ewm(span=240, min_periods=1).mean().replace(0.0, np.nan)).fillna(1.0).clip(0.75, 1.35)
        target_pos = target_pos.div(vol_reg, axis=0)

        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            curr = arr[0].copy()
            held[0] = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
