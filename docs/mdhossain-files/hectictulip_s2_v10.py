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
        self.ridge_w = None
        self.seeds = [31, 89, 173, 257]

        self.lgb_params = {
            "objective": "huber",
            "alpha": 0.03,
            "boosting_type": "gbdt",
            "num_leaves": 6,
            "max_depth": 3,
            "learning_rate": 0.025,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.70,
            "min_child_samples": 850,
            "reg_alpha": 2.5,
            "reg_lambda": 30.0,
            "n_estimators": 55,
            "n_jobs": 4,
            "verbose": -1,
        }

        self.xgb_params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.015,
            "max_depth": 3,
            "eta": 0.025,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 850,
            "reg_lambda": 20.0,
            "reg_alpha": 1.5,
            "verbosity": 0,
            "nthread": 4,
        }
        self.xgb_rounds = 35

        self.pre_span = 8
        self.ema1_span = 155
        self.ema2_span = 105
        self.slow_span = 1800
        self.trade_band = 0.32

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.75 * z)

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _orthogonalize(
        target_df: pd.DataFrame, basis_df: pd.DataFrame, strength: float = 1.0
    ) -> pd.DataFrame:
        num = (target_df * basis_df).sum(axis=1)
        den = (basis_df * basis_df).sum(axis=1).replace(0.0, np.nan)
        beta = (strength * (num / den)).fillna(0.0)
        return target_df.sub(basis_df.mul(beta, axis=0), axis=0)

    def _design(self, features: pd.DataFrame):
        g_fast = {
            i: features[f"Feature.{i}"].ewm(span=self.pre_span, min_periods=1).mean()
            for i in range(1, 7)
        }
        g_med = {
            i: features[f"Feature.{i}"].ewm(span=48, min_periods=1).mean()
            for i in range(1, 7)
        }

        t_feat = {i: self._robust_tanh(g_fast[i]) for i in range(1, 7)}
        t_med = {i: self._robust_tanh(g_med[i]) for i in range(1, 7)}
        r = {i: self._csrank(g_fast[i]) for i in range(1, 7)}

        energy = (
            t_feat[1].abs() + t_feat[2].abs() + t_feat[3].abs() + t_feat[4].abs()
        ) / 4.0
        spread_14 = t_feat[1] - t_feat[4]
        spread_23 = t_feat[2] - t_feat[3]

        ref_axis = t_feat[5] - t_feat[3]
        rot_14 = self._orthogonalize(spread_14, ref_axis, strength=0.65)

        rat_14 = self._csrank(rot_14 / (0.45 + energy))
        orth_23 = self._csrank(self._orthogonalize(spread_23, spread_14, strength=1.0))

        acc_14 = self._csrank(rot_14 - (t_med[1] - t_med[4]))
        vel_6 = self._csrank(r[6] - r[6].ewm(span=100, min_periods=1).mean())
        div_65 = self._csrank(t_feat[6] - t_feat[5])

        geom_146 = np.sign(rot_14) * np.sqrt(rot_14.abs() * r[6].abs())
        geom_235 = np.sign(spread_23) * np.sqrt(spread_23.abs() * r[5].abs())
        harm_56 = self._csrank(
            (2.0 * t_feat[5] * t_feat[6])
            / (t_feat[5].abs() + t_feat[6].abs() + 0.35)
        )

        cs_disp = rot_14.std(axis=1).replace(0.0, np.nan)
        disp_norm = (cs_disp / cs_disp.ewm(span=250, min_periods=1).mean()).fillna(1.0)
        regime_14 = rat_14.mul(np.sqrt(disp_norm.clip(0.6, 1.7)), axis=0)

        f6_rot = self._csrank(self._orthogonalize(t_feat[6], ref_axis, strength=0.65))

        blocks = [
            rat_14,
            orth_23,
            acc_14,
            vel_6,
            div_65,
            geom_146,
            geom_235,
            harm_56,
            regime_14,
            f6_rot,
        ]

        tickers = list(rat_14.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, rat_14.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)

        y_df = target[tickers]
        y_cs = y_df.sub(y_df.mean(axis=1), axis=0)
        y_std = y_cs.std(axis=1).replace(0.0, np.nan)
        y_norm = y_cs.div(y_std, axis=0).clip(-4.0, 4.0)

        y = y_norm.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        n_feat = X_ok.shape[1]
        xtx = X_ok.T @ X_ok + 5000.0 * np.eye(n_feat)
        xty = X_ok.T @ y_ok
        self.ridge_w = np.linalg.solve(xtx, xty)

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
        raw_xgb = np.mean(
            [m.predict(xgb.DMatrix(X_clean)) for m in self.xgb_models], axis=0
        )
        raw_ridge = X_clean @ self.ridge_w

        raw = 0.55 * raw_lgb + 0.25 * raw_xgb + 0.20 * raw_ridge
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.40)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        n_rows, n_tickers = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            curr = arr[0].copy()
            curr -= curr.mean()
            held[0] = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                active = np.abs(diff) > band
                if np.any(active):
                    move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                    curr = curr + move
                    n_act = active.sum()
                    curr[active] -= curr.mean() * (n_tickers / n_act)
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
