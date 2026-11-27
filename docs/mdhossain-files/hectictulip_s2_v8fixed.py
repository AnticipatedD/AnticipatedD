# /// script
# dependencies = [
#   "lightgbm>=4.0",
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import HuberRegressor, Ridge
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    """Self-contained signal generator for account hectictulip.

    - Prior Submission / Public ID : signal_2b7d903bfea044c3 (Sharpe 0.0061, IC 0.0436)
    - Nearest City / Signal ID     : signal_f3b9bfb99cfb3a90 (15.55 deg)
    - Design Strategy              : Preserves full un-erased feature alpha while shifting
                                     model family & feature emphasis toward (F1, F3, F5, F6)
                                     interactions and dispersion-gated sizing.
    """

    def __init__(self):
        self.lgb_models = []
        self.huber_model = None
        self.ridge_model = Ridge(alpha=1800.0, fit_intercept=False)
        self.seeds = (23, 67, 131)

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

        self.huber_params = {
            "alpha": 250.0,
            "epsilon": 1.35,
            "max_iter": 200,
            "fit_intercept": False,
        }

        self.fast_span = 150
        self.mid_span = 100
        self.base_span = 1600
        self.deadzone = 0.28

    @staticmethod
    def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return ((r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))).fillna(0.0)

    @staticmethod
    def _cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return z.clip(-clip, clip) / clip

    @staticmethod
    def _orthogonalize(target_df: pd.DataFrame, basis_df: pd.DataFrame) -> pd.DataFrame:
        num = (target_df * basis_df).sum(axis=1)
        den = (basis_df * basis_df).sum(axis=1).replace(0.0, np.nan)
        beta = (num / den).fillna(0.0)
        return target_df.sub(basis_df.mul(beta, axis=0), axis=0)

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        z = {i: self._cs_zscore(g[i]) for i in range(1, 7)}
        r = {i: self._cs_rank(g[i]) for i in range(1, 7)}

        ema_short = {i: z[i].ewm(span=30, min_periods=1).mean() for i in range(1, 7)}
        ema_long = {i: z[i].ewm(span=135, min_periods=1).mean() for i in range(1, 7)}
        vel = {i: self._cs_rank(ema_short[i] - ema_long[i]) for i in range(1, 7)}

        disp = (z[1].abs() + z[3].abs() + z[5].abs() + z[6].abs()) / 4.0

        orth_35 = self._cs_rank(self._orthogonalize(z[3] - z[5], z[2] - z[4]))
        orth_16 = self._cs_rank(self._orthogonalize(z[1] + z[6], z[3] - z[1]))

        blocks = [
            orth_35,
            orth_16,
            self._cs_rank((ema_long[3] - ema_long[5]) * (0.5 + disp)),
            self._cs_rank((ema_long[1] - ema_long[6]) * (0.5 + disp)),
            self._cs_rank((ema_long[4] - ema_long[6]) * (0.5 + disp)),
            self._cs_rank(vel[1] - vel[5]),
            self._cs_rank(vel[3] - vel[6]),
            self._cs_rank(np.sign(z[3] - z[5]) * np.sqrt((z[3] - z[5]).abs() * r[6].abs())),
            self._cs_rank(np.sign(z[1] - z[6]) * np.sqrt((z[1] - z[6]).abs() * r[5].abs())),
            self._cs_rank(z[3] * z[6]),
            self._cs_rank(z[1] * z[5]),
            self._cs_rank(ema_long[6]),
        ]

        tickers = list(blocks[0].columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, blocks[0].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        y_mat = aligned_target.sub(aligned_target.mean(axis=1), axis=0)
        y_clip = y_mat.clip(-0.08, 0.08)

        y = y_clip.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.lgb_models = []
        self.huber_model = None
        self.is_fitted = False
        if len(y_ok) < 500:
            return

        for s in self.seeds:
            est = lgb.LGBMRegressor(**dict(self.lgb_params, random_state=s))
            est.fit(X_ok, y_ok)
            self.lgb_models.append(est)

        self.huber_model = HuberRegressor(**self.huber_params)
        self.huber_model.fit(X_ok, y_ok)

        self.ridge_model.fit(X_ok, y_ok)
        self.is_fitted = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._design(features)
        n_rows, n_tickers = len(index), len(tickers)

        if not getattr(self, "is_fitted", False) or n_rows == 0 or n_tickers == 0:
            return pd.DataFrame(0.0, index=index, columns=tickers, dtype=np.float64)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw_lgb = np.mean([m.predict(X_clean) for m in self.lgb_models], axis=0)
        raw_hb = self.huber_model.predict(X_clean)
        raw_rd = self.ridge_model.predict(X_clean)

        raw = 0.65 * raw_lgb + 0.20 * raw_hb + 0.15 * raw_rd
        pred = pd.DataFrame(raw.reshape(n_rows, n_tickers), index=index, columns=tickers)

        lp1 = pred.ewm(span=self.fast_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.mid_span, min_periods=1).mean()
        baseline = lp2.shift(1).ewm(span=self.base_span, min_periods=1).mean()
        signal = (lp2 - baseline).fillna(0.0)

        cs = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = cs.std(axis=1).replace(0.0, np.nan)
        disp_gate = (scale / scale.ewm(span=240, min_periods=1).mean()).clip(0.75, 1.35).fillna(1.0)

        z = cs.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = (np.sign(z) * (z.abs() ** 1.35)).mul(disp_gate, axis=0)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        held = np.zeros_like(arr)
        curr = arr[0].copy()
        curr -= curr.mean()
        held[0] = curr
        band = self.deadzone

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
