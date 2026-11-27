# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import HuberRegressor
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.et_models = []
        self.huber_model = None
        self.seeds = (17, 53, 109)

        # Extremely randomized trees with large leaf floors resist noise memorization
        self.et_params = {
            "n_estimators": 60,
            "max_depth": 3,
            "min_samples_leaf": 1000,
            "max_features": 0.55,
            "bootstrap": True,
            "max_samples": 0.70,
            "n_jobs": 4,
        }

        # Heavy L2-regularized Huber regression for outlier-proof linear baseline
        self.huber_params = {
            "alpha": 250.0,
            "epsilon": 1.35,
            "max_iter": 200,
            "fit_intercept": False,
        }

        # Horizon & turnover parameters
        self.fast_span = 160
        self.mid_span = 110
        self.base_span = 1650
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

        # 1. Temporal drift & acceleration features (captures persistent flow vs static snapshots)
        ema_short = {i: z[i].ewm(span=24, min_periods=1).mean() for i in range(1, 7)}
        ema_long = {i: z[i].ewm(span=120, min_periods=1).mean() for i in range(1, 7)}
        vel = {i: self._cs_rank(ema_short[i] - ema_long[i]) for i in range(1, 7)}

        # 2. Orthogonalize away from crowded static spreads to target a new Signal City (>60 deg)
        crowded_basis_a = z[2] - z[4]
        crowded_basis_b = z[3] - z[1]

        orth_45 = self._orthogonalize(z[4] - z[5], crowded_basis_a)
        orth_45 = self._cs_rank(self._orthogonalize(orth_45, crowded_basis_b))

        orth_16 = self._orthogonalize(z[1] + z[6], crowded_basis_a)
        orth_16 = self._cs_rank(self._orthogonalize(orth_16, crowded_basis_b))

        # 3. Cross-sectional dispersion & regime-conditioned interactions
        disp_145 = (z[1].abs() + z[4].abs() + z[5].abs()) / 3.0
        snr_45 = self._cs_rank((ema_long[4] - ema_long[5]) / (0.4 + disp_145))
        snr_16 = self._cs_rank((ema_long[1] - ema_long[6]) / (0.4 + disp_145))

        # 4. Asymmetric tail-gated products (focuses on Feature 4, 5, 6 regime shifts)
        gate_6 = np.tanh(1.5 * ema_long[6])
        mod_45 = self._cs_rank((z[4] - z[5]) * gate_6)
        mod_13 = self._cs_rank((vel[1] - vel[3]) * r[5])

        blocks = [
            orth_45,
            orth_16,
            snr_45,
            snr_16,
            mod_45,
            mod_13,
            vel[4] - vel[2],
            vel[5] - vel[6],
            self._cs_rank(ema_long[4] * ema_long[5]),
            self._cs_rank(ema_long[6]),
        ]

        tickers = list(orth_45.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, orth_45.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        # Winsorize target tails so extreme jumps do not distort tree splits or linear slopes
        y_mat = aligned_target.sub(aligned_target.mean(axis=1), axis=0)
        y_clip = y_mat.clip(-0.08, 0.08)

        y = y_clip.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.et_models = []
        self.huber_model = None
        if len(y_ok) < 500:
            return

        for s in self.seeds:
            et = ExtraTreesRegressor(**dict(self.et_params, random_state=s))
            et.fit(X_ok, y_ok)
            self.et_models.append(et)

        self.huber_model = HuberRegressor(**self.huber_params)
        self.huber_model.fit(X_ok, y_ok)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._design(features)
        n_rows, n_tickers = len(index), len(tickers)

        if not self.et_models or self.huber_model is None or n_rows == 0 or n_tickers == 0:
            return pd.DataFrame(0.0, index=index, columns=tickers, dtype=np.float64)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw_et = np.mean([m.predict(X_clean) for m in self.et_models], axis=0)
        raw_hb = self.huber_model.predict(X_clean)

        # Blend randomized trees (70%) with regularized Huber linear anchor (30%)
        raw = 0.70 * raw_et + 0.30 * raw_hb
        pred = pd.DataFrame(raw.reshape(n_rows, n_tickers), index=index, columns=tickers)

        # Two-stage low-pass + slow baseline removal to preserve IC and cut IC dispersion
        lp1 = pred.ewm(span=self.fast_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.mid_span, min_periods=1).mean()
        baseline = lp2.shift(1).ewm(span=self.base_span, min_periods=1).mean()
        signal = (lp2 - baseline).fillna(0.0)

        # Cross-sectional standardization with soft conviction shaping
        cs = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = cs.std(axis=1).replace(0.0, np.nan)
        z = cs.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.25)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Hysteresis deadzone to suppress unnecessary turnover
        arr = target_pos.values
        held = np.zeros_like(arr)
        curr = arr[0].copy()
        held[0] = curr
        band = self.deadzone
        for t in range(1, n_rows):
            diff = arr[t] - curr
            move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
            curr = curr + move
            held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
