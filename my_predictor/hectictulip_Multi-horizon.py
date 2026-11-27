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
from sklearn.linear_model import HuberRegressor, Ridge
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.et_models = []
        self.huber_model = None
        self.ridge_model = Ridge(alpha=1500.0, fit_intercept=False)
        self.seeds = [41, 89, 173]

        self.et_params = {
            "n_estimators": 75,
            "max_depth": 3,
            "min_samples_leaf": 900,
            "max_features": 0.50,
            "bootstrap": True,
            "max_samples": 0.70,
            "n_jobs": 4,
        }

        self.huber_params = {
            "alpha": 220.0,
            "epsilon": 1.35,
            "max_iter": 250,
            "fit_intercept": False,
        }

        # Parallel fast + slow horizons (eliminates sequential EMA lag to boost Sharpe)
        self.fast_span = 30
        self.slow_span = 150
        self.base_span = 1600
        self.fast_weight = 0.65
        self.deadzone = 0.22

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
        z = {i: self._cs_zscore(g.get(i)) for i in range(1, 7)}
        r = {i: self._cs_rank(g.get(i)) for i in range(1, 7)}

        # Multi-scale temporal drift & cross-sectional rank velocity
        ema_short = {i: z.get(i).ewm(span=20, min_periods=1).mean() for i in range(1, 7)}
        ema_long = {i: z.get(i).ewm(span=110, min_periods=1).mean() for i in range(1, 7)}
        vel_ema = {i: self._cs_rank(ema_short.get(i) - ema_long.get(i)) for i in range(1, 7)}
        vel_rk5 = {i: self._cs_rank(r.get(i) - r.get(i).shift(5).fillna(r.get(i))) for i in range(1, 7)}

        # Project out crowded static bases to preserve ~18+ City Novelty
        crowded_a = z.get(2) - z.get(4)
        crowded_b = z.get(3) - z.get(1)

        orth_45 = self._orthogonalize(z.get(4) - z.get(5), crowded_a)
        orth_45 = self._cs_rank(self._orthogonalize(orth_45, crowded_b))

        orth_16 = self._orthogonalize(z.get(1) + z.get(6), crowded_a)
        orth_16 = self._cs_rank(self._orthogonalize(orth_16, crowded_b))

        orth_56 = self._orthogonalize(z.get(6) - z.get(5), crowded_a)
        orth_56 = self._cs_rank(self._orthogonalize(orth_56, crowded_b))

        # Dispersion-normalized SNR & regime-gated products
        disp = (z.get(1).abs() + z.get(4).abs() + z.get(5).abs() + z.get(6).abs()) / 4.0
        snr_45 = self._cs_rank((ema_long.get(4) - ema_long.get(5)) / (0.4 + disp))
        snr_16 = self._cs_rank((ema_long.get(1) - ema_long.get(6)) / (0.4 + disp))

        gate_6 = np.tanh(1.5 * ema_long.get(6))
        mod_45 = self._cs_rank((z.get(4) - z.get(5)) * gate_6)
        mod_13 = self._cs_rank((vel_ema.get(1) - vel_ema.get(3)) * r.get(6))

        blocks = [
            orth_45,
            orth_16,
            orth_56,
            snr_45,
            snr_16,
            mod_45,
            mod_13,
            vel_ema.get(4) - vel_ema.get(6),
            vel_ema.get(1) - vel_ema.get(5),
            vel_rk5.get(2) - vel_rk5.get(4),
            vel_rk5.get(3) - vel_rk5.get(1),
            self._cs_rank(ema_long.get(4) * ema_long.get(6)),
            self._cs_rank(ema_long.get(5) * ema_long.get(6)),
            self._cs_rank(ema_long.get(6)),
        ]

        tickers = list(orth_45.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, orth_45.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        y_mat = aligned_target.sub(aligned_target.mean(axis=1), axis=0)
        y_clip = y_mat.clip(-0.075, 0.075)

        y = y_clip.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.et_models = []
        self.huber_model = None
        self.is_fitted = False
        if len(y_ok) < 500:
            return

        for s in self.seeds:
            et = ExtraTreesRegressor(**dict(self.et_params, random_state=s))
            et.fit(X_ok, y_ok)
            self.et_models.append(et)

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

        raw_et = np.mean([m.predict(X_clean) for m in self.et_models], axis=0)
        raw_hb = self.huber_model.predict(X_clean)
        raw_rd = self.ridge_model.predict(X_clean)

        raw = 0.65 * raw_et + 0.20 * raw_hb + 0.15 * raw_rd
        pred = pd.DataFrame(raw.reshape(n_rows, n_tickers), index=index, columns=tickers)

        # Parallel fast + slow horizons to cut phase lag and boost Sharpe
        fast_sig = pred.ewm(span=self.fast_span, min_periods=1).mean()
        slow_sig = pred.ewm(span=self.slow_span, min_periods=1).mean()
        baseline = slow_sig.shift(1).ewm(span=self.base_span, min_periods=1).mean()

        signal = (
            self.fast_weight * fast_sig + (1.0 - self.fast_weight) * (slow_sig - baseline)
        ).fillna(0.0)

        # Cross-sectional standardization with sharper conviction tails
        cs = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = cs.std(axis=1).replace(0.0, np.nan)
        z = cs.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.48)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

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
