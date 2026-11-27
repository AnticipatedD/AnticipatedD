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
        self.ridge_model = Ridge(alpha=1600.0, fit_intercept=False)
        self.seeds = [31, 79, 97, 193]

        self.et_params = {
            "n_estimators": 77,
            "max_depth": 2,
            "min_samples_leaf": 909,
            "max_features": 0.40,
            "bootstrap": True,
            "max_samples": 0.70,
            "n_jobs": 4,
        }

        self.huber_params = {
            "alpha": 240.0,
            "epsilon": 1.35,
            "max_iter": 250,
            "fit_intercept": False,
        }

        self.ema1_span = 205
        self.ema2_span = 140
        self.slow_span = 1800
        self.tail_power = 1.35
        self.trade_band = 0.22

    @staticmethod
    def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return ((r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))).fillna(0.0)

    @staticmethod
    def _robust_tanh(df: pd.DataFrame, scale: float = 0.67) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(scale * z)

    @staticmethod
    def _orthogonalize(target_df: pd.DataFrame, basis_df: pd.DataFrame) -> pd.DataFrame:
        num = (target_df * basis_df).sum(axis=1)
        den = (basis_df * basis_df).sum(axis=1).replace(0.0, np.nan)
        beta = (num / den).fillna(0.0)
        return target_df.sub(basis_df.mul(beta, axis=0), axis=0)

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        t = {i: self._robust_tanh(g.get(i)) for i in range(1, 7)}
        r = {i: self._cs_rank(g.get(i)) for i in range(1, 7)}

        ema_60 = {i: self._cs_rank(t.get(i).ewm(span=60, min_periods=1).mean()) for i in range(1, 7)}
        ema_140 = {i: self._cs_rank(t.get(i).ewm(span=140, min_periods=1).mean()) for i in range(1, 7)}
        drift = {i: self._cs_rank(ema_60.get(i) - ema_140.get(i)) for i in range(1, 7)}

        basis_a = t.get(2) - t.get(4)
        basis_b = t.get(3) - t.get(1)

        orth_45 = self._orthogonalize(t.get(4) - t.get(5), basis_a)
        orth_45 = self._cs_rank(self._orthogonalize(orth_45, basis_b))

        orth_16 = self._orthogonalize(t.get(1) + t.get(6), basis_a)
        orth_16 = self._cs_rank(self._orthogonalize(orth_16, basis_b))

        orth_56 = self._orthogonalize(t.get(6) - t.get(5), basis_a)
        orth_56 = self._cs_rank(self._orthogonalize(orth_56, basis_b))

        disp = (t.get(1).abs() + t.get(4).abs() + t.get(5).abs() + t.get(6).abs()) / 4.0
        snr_45 = self._cs_rank((ema_140.get(4) - ema_140.get(5)) / (0.45 + disp))
        snr_16 = self._cs_rank((ema_140.get(1) - ema_140.get(6)) / (0.45 + disp))

        gate_6 = np.tanh(1.5 * ema_140.get(6))
        mod_45 = self._cs_rank((t.get(4) - t.get(5)) * gate_6)
        mod_13 = self._cs_rank((drift.get(1) - drift.get(3)) * r.get(6))

        blocks = [
            orth_45,
            orth_16,
            orth_56,
            snr_45,
            snr_16,
            mod_45,
            mod_13,
            drift.get(4) - drift.get(6),
            drift.get(1) - drift.get(5),
            self._cs_rank(ema_140.get(4) * ema_140.get(6)),
            self._cs_rank(ema_140.get(5) * ema_140.get(6)),
            self._cs_rank(ema_140.get(6)),
        ]

        tickers = list(orth_45.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, orth_45.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        y_mat = aligned_target.sub(aligned_target.mean(axis=1), axis=0)
        y_clip = y_mat.clip(-0.067, 0.067)

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

        raw = 0.75 * raw_et + 0.20 * raw_hb + 0.25 * raw_rd
        pred = pd.DataFrame(raw.reshape(n_rows, n_tickers), index=index, columns=tickers)

        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** self.tail_power)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        held = np.zeros_like(arr)
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
