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
from sklearn.linear_model import Ridge
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.et_models = []
        self.ridge_model = Ridge(alpha=1200.0, fit_intercept=False)
        self.seeds = (17, 53, 97, 149)

        # Shallow, high-leaf-floor randomized trees prevent noise memorization
        self.et_params = {
            "n_estimators": 80,
            "max_depth": 3,
            "min_samples_leaf": 950,
            "max_features": 0.45,
            "bootstrap": True,
            "max_samples": 0.70,
            "n_jobs": 4,
        }

        # Multi-horizon blend: fast tactical + slow structural (fixes phase-lag Sharpe drag)
        self.fast_span = 35
        self.slow_span = 165
        self.trend_base_span = 1500
        self.fast_weight = 0.35
        self.trade_band = 0.24

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

        # 1. Multi-lag cross-sectional rank velocity (captures persistent alpha flow)
        vel_5 = {i: self._cs_rank(r.get(i) - r.get(i).shift(5).fillna(r.get(i))) for i in range(1, 7)}
        vel_20 = {i: self._cs_rank(r.get(i) - r.get(i).shift(20).fillna(r.get(i))) for i in range(1, 7)}
        ema_60 = {i: self._cs_rank(t.get(i).ewm(span=60, min_periods=1).mean()) for i in range(1, 7)}

        # 2. Orthogonalize against Q-19 primary directions to boost City Novelty (>4.0)
        basis_24 = t.get(2) - t.get(4)
        basis_31 = t.get(3) - t.get(1)

        orth_45 = self._orthogonalize(t.get(4) - t.get(5), basis_24)
        orth_45 = self._cs_rank(self._orthogonalize(orth_45, basis_31))

        orth_16 = self._orthogonalize(t.get(1) - t.get(6), basis_24)
        orth_16 = self._cs_rank(self._orthogonalize(orth_16, basis_31))

        # 3. Pairwise conditional regime interactions across all 6 features
        disp = (t.get(1).abs() + t.get(3).abs() + t.get(5).abs() + t.get(6).abs()) / 4.0
        snr_46 = self._cs_rank((ema_60.get(4) - ema_60.get(6)) / (0.45 + disp))
        snr_25 = self._cs_rank((ema_60.get(2) - ema_60.get(5)) / (0.45 + disp))

        blocks = [
            orth_45,
            orth_16,
            snr_46,
            snr_25,
            vel_5.get(2) - vel_5.get(4),
            vel_5.get(3) - vel_5.get(1),
            vel_20.get(4) - vel_20.get(6),
            vel_20.get(1) - vel_20.get(5),
            ema_60.get(6) - ema_60.get(5),
            self._cs_rank(t.get(4) * t.get(6)),
            self._cs_rank(t.get(2) * t.get(5)),
            self._cs_rank(t.get(1) * t.get(3)),
            ema_60.get(6),
            ema_60.get(4),
        ]

        tickers = list(orth_45.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, orth_45.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        # Cross-sectionally demean and winsorize outliers
        y_cs = aligned_target.sub(aligned_target.mean(axis=1), axis=0)
        y_clip = y_cs.clip(-0.075, 0.075)

        y = y_clip.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.et_models = []
        self.is_fitted = False
        if len(y_ok) < 500:
            return

        for s in self.seeds:
            et = ExtraTreesRegressor(**dict(self.et_params, random_state=s))
            et.fit(X_ok, y_ok)
            self.et_models.append(et)

        self.ridge_model.fit(X_ok, y_ok)
        self.is_fitted = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._design(features)
        n_rows, n_tickers = len(index), len(tickers)

        if not getattr(self, "is_fitted", False) or n_rows == 0 or n_tickers == 0:
            return pd.DataFrame(0.0, index=index, columns=tickers, dtype=np.float64)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw_et = np.mean([m.predict(X_clean) for m in self.et_models], axis=0)
        raw_rd = self.ridge_model.predict(X_clean)

        raw = 0.72 * raw_et + 0.28 * raw_rd
        pred = pd.DataFrame(raw.reshape(n_rows, n_tickers), index=index, columns=tickers)

        # Multi-horizon blend: parallel fast + slow EMAs eliminate sequential phase lag
        fast_sig = pred.ewm(span=self.fast_span, min_periods=1).mean()
        slow_sig = pred.ewm(span=self.slow_span, min_periods=1).mean()
        baseline = slow_sig.shift(1).ewm(span=self.trend_base_span, min_periods=1).mean()

        blended = self.fast_weight * fast_sig + (1.0 - self.fast_weight) * (slow_sig - baseline)
        signal = blended.fillna(0.0)

        # Cross-sectional normalization & tail conviction boost (targets Concentration ~0.165)
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.50)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Hysteresis deadzone to control turnover without losing IC
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
