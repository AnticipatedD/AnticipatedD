# /// script
# dependencies = [
#   "xgboost>=3.2.0",
#   "numpy",
#   "pandas",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    """Contemporaneous cross-sectional predictor engineered from METRICS.md and
    COMPETITION.md specifications:
    1. Strictly contemporaneous features (passes N=100 label permutation gate).
    2. Orthogonalized off the crowded (F2-F4, F6) subspace to target a >60 deg
       independent Quality Set city without erasing (F1, F3, F5) alpha.
    3. Burn-in damped, L1-normalized hysteresis execution directly minimizing
       the 5 bp ||P(i-1) - P(i-2)||_1 transaction cost term.
    """

    def __init__(self):
        self.models = []
        self.seeds = (11, 29, 53)
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.015,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.72,
            "min_child_weight": 948,
            "reg_lambda": 22.0,
            "reg_alpha": 2.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 28
        self.ema_span = 165
        self.trade_band = 0.014

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

    def _orth_clean(self, df: pd.DataFrame, b1: pd.DataFrame, b2: pd.DataFrame) -> pd.DataFrame:
        res = self._orthogonalize(df, b1)
        res = self._orthogonalize(res, b2)
        return self._csrank(res)

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 100% contemporaneous features only (zero rolling windows -> passes permutation gate)
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        # Orthogonalize off the primary (F2 - F4) and F6 directions so this signal
        # explores a new >60 deg Quality Set city without colliding with existing entries
        basis_24 = g - g
        basis_6 = g

        s13 = self._orth_clean(g - g, basis_24, basis_6)
        s15 = self._orth_clean(g - g, basis_24, basis_6)
        s35 = self._orth_clean(g - g, basis_24, basis_6)
        s45 = self._orth_clean(g - g, basis_24, basis_6)

        i13_5 = self._orth_clean((r - r) * r, basis_24, basis_6)
        i15_3 = self._orth_clean((r - r) * r, basis_24, basis_6)
        i34_5 = self._orth_clean((r - r) * r, basis_24, basis_6)
        p15 = self._orth_clean(r * r, basis_24, basis_6)
        p35 = self._orth_clean(r * r, basis_24, basis_6)

        return {
            "s13": s13,
            "s15": s15,
            "s35": s35,
            "s45": s45,
            "i13_5": i13_5,
            "i15_3": i15_3,
            "i34_5": i34_5,
            "p15": p15,
            "p35": p35,
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["s13"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["s13"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)

        dtrain = xgb.DMatrix(X[ok], label=y[ok])
        self.models = []
        for s in self.seeds:
            p = dict(self.params, seed=s)
            self.models.append(
                xgb.train(p, dtrain, num_boost_round=self.n_rounds)
            )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.models) > 2
        X, tickers, index = self._design(features)
        dtest = xgb.DMatrix(np.nan_to_num(X, nan=0.0))

        raw = np.mean([m.predict(dtest) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Single-stage causal EWM (82-bar lag vs 149-bar cascade) + causal expanding tilt removal
        smoothed = pred.ewm(span=self.ema_span, min_periods=1).mean()
        expanding_tilt = smoothed.shift(1).expanding(min_periods=1).mean()
        signal = (smoothed - expanding_tilt).fillna(0.0)

        # Cross-sectional centering, conviction sharpening, and L1 gross normalization
        # so the 5 bp ||P(i-1) - P(i-2)||_1 turnover penalty is strictly scale-controlled
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        shaped = np.sign(z) * (z.abs() ** 1.42)
        shaped = shaped.sub(shaped.mean(axis=1), axis=0).fillna(0.0)

        l1_norm = shaped.abs().sum(axis=1).replace(0.0, np.nan)
        target_pos = shaped.div(l1_norm, axis=0).fillna(0.0)

        # Smooth burn-in ramp prevents early EWM noise from incurring full turnover fees
        n_rows = len(target_pos)
        if n_rows > 0:
            ramp = 1.0 - np.exp(-np.arange(n_rows, dtype=np.float64) / 240.0)
            target_pos = target_pos.mul(ramp, axis=0)

        # L1-calibrated causal hysteresis no-trade band
        arr = target_pos.values
        held = np.zeros_like(arr)
        if n_rows > 2:
            curr = arr.copy()
            held = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
