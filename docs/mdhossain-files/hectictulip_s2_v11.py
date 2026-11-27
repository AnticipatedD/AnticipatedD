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
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (13, 47, 109, 181)
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.02,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 900,
            "reg_lambda": 22.0,
            "reg_alpha": 1.5,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 36
        self.ema1_span = 135
        self.ema2_span = 85
        self.slow_span = 1500
        self.trade_band = 0.30

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _csz(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        return df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0).clip(-3.5, 3.5)

    @staticmethod
    def _orthogonalize(target_df: pd.DataFrame, basis_df: pd.DataFrame) -> pd.DataFrame:
        num = (target_df * basis_df).sum(axis=1)
        den = (basis_df * basis_df).sum(axis=1).replace(0.0, np.nan)
        beta = (num / den).fillna(0.0)
        return target_df.sub(basis_df.mul(beta, axis=0), axis=0)

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}
        z = {i: self._csz(g.get(i)) for i in range(1, 7)}

        # Directional spreads & orthogonalized cross-feature contrasts
        d21 = z.get(2) - z.get(1)
        d24 = z.get(2) - z.get(4)
        d31 = z.get(3) - z.get(1)

        s21 = self._csrank(d21)
        s24 = self._csrank(self._orthogonalize(d24, d21))
        s31 = self._csrank(self._orthogonalize(d31, d21))

        # Contemporaneous non-linear & rank-interaction terms (zero rolling burn-in)
        i21_6 = self._csrank((r.get(2) - r.get(1)) * r.get(6))
        i24_5 = self._csrank((r.get(2) - r.get(4)) * r.get(5))
        quad_56 = self._csrank(np.tanh(z.get(5) * z.get(6)))

        # True raw cross-sectional dispersion (fixes constant-1.0 z-std bug)
        raw_disp = (
            g.get(1).std(axis=1) + g.get(2).std(axis=1) + g.get(4).std(axis=1)
        ) / 3.0
        disp_norm = (
            raw_disp / raw_disp.ewm(span=240, min_periods=1).mean().replace(0.0, np.nan)
        ).fillna(1.0).clip(0.5, 2.0)

        disp_s21 = self._csrank(d21.mul(disp_norm, axis=0))
        disp_s24 = self._csrank(d24.mul(disp_norm, axis=0))

        return {
            "s21": s21,
            "s24": s24,
            "s31": s31,
            "i21_6": i21_6,
            "i24_5": i24_5,
            "quad_56": quad_56,
            "disp_s21": disp_s21,
            "disp_s24": disp_s24,
            "r6": r.get(6),
            "r5": r.get(5),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b.get("s21").columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b.get("s21").index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y_df = target[tickers]
        y_cs = y_df.sub(y_df.mean(axis=1), axis=0)
        y = y_cs.clip(-0.08, 0.08).values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)

        dtrain = xgb.DMatrix(X[ok], label=y[ok])
        self.models = []
        for s in self.seeds:
            p = dict(self.params, seed=s)
            self.models.append(
                xgb.train(p, dtrain, num_boost_round=self.n_rounds)
            )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.models) > 0
        X, tickers, index = self._design(features)
        dtest = xgb.DMatrix(np.nan_to_num(X, nan=0.0))

        raw = np.mean([m.predict(dtest) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # 2nd-order causal low-pass filter + baseline removal (halved phase delay)
        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        # Dispersion-gated unit-variance scaling + conviction sharpening
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        disp_gate = (
            scale / scale.ewm(span=240, min_periods=1).mean().replace(0.0, np.nan)
        ).clip(0.75, 1.35).fillna(1.0)

        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = (np.sign(z) * (z.abs() ** 1.40)).mul(disp_gate, axis=0)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Neutrality-preserving hysteresis band (zero phantom trades on locked tickers)
        arr = target_pos.values
        n_rows, n_tickers = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 2:
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
