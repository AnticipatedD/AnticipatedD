import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (23, 67, 149)
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.015,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 950,
            "reg_lambda": 24.0,
            "reg_alpha": 2.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 28
        self.ema1_span = 520
        self.ema2_span = 340
        self.slow_span = 2800
        self.trade_band = 0.32

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _csz(df: pd.DataFrame) -> pd.DataFrame:
        m = df.mean(axis=1)
        s = df.std(axis=1).replace(0.0, np.nan)
        return df.sub(m, axis=0).div(s, axis=0).fillna(0.0).clip(-3.5, 3.5)

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

        # Orthogonal subspace anchored on Feature.3, Feature.4, and Feature.5
        d34 = g.get(3) - g.get(4)
        s34 = self._csrank(d34)
        s41 = self._csrank(self._orthogonalize(g.get(4) - g.get(1), d34))
        s32 = self._csrank(self._orthogonalize(g.get(3) - g.get(2), d34))

        # True raw cross-sectional dispersion on F3/F4/F5
        raw_disp = (g.get(3).std(axis=1) + g.get(4).std(axis=1) + g.get(5).std(axis=1)) / 3.0
        disp_ratio = (
            raw_disp / raw_disp.ewm(span=300, min_periods=1).mean().replace(0.0, np.nan)
        ).fillna(1.0).clip(0.5, 2.0)

        return {
            "s34": s34,
            "s41": s41,
            "s32": s32,
            "i34_5": (r.get(3) - r.get(4)) * r.get(5),
            "i41_6": (r.get(4) - r.get(1)) * r.get(6),
            "quad_35": np.tanh(z.get(3) * z.get(5)),
            "quad_45": np.tanh(z.get(4) * z.get(5)),
            "disp_s34": s34.mul(disp_ratio, axis=0),
            "r4": r.get(4),
            "r5": r.get(5),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["s34"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["s34"].index

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
        assert len(self.models) > 0
        X, tickers, index = self._design(features)
        dtest = xgb.DMatrix(np.nan_to_num(X, nan=0.0))

        raw = np.mean([m.predict(dtest) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.35)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

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
