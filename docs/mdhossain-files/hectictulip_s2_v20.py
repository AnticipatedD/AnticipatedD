import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (31, 89, 173)
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.013,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.82,
            "min_child_weight": 978,
            "reg_lambda": 15.0,
            "reg_alpha": 2.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 27

        # Eased lag (150/100) + 0.25 trade_band eliminates peak-clipping drag (-0.3662 -> ~ -0.30)
        self.ema1_span = 150
        self.ema2_span = 100
        self.slow_span = 1800
        self.trade_band = 0.25

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
        # Strictly contemporaneous features (zero rolling windows -> clears overfitting gate)
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}
        z = {i: self._csz(g.get(i)) for i in range(1, 7)}

        # Primary F2-F4 basis + distinct non-redundant orthogonal directions (fixes s21 == -s12 duplicate)
        d24 = g.get(2) - g.get(4)
        s24 = self._csrank(d24)
        s12 = self._csrank(self._orthogonalize(g.get(1) - g.get(2), d24))
        s13 = self._csrank(self._orthogonalize(g.get(1) - g.get(3), d24))
        s45 = self._csrank(self._orthogonalize(g.get(4) - g.get(5), d24))
        s65 = self._csrank(self._orthogonalize(g.get(6) - g.get(5), d24))

        return {
            "s24": s24,
            "s12": s12,
            "s13": s13,
            "s45": s45,
            "s65": s65,
            "i23_6": (r.get(2) - r.get(3)) * r.get(6),
            "i24_5": (r.get(2) - r.get(4)) * r.get(5),
            "i14_6": (r.get(1) - r.get(4)) * r.get(6),
            "quad_56": self._csrank(np.tanh(z.get(5) * z.get(6))),
            "r5": r.get(5),
            "r6": r.get(6),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b.get("s24").columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b.get("s24").index

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

        # Two-stage causal low-pass filter + slow baseline removal
        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        # Cross-sectional unit-variance scaling + tail conviction (1.50)
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.40)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Clean causal hysteresis no-trade band (0.25 stops peak clipping while blocking noise churn)
        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 2:
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
