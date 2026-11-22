import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (37, 89, 163, 251)
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.015,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 1000,
            "reg_lambda": 24.0,
            "reg_alpha": 2.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 28
        self.ema1_span = 540
        self.ema2_span = 340
        self.slow_span = 3000
        self.trade_band = 0.40

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.70 * z)

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
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        t_feat = {i: self._robust_tanh(g.get(i)) for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}

        # Orthogonal subspace anchored on Feature.1, Feature.5, and Feature.6
        # (avoids [★ Q15]'s primary rat_24 and rat_31 axes)
        base_15 = t_feat.get(1) - t_feat.get(5)
        s15 = self._csrank(base_15)
        s62 = self._csrank(self._orthogonalize(t_feat.get(6) - t_feat.get(2), base_15))
        s35 = self._csrank(self._orthogonalize(t_feat.get(3) - t_feat.get(5), base_15))

        # Signed geometric interactions + multi-hour rank momentum
        geom_156 = np.sign(t_feat.get(1) - t_feat.get(5)) * np.sqrt(
            (t_feat.get(1) - t_feat.get(5)).abs() * r.get(6).abs()
        )
        geom_624 = np.sign(t_feat.get(6) - t_feat.get(2)) * np.sqrt(
            (t_feat.get(6) - t_feat.get(2)).abs() * r.get(4).abs()
        )
        v6_slow = self._csrank(r.get(6) - r.get(6).ewm(span=72, min_periods=1).mean())

        return {
            "s15": s15,
            "s62": s62,
            "s35": s35,
            "geom_156": geom_156,
            "geom_624": geom_624,
            "v6_slow": v6_slow,
            "prod_16": t_feat.get(1) * t_feat.get(6),
            "prod_36": t_feat.get(3) * t_feat.get(6),
            "t6": t_feat.get(6),
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["s15"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["s15"].index

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
        target_pos = np.sign(z) * (z.abs() ** 1.38)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
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
