# /// script
# dependencies = [
#   "numpy",
#   "pandas"
# ]
# ///

import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    """
    Asymmetric Rank-Difference & Product Engine
    Designed for high City / Global novelty while improving stability
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_concentration = 0.24
        self.ema_alpha = 0.15          # slightly stronger smoothing
        self.clip_bound = 0.22

    def train(self, features: pd.DataFrame, target=None) -> None:
        if features is not None and not features.empty:
            try:
                names = list(features.columns.get_level_values(0).unique())
                self.feature_names = sorted(names)[:6]
            except Exception:
                self.feature_names = None
        else:
            self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        zero = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if (
            features is None
            or features.empty
            or self.feature_names is None
            or len(self.feature_names) < 6
        ):
            return zero

        try:
            # 1. Cross-sectional ranks → [-1, 1]
            ranks = []
            for name in self.feature_names:
                block = features[name].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks.append(np.nan_to_num(r.to_numpy(), nan=0.0))

            f = ranks  # list of (T, J)
            T, J = f[0].shape

            # 2. Asymmetric difference + product basis (novelty driver)
            s = np.zeros((T, J), dtype=np.float64)

            # Strong differences
            s += 1.00 * (f[0] - f[3])
            s += 0.70 * (f[1] - f[4])
            s += 0.50 * (f[2] - f[5])

            # Selected asymmetric products
            s += 0.40 * (f[0] * np.abs(f[5]))
            s += 0.35 * (f[1] * f[2])
            s += 0.30 * (np.sign(f[3]) * f[4] * f[5])

            # Higher-order term
            s += 0.25 * (f[0] * f[1] - f[2] * f[3])

            # 3. Demean
            s -= s.mean(axis=1, keepdims=True)

            # 4. Scale to target concentration
            norms = np.linalg.norm(s, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            signal = s / norms * self.target_concentration

            # 5. Causal EMA (turnover control)
            out = np.zeros_like(signal)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and len(self.prev_signal) == J
                and list(self.prev_tickers) == list(tickers)
            ):
                active = self.prev_signal.copy()
            else:
                active = signal[0].copy()

            for t in range(T):
                active = self.ema_alpha * signal[t] + (1.0 - self.ema_alpha) * active
                active -= active.mean()
                out[t] = active

            # 6. Final cleanup
            out -= out.mean(axis=1, keepdims=True)
            out = np.clip(out, -self.clip_bound, self.clip_bound)
            out = np.nan_to_num(out, nan=0.0)

            self.prev_signal = out[-1].copy()
            self.prev_tickers = list(tickers)

            return pd.DataFrame(out.astype(np.float32), index=features.index, columns=tickers)

        except Exception:
            return zero


predictor = MyPredictor()
