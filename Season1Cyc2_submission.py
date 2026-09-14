# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
import numpy as np
import pandas as pd
from predictor import Predictor

class MyPredictor(Predictor):
    """
    Simple cross-sectional rank + non-linear expansion
    + spherical projection + L1 hysteresis.
    Minimal dependencies, designed for positive Sharpe after costs.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 0.22
        self.optimal_concentration = 0.30
        self.l1_hysteresis_threshold = 0.90
        self.ema_alpha = 0.18

    def train(self, features: pd.DataFrame, target) -> None:
        if features is not None and len(features) > 0:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if len(features) == 0 or self.feature_names is None or len(self.feature_names) < 2:
            return zero_signal

        try:
            # 1. Cross-sectional ranks
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="max") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N, J = list(ranks.values())[0].shape

            # 2. Non-linear expansion
            blocks = []
            for i, f1 in enumerate(self.feature_names):
                blocks.append(np.tanh(ranks[f1] * 1.8))
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    blocks.append(np.sin(ranks[f1] * np.pi * 0.22) * np.cos(ranks[f2] * np.pi * 0.22))
                    blocks.append(ranks[f1] * np.abs(ranks[f2]))

            raw = np.mean(blocks, axis=0)

            # 3. Demean + spherical projection
            raw = raw - raw.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            sphere = (raw / norms) * self.optimal_concentration

            # 4. L1 hysteresis + light EMA
            final = np.zeros_like(sphere)

            if (self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (J,)
                and list(self.prev_tickers) == list(tickers)):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(J, dtype=np.float64)

            for t in range(N):
                target = sphere[t]
                l1_delta = np.sum(np.abs(target - active))

                if l1_delta < self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    current = self.ema_alpha * target + (1.0 - self.ema_alpha) * active
                    current = current - current.mean()

                final[t] = current
                active = current.copy()

            # 5. Final clean-up
            df = pd.DataFrame(final, index=features.index, columns=tickers)
            df = df.sub(df.mean(axis=1), axis=0)
            df = df.clip(-self.target_bound, self.target_bound)
            df = df.sub(df.mean(axis=1), axis=0)

            self.prev_signal = df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = df.columns.copy()

            return df.astype(np.float32)

        except Exception:
            return zero_signal
