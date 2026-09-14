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
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # restored stronger concentration + milder hysteresis
        self.target_bound = 0.30
        self.optimal_concentration = 0.32
        self.l1_hysteresis_threshold = 1.05
        self.ema_alpha = 0.22

    def train(self, features: pd.DataFrame, target) -> None:
        if features is not None and len(features) > 0:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if len(features) == 0 or self.feature_names is None or len(self.feature_names) < 2:
            return zero

        try:
            # 1. ranks
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="max") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N, J = list(ranks.values())[0].shape

            # 2. stronger & slightly different non-linear expansion
            #    (changes the geometric fingerprint → higher city/global angle)
            blocks = []
            for i, f1 in enumerate(self.feature_names):
                x = ranks[f1]
                blocks.append(np.tanh(x * 2.4))
                blocks.append(np.sign(x) * np.sqrt(np.abs(x)))          # new term
                for j in range(i + 1, len(self.feature_names)):
                    y = ranks[self.feature_names[j]]
                    blocks.append(np.sin(x * np.pi * 0.31) * np.cos(y * np.pi * 0.19))
                    blocks.append(x * y * (1.0 - np.abs(x)))             # new interaction

            raw = np.mean(blocks, axis=0)

            # 3. demean + spherical projection (stronger concentration)
            raw = raw - raw.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            sphere = (raw / norms) * self.optimal_concentration

            # 4. milder L1 hysteresis + EMA
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
                l1 = np.sum(np.abs(target - active))
                if l1 < self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    current = self.ema_alpha * target + (1.0 - self.ema_alpha) * active
                    current -= current.mean()
                final[t] = current
                active = current.copy()

            # 5. final compliance
            df = pd.DataFrame(final, index=features.index, columns=tickers)
            df = df.sub(df.mean(axis=1), axis=0)
            df = df.clip(-self.target_bound, self.target_bound)
            df = df.sub(df.mean(axis=1), axis=0)

            self.prev_signal = df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = df.columns.copy()
            return df.astype(np.float32)

        except Exception:
            return zero
