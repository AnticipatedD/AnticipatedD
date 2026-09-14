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

        self.target_bound = 0.28
        self.l1_threshold = 1.12          # much looser
        self.ema_alpha = 0.14             # faster reaction

    def train(self, features: pd.DataFrame, target) -> None:
        if features is not None and len(features) > 0:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if len(features) == 0 or self.feature_names is None or len(self.feature_names) < 2:
            return zero

        try:
            # 1. Simple cross-sectional ranks
            ranks = []
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks.append(r.fillna(0.0).to_numpy())
            ranks = np.stack(ranks, axis=-1)          # (T, J, F)

            T, J, F = ranks.shape

            # 2. Stronger non-linear combination (different from previous)
            signal = np.zeros((T, J))
            for f in range(F):
                x = ranks[:, :, f]
                signal += np.tanh(x * 2.8)
                signal += np.sign(x) * (np.abs(x) ** 1.2)

            # pairwise
            for i in range(F):
                for j in range(i+1, F):
                    signal += ranks[:, :, i] * ranks[:, :, j] * 0.7

            # 3. Only demean (NO spherical projection – this was killing concentration)
            signal = signal - signal.mean(axis=1, keepdims=True)

            # scale to reasonable magnitude
            stds = np.std(signal, axis=1, keepdims=True)
            stds = np.maximum(stds, 1e-12)
            signal = signal / stds * 0.28

            # 4. Mild turnover control only
            final = np.zeros_like(signal)
            if (self.prev_signal is not None and
                self.prev_tickers is not None and
                self.prev_signal.shape == (J,) and
                list(self.prev_tickers) == list(tickers)):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(J)

            for t in range(T):
                target = signal[t]
                l1 = np.sum(np.abs(target - active))
                if l1 < self.l1_threshold:
                    current = active
                else:
                    current = self.ema_alpha * target + (1.0 - self.ema_alpha) * active
                    current -= current.mean()
                final[t] = current
                active = current.copy()

            # 5. Final safety
            df = pd.DataFrame(final, index=features.index, columns=tickers)
            df = df.sub(df.mean(axis=1), axis=0)
            df = df.clip(-self.target_bound, self.target_bound)
            df = df.sub(df.mean(axis=1), axis=0)

            self.prev_signal = df.iloc[-1].to_numpy()
            self.prev_tickers = df.columns.copy()
            return df.astype(np.float32)

        except Exception:
            return zero
