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
    AlphaNova Season 1 Cycle 4 - Cross Sectional Approach
    Quantile Transform + Power Non-linearity 
    + Exact L1 Projection + Unique Signal Decay
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_position = None
        self.prev_tickers = None

        # Parameters
        self.power = 1.12              # non-linearity strength
        self.l1_target = 0.28          # exact L1 norm target
        self.decay = 0.80              # slow decay factor
        self.clip_bound = 0.20

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is not None and not features.empty:
            try:
                self.feature_names = list(features.columns.get_level_values(0).unique())
            except Exception:
                self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        zero = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if (features is None or features.empty or 
            self.feature_names is None or len(self.feature_names) < 2):
            return zero

        try:
            # -------------------------------------------------
            # 1. Cross-sectional quantile transform (0 → 1)
            # -------------------------------------------------
            q_list = []
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                block = features[feat].astype(np.float64)
                q = block.rank(axis=1, pct=True, method="average")
                q = q.fillna(0.5)
                q_list.append(q.to_numpy())

            if len(q_list) == 0:
                return zero

            # Average quantiles across features
            avg_q = np.mean(q_list, axis=0)          # (T, J)

            # Center around 0.5 → range approx [-0.5, 0.5]
            centered = avg_q - 0.5

            # -------------------------------------------------
            # 2. Power non-linearity (preserve sign)
            # -------------------------------------------------
            signal = np.sign(centered) * (np.abs(centered) ** self.power)

            # -------------------------------------------------
            # 3. Exact L1 projection
            # -------------------------------------------------
            l1 = np.sum(np.abs(signal), axis=1, keepdims=True) + 1e-10
            signal = signal * (self.l1_target / l1)

            # -------------------------------------------------
            # 4. Slow decay toward previous position
            # -------------------------------------------------
            T, J = signal.shape
            out = np.zeros_like(signal)

            if (self.prev_position is not None and 
                self.prev_tickers is not None and 
                len(self.prev_position) == J and 
                self.prev_tickers.equals(tickers)):
                prev = self.prev_position.copy()
            else:
                prev = np.zeros(J)

            for t in range(T):
                current = self.decay * prev + (1.0 - self.decay) * signal[t]
                current -= current.mean()
                out[t] = current
                prev = current.copy()

            # -------------------------------------------------
            # 5. Final formatting
            # -------------------------------------------------
            df = pd.DataFrame(out, index=features.index, columns=tickers)
            df = df.clip(-self.clip_bound, self.clip_bound)
            df = df.sub(df.mean(axis=1), axis=0)

            # Store state
            self.prev_position = df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = df.columns.copy()

            return df.astype(np.float32)

        except Exception:
            return zero


predictor = MyPredictor()
