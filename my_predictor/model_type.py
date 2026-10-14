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
    AlphaNova Season 1 - Fourth Approach
    Extreme-Tail Asymmetry + Sparse Projection
    Goal: High City Novelty
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_position = None
        self.prev_tickers = None

        # Novelty-oriented parameters
        self.tail_threshold = 0.18      # only top/bottom ~18% are active
        self.sparsity_power = 1.8       # pushes small values toward zero
        self.l1_target = 0.22
        self.decay = 0.88               # moderate memory
        self.clip_bound = 0.25

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
            self.feature_names is None or len(self.feature_names) < 1):
            return zero

        try:
            # -------------------------------------------------
            # 1. Cross-sectional quantile for every feature
            # -------------------------------------------------
            q_list = []
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                block = features[feat].astype(np.float64)
                q = block.rank(axis=1, pct=True, method="average").fillna(0.5)
                q_list.append(q.to_numpy())

            if len(q_list) == 0:
                return zero

            avg_q = np.mean(q_list, axis=0)          # (T, J)  in [0, 1]

            # -------------------------------------------------
            # 2. Extreme-tail asymmetry (key novelty driver)
            # -------------------------------------------------
            # Only keep the extreme tails, zero the middle
            upper = np.clip((avg_q - (1.0 - self.tail_threshold)) / self.tail_threshold, 0, 1)
            lower = np.clip((self.tail_threshold - avg_q) / self.tail_threshold, 0, 1)

            # Asymmetric combination: long extreme highs, short extreme lows
            # (different strength on each side creates asymmetry)
            raw = (upper ** 1.1) - (lower ** 1.4)

            # -------------------------------------------------
            # 3. Sparsity push (many near-zero weights)
            # -------------------------------------------------
            raw = np.sign(raw) * (np.abs(raw) ** self.sparsity_power)

            # -------------------------------------------------
            # 4. Exact L1 projection + strict demean
            # -------------------------------------------------
            raw = raw - raw.mean(axis=1, keepdims=True)
            l1 = np.sum(np.abs(raw), axis=1, keepdims=True) + 1e-8
            signal = raw * (self.l1_target / l1)

            # -------------------------------------------------
            # 5. Moderate temporal decay
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
            # 6. Final formatting
            # -------------------------------------------------
            df = pd.DataFrame(out, index=features.index, columns=tickers)
            df = df.clip(-self.clip_bound, self.clip_bound)
            df = df.sub(df.mean(axis=1), axis=0)

            self.prev_position = df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = df.columns.copy()

            return df.astype(np.float32)

        except Exception:
            return zero


predictor = MyPredictor()
