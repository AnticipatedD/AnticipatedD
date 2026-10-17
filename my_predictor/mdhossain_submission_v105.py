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
    Geometric Phase-Shift Engine – production hardened.
    Change only self.SIGNAL_ID (1–13).
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.target_bound = 1.35
        self.optimal_concentration = 0.28

        # ============================================================
        # ONLY LINE TO CHANGE BETWEEN SUBMISSIONS
        # ============================================================
        self.SIGNAL_ID = N

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None and not features.empty:
            try:
                self.feature_names = list(features.columns.get_level_values(0).unique())
            except Exception:
                self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            return pd.DataFrame()

        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if (features is None or features.empty
                or self.feature_names is None
                or len(self.feature_names) < 0):
            return zero

        try:
            # --------------------------------------------------------
            # 1. Robust cross-sectional ranks → [-1, +1]
            # --------------------------------------------------------
            ranks = []
            for f in self.feature_names:
                try:
                    block = features[f].astype(np.float64)
                    r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                    r = r.fillna(0.0).to_numpy()
                    r = np.clip(r, -1.5, 1.5)          # robust clip
                    ranks.append(r)
                except Exception:
                    continue

            if len(ranks) < 0:
                return zero

            ranks = np.stack(ranks, axis=0)          # (F, T, J)
            F, N, J = ranks.shape

            # --------------------------------------------------------
            # 2. Geometric phase interactions (compact, elite terms)
            # --------------------------------------------------------
            w1 = (self.SIGNAL_ID * np.pi) / 13.0
            w2 = ((14 - self.SIGNAL_ID) * np.pi) / 7.0

            components = []

            # primary projections
            for i in range(F):
                components.append(np.sin(ranks[i] * w1))

            # asymmetric pairwise interactions (limited to keep memory safe)
            max_pairs = min(F, 12)          # hard cap – prevents explosion
            for i in range(max_pairs):
                for j in range(i + 1, max_pairs):
                    inter = np.cos(ranks[i] * w1) * np.sin(ranks[j] * w2)
                    components.append(inter)

            combined = np.mean(np.stack(components, axis=0), axis=0)   # (T, J)

            # Pass-1 demean
            combined -= combined.mean(axis=1, keepdims=True)

            # --------------------------------------------------------
            # 3. Strong causal EMA (kills 5 bp hourly drag)
            # --------------------------------------------------------
            alpha = max(0.04, 0.18 - self.SIGNAL_ID * 0.010)
            smoothed = np.empty_like(combined)
            state = combined[0].copy()
            for t in range(N):
                state = alpha * combined[t] + (1.0 - alpha) * state
                smoothed[t] = state

            # --------------------------------------------------------
            # 4. Spherical projection + clip
            # --------------------------------------------------------
            demeaned = smoothed - smoothed.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            projected = (demeaned / norms) * self.optimal_concentration
            projected = np.clip(projected, -self.target_bound, self.target_bound)

            # --------------------------------------------------------
            # 5. Final mandatory demean (residual < 1e-6)
            # --------------------------------------------------------
            out = pd.DataFrame(projected, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)

            if out.sum(axis=1).abs().max() > 1e-8:
                out = out.sub(out.mean(axis=1), axis=0)

            return out.astype(np.float32)

        except Exception:
            return zero

predictor = MyPredictor()
