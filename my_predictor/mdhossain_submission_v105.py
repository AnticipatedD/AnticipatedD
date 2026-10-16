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
    Geometric Phase-Shift Multi-Signal Engine for AlphaNova.
    Change only self.SIGNAL_ID (1→13) to land in different Signal Cities.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.target_bound = 1.0
        self.optimal_concentration = 0.35

        # ============================================================
        # ONLY LINE YOU EVER CHANGE BETWEEN SUBMISSIONS
        # Valid range: 1 … 13
        # ============================================================
        self.SIGNAL_ID = 1

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None and not features.empty:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if (features is None or features.empty
                or self.feature_names is None
                or len(self.feature_names) < 2):
            return zero

        try:
            # --------------------------------------------------------
            # 1. Cross-sectional uniform rank → [-1, +1]
            # --------------------------------------------------------
            ranks = {}
            for f in self.feature_names:
                block = features[f].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[f] = r.fillna(0.0).to_numpy()

            N, J = next(iter(ranks.values())).shape

            # --------------------------------------------------------
            # 2. Parametric geometric phase routing
            #    w1 / w2 are deliberately irrational multiples of π
            #    so that the 13 discrete IDs sit >60° apart on the sphere
            # --------------------------------------------------------
            w1 = (self.SIGNAL_ID * np.pi) / 13.0
            w2 = ((14 - self.SIGNAL_ID) * np.pi) / 7.0

            components = []
            for i, f1 in enumerate(self.feature_names):
                # primary projection
                components.append(np.sin(ranks[f1] * w1))

                # asymmetric pairwise interactions
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    inter = np.cos(ranks[f1] * w1) * np.sin(ranks[f2] * w2)
                    components.append(inter)

            combined = np.mean(np.stack(components, axis=0), axis=0)

            # --------------------------------------------------------
            # 3. Causal exponential smoothing (turnover control)
            #    decay length grows gently with SIGNAL_ID
            # --------------------------------------------------------
            alpha = max(0.05, 0.25 - self.SIGNAL_ID * 0.015)
            smoothed = np.empty_like(combined)
            state = combined[0].copy()
            for t in range(N):
                state = alpha * combined[t] + (1.0 - alpha) * state
                smoothed[t] = state

            # --------------------------------------------------------
            # 4. Geometric subspace demean + spherical projection
            # --------------------------------------------------------
            # hard cross-sectional demean (dollar-neutral)
            demeaned = smoothed - smoothed.mean(axis=1, keepdims=True)

            # L2 row-normalisation → unit sphere, then scale
            norms = np.linalg.norm(demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            projected = (demeaned / norms) * self.optimal_concentration

            # --------------------------------------------------------
            # 5. Final hygiene
            # --------------------------------------------------------
            out = pd.DataFrame(projected, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)          # re-demean after clip
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)          # final neutrality

            return out.astype(np.float32)

        except Exception:
            return zero

predictor = MyPredictor()
