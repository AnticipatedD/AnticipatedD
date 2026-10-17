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
    Cross-Sectional Block Variance + Orthogonal Rank Tail Sensitivity
    with strict multi-pass demeaning and intra-batch causal smoothing.
    Stateless – deterministic – zero-investment neutral.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.target_bound = 1.35
        self.optimal_concentration = 0.28

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
                or len(self.feature_names) < 2):
            return zero

        try:
            # --------------------------------------------------------
            # 1. Cross-sectional rank → [-1, +1]
            # --------------------------------------------------------
            ranks = {}
            for feat in self.feature_names:
                try:
                    block = features[feat].astype(np.float64)
                    r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                    ranks[feat] = r.fillna(0.0).to_numpy()
                except Exception:
                    continue

            if len(ranks) < 2:
                return zero

            N_time, J_assets = next(iter(ranks.values())).shape
            feat_list = list(ranks.keys())

            # --------------------------------------------------------
            # 2. Individual block processing (your original operators)
            # --------------------------------------------------------
            processed = []

            for i, f1 in enumerate(feat_list):
                # Baseline tanh
                b = np.tanh(ranks[f1] * 0.5) * 2.0
                b -= b.mean(axis=1, keepdims=True)
                n = np.linalg.norm(b, axis=1, keepdims=True)
                n = np.maximum(n, 1e-8)
                processed.append(b / n)

                for j in range(i + 1, len(feat_list)):
                    f2 = feat_list[j]

                    # Phase displacement
                    p = np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25)
                    p -= p.mean(axis=1, keepdims=True)
                    n = np.linalg.norm(p, axis=1, keepdims=True)
                    n = np.maximum(n, 1e-8)
                    processed.append(p / n)

                    # Volatility asymmetry
                    v = ranks[f1] * np.abs(ranks[f2])
                    v -= v.mean(axis=1, keepdims=True)
                    n = np.linalg.norm(v, axis=1, keepdims=True)
                    n = np.maximum(n, 1e-8)
                    processed.append(v / n)

            stacked = np.stack(processed, axis=0)          # (B, T, J)

            # Weight by cross-sectional dispersion
            weights = np.std(stacked, axis=2, keepdims=True) + 1e-8
            combined = np.sum(stacked * weights, axis=0)   # (T, J)

            # --------------------------------------------------------
            # 3. Subspace demean + spherical projection
            # --------------------------------------------------------
            combined -= combined.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            target = (combined / norms) * self.optimal_concentration

            # --------------------------------------------------------
            # 4. Intra-batch causal smoothing (stateless, kills churn)
            # --------------------------------------------------------
            alpha = 0.12          # strong memory → low turnover
            smoothed = np.empty_like(target)
            state = target[0].copy()
            for t in range(N_time):
                state = alpha * target[t] + (1.0 - alpha) * state
                state -= state.mean()          # keep neutral every step
                smoothed[t] = state

            # --------------------------------------------------------
            # 5. Final clip + mandatory demean (residual < 1e-6)
            # --------------------------------------------------------
            smoothed = np.clip(smoothed, -self.target_bound, self.target_bound)
            final = smoothed - smoothed.mean(axis=1, keepdims=True)

            out = pd.DataFrame(final, index=features.index, columns=tickers)

            # Absolute neutrality verification
            if out.sum(axis=1).abs().max() > 1e-6:
                out = out.sub(out.mean(axis=1), axis=0)

            return out.astype(np.float32)

        except Exception:
            return zero

predictor = MyPredictor()
