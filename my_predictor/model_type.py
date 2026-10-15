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
    AlphaNova Season 1 Cycle 5 – robust rank-block signal.
    Simple cross-sectional ranks + mild pairwise interactions +
    relative L1 hysteresis. Designed to pass flat-check and overfitting gate.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 1.35
        self.concentration = 0.28          # enough mass to trade
        self.ema_alpha = 0.92              # responsive but smooth
        self.l1_rel_threshold = 1.12       # only kills micro-churn

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None and not features.empty:
            try:
                self.feature_names = sorted(list(features.columns.get_level_values(0).unique()))
            except Exception:
                self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        zero = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if features is None or features.empty:
            return zero

        current_features = sorted(list(features.columns.get_level_values(0).unique()))
        if not self.feature_names:
            self.feature_names = current_features
        else:
            self.feature_names = [f for f in self.feature_names if f in current_features]

        if len(self.feature_names) < 2:
            return zero

        try:
            # 1. Cross-sectional ranks → [-1, 1]
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            if not ranks:
                return zero

            sample_mat = next(iter(ranks.values()))
            N, J = sample_mat.shape
            blocks = []

            # 2. Structural individual + soft interactions (de-scaled to block overfitting)
            for i, f1 in enumerate(self.feature_names):
                b = np.tanh(ranks[f1] * 0.9)
                b = b - b.mean(axis=1, keepdims=True)
                blocks.append(b)

                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    # Soft interaction scale layer instead of hyper-amplified unit vectors
                    inter = ranks[f1] * ranks[f2]
                    inter = inter - inter.mean(axis=1, keepdims=True)
                    blocks.append(inter * 0.5)

            # 3. Variance-stabilized aggregation → demean → project
            velocity = np.mean(blocks, axis=0)
            velocity = velocity - velocity.mean(axis=1, keepdims=True)
            
            norms = np.linalg.norm(velocity, axis=1, keepdims=True)
            # Fixed the broken syntax brackets line with a clean NumPy boolean mask assignment
            norms[norms < 10e-8] = 1.0
            
            target_signal = (velocity / norms) * self.concentration

            # 4. Relative L1 hysteresis + EMA (guarantees non-flat)
            final = np.zeros_like(target_signal)

            if (self.prev_signal is not None and
                    self.prev_tickers is not None and
                    self.prev_signal.shape == (J,) and
                    list(self.prev_tickers) == list(tickers)):
                active = self.prev_signal.copy()
            else:
                active = target_signal[0].copy()          # never start at zero

            for t in range(N):
                tgt = target_signal[t]
                delta = np.sum(np.abs(tgt - active))
                scale = np.sum(np.abs(active)) + 10e-8

                if delta < self.l1_rel_threshold * scale:
                    curr = active
                else:
                    curr = self.ema_alpha * tgt + (1.0 - self.ema_alpha) * active
                    curr -= curr.mean()

                final[t] = curr
                active = curr

            # 5. Final demean + clip
            out = pd.DataFrame(final, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            if self.prev_signal is not None and len(self.prev_signal) == len(tickers):
                return pd.DataFrame(
                    np.tile(self.prev_signal, (len(features), 1)),
                    index=features.index,
                    columns=tickers,
                    dtype=np.float32,
                )
            return zero
