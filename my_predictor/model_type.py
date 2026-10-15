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
    AlphaNova Biweekly Season 1 Cycle 5
    Cross-sectional rank interactions + dispersion-weighted aggregation
    + mild causal EMA hysteresis (prevents micro-churn without going flat).
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # Calibrated, tradeable hyperparameters
        self.target_bound = 1.5
        self.concentration = 0.35          # radius of the hypersphere
        self.ema_alpha = 0.35              # higher = more responsive
        self.l1_rel_threshold = 0.25       # relative L1 gate (fraction of current L1)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None and len(features.columns) > 0:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if (len(features) == 0 or self.feature_names is None
                or len(self.feature_names) < 2):
            return zero_signal

        try:
            # 1. Cross-sectional rank transform → [-1, 1]
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J = next(iter(ranks.values())).shape

            # 2. Build individual + pairwise interaction blocks
            blocks = []
            for i, f1 in enumerate(self.feature_names):
                # Base non-linear transform
                b = np.tanh(ranks[f1] * 0.8)
                b = b - b.mean(axis=1, keepdims=True)
                nrm = np.linalg.norm(b, axis=1, keepdims=True)
                nrm = np.maximum(nrm, 1e-8)
                blocks.append(b / nrm)

                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    # Phase interaction
                    phase = np.sin(ranks[f1] * np.pi * 0.3) * np.cos(ranks[f2] * np.pi * 0.3)
                    phase = phase - phase.mean(axis=1, keepdims=True)
                    nrm = np.linalg.norm(phase, axis=1, keepdims=True)
                    nrm = np.maximum(nrm, 1e-8)
                    blocks.append(phase / nrm)

                    # Volatility asymmetry
                    vol = ranks[f1] * np.abs(ranks[f2])
                    vol = vol - vol.mean(axis=1, keepdims=True)
                    nrm = np.linalg.norm(vol, axis=1, keepdims=True)
                    nrm = np.maximum(nrm, 1e-8)
                    blocks.append(vol / nrm)

            # 3. Dispersion-weighted aggregation → hypersphere projection
            stacked = np.stack(blocks, axis=0)                    # (B, T, J)
            weights = np.std(stacked, axis=2, keepdims=True) + 1e-8
            velocity = np.sum(stacked * weights, axis=0)          # (T, J)

            velocity = velocity - velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-9)
            target = (velocity / norms) * self.concentration

            # 4. Causal EMA + relative L1 hysteresis (never stays at zero)
            final = np.zeros_like(target)

            if (self.prev_signal is not None
                    and self.prev_tickers is not None
                    and self.prev_signal.shape == (J,)
                    and list(self.prev_tickers) == list(tickers)):
                active = self.prev_signal.copy()
            else:
                # Start with a non-zero position on the first bar
                active = target[0].copy()

            for t in range(N_time):
                tgt = target[t]
                delta_l1 = np.sum(np.abs(tgt - active))
                active_l1 = np.sum(np.abs(active)) + 1e-8

                if delta_l1 < self.l1_rel_threshold * active_l1:
                    # Small move → pure hold (zero transaction cost)
                    curr = active
                else:
                    # Meaningful move → EMA update
                    curr = self.ema_alpha * tgt + (1.0 - self.ema_alpha) * active
                    curr = curr - curr.mean()

                final[t] = curr
                active = curr

            # 5. Final compliance
            out = pd.DataFrame(final, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            # Persist state
            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            # Even on error try to return the previous non-zero signal if possible
            if self.prev_signal is not None and len(self.prev_signal) == len(tickers):
                return pd.DataFrame(
                    np.tile(self.prev_signal, (len(features), 1)),
                    index=features.index,
                    columns=tickers,
                    dtype=np.float32,
                )
            return zero_signal
