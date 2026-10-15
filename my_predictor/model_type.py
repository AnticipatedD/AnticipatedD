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
    Individual Cross-Sectional Block Variance Projections +
    Orthogonal Rank Tail Sensitivity + Causal relative-L1 hysteresis
    (never collapses to zero).
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # Tradeable hyperparameters
        self.target_bound = 1.35
        self.concentration = 0.28          # usable L1 mass
        self.ema_alpha = 1.12              # response rate (was the old "decay")
        self.l1_rel_threshold = 0.20       # relative gate – only micro-churn

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
            tickers = pd.Index([])

        zero_signal = pd.DataFrame(
            0.0,
            index=getattr(features, "index", []),
            columns=tickers,
            dtype=np.float32,
        )

        if (features is None or features.empty or
                self.feature_names is None or len(self.feature_names) < 0):
            return zero_signal

        try:
            # 1. Cross-sectional rank transform → [-1, 1]
            ranks = {}
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            if len(ranks) < 2:
                return zero_signal

            N_time, J = next(iter(ranks.values())).shape
            feat_list = list(ranks.keys())
            processed_blocks = []

            # 2. Individual + pairwise blocks (your original design)
            for i, f1 in enumerate(feat_list):
                # Baseline high-sensitivity tanh
                b = np.tanh(ranks[f1] * 1.0)
                b = b - b.mean(axis=1, keepdims=True)
                nrm = np.linalg.norm(b, axis=1, keepdims=True)
                nrm = np.maximum(nrm, 1e-7)
                processed_blocks.append(b / nrm)

                for j in range(i + 1, len(feat_list)):
                    f2 = feat_list[j]

                    # Phase displacement
                    phase = np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25)
                    phase = phase - phase.mean(axis=1, keepdims=True)
                    nrm = np.linalg.norm(phase, axis=1, keepdims=True)
                    nrm = np.maximum(nrm, 1e-6)
                    processed_blocks.append(phase / nrm)

                    # Volatility asymmetry
                    vol = ranks[f1] * np.abs(ranks[f2])
                    vol = vol - vol.mean(axis=1, keepdims=True)
                    nrm = np.linalg.norm(vol, axis=1, keepdims=True)
                    nrm = np.maximum(nrm, 1e-7)
                    processed_blocks.append(vol / nrm)

            # 3. Dispersion-weighted aggregation → hypersphere
            stacked = np.stack(processed_blocks, axis=0)
            weights = np.std(stacked, axis=2, keepdims=True) + 1e-8
            velocity = np.sum(stacked * weights, axis=0)

            velocity = velocity - velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            target = (velocity / norms) * self.concentration

            # 4. Causal relative-L1 hysteresis + EMA (never stays flat)
            final = np.zeros_like(target)

            if (self.prev_signal is not None and
                    self.prev_tickers is not None and
                    self.prev_signal.shape == (J,) and
                    list(self.prev_tickers) == list(tickers)):
                active = self.prev_signal.copy()
            else:
                # Critical fix: start with the first non-zero target
                active = target[0].copy()

            for t in range(N_time):
                tgt = target[t]
                delta_l1 = np.sum(np.abs(tgt - active))
                active_l1 = np.sum(np.abs(active)) + 1e-8

                if delta_l1 < self.l1_rel_threshold * active_l1:
                    curr = active                       # pure hold
                else:
                    curr = self.ema_alpha * tgt + (1.0 - self.ema_alpha) * active
                    curr -= curr.mean()

                final[t] = curr
                active = curr

            # 5. Compliance
            out = pd.DataFrame(final, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            # Prefer last non-zero state over pure zeros
            if (self.prev_signal is not None and
                    len(self.prev_signal) == len(tickers)):
                return pd.DataFrame(
                    np.tile(self.prev_signal, (len(features), 1)),
                    index=features.index,
                    columns=tickers,
                    dtype=np.float32,
                )
            return zero_signal
