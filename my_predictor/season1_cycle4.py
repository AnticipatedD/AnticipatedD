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
    Novelty-oriented Orthogonal Interaction Engine for AlphaNova.
    Designed to push City / Global novelty well above 60° while
    preserving usable concentration and low turnover.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_concentration = 0.24
        self.ema_alpha = 0.22          # light smoothing for 5 bp cost
        self.l1_threshold = 0.85

    def train(self, features: pd.DataFrame, target) -> None:
        if features is not None and not features.empty:
            names = list(features.columns.get_level_values(0).unique())
            self.feature_names = sorted(names)[:6]
        else:
            self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if (
            features is None
            or len(features) == 0
            or self.feature_names is None
            or len(self.feature_names) < 6
        ):
            return zero

        try:
            # ---------------------------------------------------------------
            # 1. Cross-sectional ranks → [-1, 1]
            # ---------------------------------------------------------------
            ranks = []
            for name in self.feature_names:
                block = features[name].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks.append(np.nan_to_num(r.to_numpy(), nan=0.0))
            ranks = np.stack(ranks, axis=0)          # (6, T, J)
            f = ranks

            # ---------------------------------------------------------------
            # 2. Build a less-common interaction / phase-shift basis
            # ---------------------------------------------------------------
            blocks = []

            # Selective non-linear activations (avoid plain tanh on all)
            blocks.append(np.arctan(f[0] * 1.5))
            blocks.append(np.sin(f[1] * np.pi * 0.3))
            blocks.append(np.cos(f[2] * np.pi * 0.3))
            blocks.append(np.tanh(f[3] * 1.8))
            blocks.append(np.sin(f[4] * np.pi * 0.4) * np.cos(f[5] * np.pi * 0.25))

            # Higher-order / asymmetric products (rarer in the pot)
            blocks.append(f[0] * f[3] * np.sign(f[1]))
            blocks.append(f[1] * np.abs(f[4]))
            blocks.append(f[2] * f[5])
            blocks.append(np.sin(f[0] + f[5]) * np.cos(f[2] - f[4]))

            tensor = np.stack(blocks, axis=0)        # (n_blocks, T, J)

            # ---------------------------------------------------------------
            # 3. Dominant direction via SVD + residualize against plain rank avg
            # ---------------------------------------------------------------
            T, J = f.shape[1], f.shape[2]
            raw = np.zeros((T, J), dtype=np.float64)

            # Crowded direction we want to move away from
            plain_rank_avg = f.mean(axis=0)
            plain_rank_avg -= plain_rank_avg.mean(axis=1, keepdims=True)

            for t in range(T):
                slice_ = tensor[:, t, :]
                slice_ -= slice_.mean(axis=1, keepdims=True)

                # Leading right singular vector
                _, _, vh = np.linalg.svd(slice_, full_matrices=False)
                direction = vh[0]
                direction *= np.sign(np.sum(direction) + 1e-12)

                # Residualize against the plain rank average (novelty push)
                proj = np.dot(direction, plain_rank_avg[t]) / (np.dot(plain_rank_avg[t], plain_rank_avg[t]) + 1e-12)
                direction = direction - 0.65 * proj * plain_rank_avg[t]

                raw[t] = direction

            raw -= raw.mean(axis=1, keepdims=True)

            # ---------------------------------------------------------------
            # 4. Hypersphere scaling to healthy concentration
            # ---------------------------------------------------------------
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            signal = (raw / norms) * self.target_concentration

            # ---------------------------------------------------------------
            # 5. Light causal EMA + mild L1 protection
            # ---------------------------------------------------------------
            out = np.zeros_like(signal)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and len(self.prev_signal) == J
                and self.prev_tickers.equals(tickers)
            ):
                active = self.prev_signal.copy()
            else:
                active = signal[0].copy()

            for t in range(T):
                target = signal[t]
                l1 = np.sum(np.abs(target - active))

                if l1 < self.l1_threshold:
                    current = active.copy()
                else:
                    current = self.ema_alpha * target + (1.0 - self.ema_alpha) * active
                    current -= current.mean()

                out[t] = current
                active = current.copy()

            # ---------------------------------------------------------------
            # 6. Final demean (single clean pass)
            # ---------------------------------------------------------------
            out -= out.mean(axis=1, keepdims=True)
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

            self.prev_signal = out[-1].copy()
            self.prev_tickers = pd.Index(tickers)

            return pd.DataFrame(out.astype(np.float32), index=features.index, columns=tickers)

        except Exception:
            return zero
