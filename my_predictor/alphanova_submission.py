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
    Sign-Decoupled Orthogonal Alpha Engine (SVD version)
    Rank → nonlinear interactions → leading right singular vector
    → hypersphere projection → L1 hysteresis.
    Tuned for positive out-of-sample Sharpe on AlphaNova residuals.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # Tuned hyper-parameters
        self.target_bound = 0.20
        self.optimal_concentration = 0.30   # slightly higher for better SNR
        self.l1_hysteresis_threshold = 0.95 # more responsive than 1.12
        self.blend_new = 0.30               # 30 % new / 70 % old

    def train(self, features: pd.DataFrame, target) -> None:
        """Lock feature names in deterministic order."""
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
            # ------------------------------------------------------------------
            # 1. Cross-sectional ranks → [-1, 1]
            # ------------------------------------------------------------------
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[feat] = np.nan_to_num(r.to_numpy(), nan=0.0)

            N_time, J = next(iter(ranks.values())).shape
            f = [ranks[name] for name in self.feature_names]

            # ------------------------------------------------------------------
            # 2. Nonlinear interaction library (12 blocks)
            # ------------------------------------------------------------------
            blocks = []

            # Base tanh activations
            for i in range(6):
                blocks.append(np.tanh(f[i] * 2.0))

            # Pair-wise phase-shift & product terms (selected strongest pairs)
            pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
            for i, j in pairs:
                blocks.append(np.sin(f[i] * np.pi * 0.25) * np.cos(f[j] * np.pi * 0.25))
                blocks.append(f[i] * np.abs(f[j]))

            # Stack → shape (n_blocks, T, J)
            tensor = np.stack(blocks, axis=0)

            # ------------------------------------------------------------------
            # 3. Leading right singular vector (dominant orthogonal direction)
            # ------------------------------------------------------------------
            raw = np.zeros((N_time, J), dtype=np.float64)

            for t in range(N_time):
                slice_ = tensor[:, t, :]                    # (n_blocks, J)
                slice_ -= slice_.mean(axis=1, keepdims=True)
                # SVD
                _, _, vh = np.linalg.svd(slice_, full_matrices=False)
                direction = vh[0]
                # Consistent sign (sum ≥ 0)
                raw[t] = direction * np.sign(np.sum(direction) + 1e-12)

            # Optional light residual from strongest single rank (Feature 0)
            raw += 0.15 * f[0]
            raw -= raw.mean(axis=1, keepdims=True)

            # ------------------------------------------------------------------
            # 4. Hypersphere projection
            # ------------------------------------------------------------------
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            sphere = (raw / norms) * self.optimal_concentration

            # ------------------------------------------------------------------
            # 5. Causal L1 hysteresis
            # ------------------------------------------------------------------
            final = np.zeros_like(sphere)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and len(self.prev_signal) == J
                and self.prev_tickers.equals(tickers)
            ):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(J, dtype=np.float64)

            for t in range(N_time):
                target = sphere[t]
                l1 = np.sum(np.abs(target - active))

                if l1 < self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    current = self.blend_new * target + (1.0 - self.blend_new) * active
                    current -= current.mean()

                final[t] = current
                active = current.copy()

            # ------------------------------------------------------------------
            # 6. Final compliance (demean + hard clip)
            # ------------------------------------------------------------------
            out = pd.DataFrame(final, index=features.index, columns=tickers)

            for _ in range(2):
                out = out.sub(out.mean(axis=1), axis=0)
                out = out.clip(-self.target_bound, self.target_bound)

            out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            out = out.sub(out.mean(axis=1), axis=0)

            # Persist state
            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            return zero
