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
    Sign-Decoupled Orthogonal Alpha Engine (wide-format, platform-compliant).
    Exactly 6 features, cross-sectional ranks → nonlinear interactions →
    per-timestep SVD direction → hypersphere projection → L1 hysteresis.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal_series = None

        # Hyper-parameters
        self.target_bound = 0.20
        self.optimal_concentration = 0.28
        self.l1_hysteresis_threshold = 1.15

    def train(self, features: pd.DataFrame, target) -> None:
        """Lock the six feature names in deterministic order."""
        if features is None or features.empty:
            self.feature_names = None
            return
        try:
            names = list(features.columns.get_level_values(0).unique())
            self.feature_names = sorted(names)[:6]          # deterministic
        except Exception:
            self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # ------------------------------------------------------------------
        # 0. Structural guards
        # ------------------------------------------------------------------
        if (
            features is None
            or features.empty
            or self.feature_names is None
            or len(self.feature_names) < 6
        ):
            # Safe zero fallback that still has the correct shape when possible
            try:
                tickers = features.columns.get_level_values(1).unique()
                return pd.DataFrame(
                    0.0, index=features.index, columns=tickers, dtype=np.float32
                )
            except Exception:
                return pd.DataFrame(dtype=np.float32)

        try:
            tickers = features.columns.get_level_values(1).unique()
            N_time = len(features)
            J_assets = len(tickers)

            # ------------------------------------------------------------------
            # 1. Cross-sectional rank transform → [-1, 1]
            # ------------------------------------------------------------------
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[feat] = np.nan_to_num(r.to_numpy(), nan=0.0)

            f1, f2, f3, f4, f5, f6 = (
                ranks[self.feature_names[i]] for i in range(6)
            )

            # ------------------------------------------------------------------
            # 2. Nonlinear interaction blocks (12 total)
            # ------------------------------------------------------------------
            interaction_blocks = []

            # Base tanh
            for feat in self.feature_names:
                interaction_blocks.append(np.tanh(ranks[feat] * 2.0))

            # Pair-wise interactions
            interaction_blocks.append(
                np.sin(f1 * np.pi * 0.25) * np.cos(f2 * np.pi * 0.25)
            )
            interaction_blocks.append(f1 * np.abs(f2))

            interaction_blocks.append(
                np.sin(f3 * np.pi * 0.25) * np.cos(f4 * np.pi * 0.25)
            )
            interaction_blocks.append(f3 * np.abs(f4))

            interaction_blocks.append(np.arctan(f5) * np.tanh(f6))
            interaction_blocks.append(f5 * f6 * np.sign(f1))

            # Shape: [n_interactions, time, assets]
            tensor = np.stack(interaction_blocks, axis=0)

            # ------------------------------------------------------------------
            # 3. Per-timestep SVD → first right singular vector
            # ------------------------------------------------------------------
            raw_velocity = np.zeros((N_time, J_assets), dtype=np.float64)

            for t in range(N_time):
                slice_ = tensor[:, t, :]
                slice_ -= slice_.mean(axis=1, keepdims=True)
                _, _, vh = np.linalg.svd(slice_, full_matrices=False)
                direction = vh[0]
                # Force non-negative sum for sign consistency
                raw_velocity[t] = direction * np.sign(np.sum(direction) + 1e-12)

            # ------------------------------------------------------------------
            # 4. Hypersphere (L2) normalisation
            # ------------------------------------------------------------------
            demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            sphere = (demeaned / norms) * self.optimal_concentration

            # ------------------------------------------------------------------
            # 5. Causal L1 hysteresis (turnover control)
            # ------------------------------------------------------------------
            final = np.zeros_like(sphere)

            if self.prev_signal_series is not None:
                active = (
                    self.prev_signal_series
                    .reindex(tickers, fill_value=0.0)
                    .to_numpy(dtype=np.float64)
                )
            else:
                active = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target = sphere[t]
                l1 = np.sum(np.abs(target - active))

                if l1 < self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    current = 0.20 * target + 0.80 * active
                    current -= current.mean()

                final[t] = current
                active = current.copy()

            # ------------------------------------------------------------------
            # 6. Final multi-pass demean + hard clip
            # ------------------------------------------------------------------
            out = pd.DataFrame(final, index=features.index, columns=tickers)

            for _ in range(3):
                out = out.sub(out.mean(axis=1), axis=0)
                out = out.clip(-self.target_bound, self.target_bound)

            # Final safety
            out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            out = out.sub(out.mean(axis=1), axis=0)

            # Persist state for next call
            self.prev_signal_series = out.iloc[-1].astype(np.float64)

            return out.astype(np.float32)

        except Exception:
            # Absolute last-resort zero signal with correct shape
            try:
                tickers = features.columns.get_level_values(1).unique()
                return pd.DataFrame(
                    0.0, index=features.index, columns=tickers, dtype=np.float32
                )
            except Exception:
                return pd.DataFrame(dtype=np.float32)
