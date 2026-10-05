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
    Advanced Quant Strategy Strategy 3: Sign-Decoupled Orthogonal Alpha Engine.
    Fully un-fragmented implementation map-locking exactly 6 explicit features.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal_series = None

        # Operational constraints
        self.target_bound = 0.20
        self.optimal_concentration = 0.28
        self.l1_hysteresis_threshold = 1.12

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Extract and lock the unique cross-sectional feature names from the multi-index.
        Order is made deterministic via sorting.
        """
        if features is not None and not features.empty:
            # Level 0 = feature names (official MultiIndex: (feature_name, ticker))
            names = list(features.columns.get_level_values(0).unique())
            self.feature_names = sorted(names)  # deterministic order
        else:
            self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(
            0.0, index=features.index, columns=tickers, dtype=np.float32
        )

        # Guard: need data and exactly the expected 6 features
        if (
            len(features) == 0
            or self.feature_names is None
            or len(self.feature_names) < 6
        ):
            return zero_signal

        try:
            # ------------------------------------------------------------------
            # 1. Shuffling-Immune Cross-Sectional Rank Transform → [-1, 1]
            # ------------------------------------------------------------------
            ranks = {}
            for feat in self.feature_names[:6]:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="max") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = next(iter(ranks.values())).shape

            f1 = ranks[self.feature_names[0]]
            f2 = ranks[self.feature_names[1]]
            f3 = ranks[self.feature_names[2]]
            f4 = ranks[self.feature_names[3]]
            f5 = ranks[self.feature_names[4]]
            f6 = ranks[self.feature_names[5]]

            # ------------------------------------------------------------------
            # 2. Complete 6-Feature Non-Linear Spatial Interaction Layout
            # ------------------------------------------------------------------
            interaction_blocks = []

            # Base tanh activations
            for f_name in self.feature_names[:6]:
                interaction_blocks.append(np.tanh(ranks[f_name] * 2.0))

            # Cross-feature interactions
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
            tensor_blocks = np.stack(interaction_blocks, axis=0)

            # ------------------------------------------------------------------
            # 3. Asymmetric Directional Weight Projection (per-timestep SVD)
            # ------------------------------------------------------------------
            raw_velocity = np.zeros((N_time, J_assets), dtype=np.float64)

            for t in range(N_time):
                cross_slice = tensor_blocks[:, t, :]          # [M, J]
                cross_slice = cross_slice - cross_slice.mean(axis=1, keepdims=True)

                # SVD – first right singular vector
                _, _, vh = np.linalg.svd(cross_slice, full_matrices=False)
                direction = vh[0]
                # Align sign so that the sum is non-negative
                raw_velocity[t] = direction * np.sign(np.sum(direction) + 1e-12)

            # ------------------------------------------------------------------
            # 4. Global Hypersphere Normalization
            # ------------------------------------------------------------------
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # ------------------------------------------------------------------
            # 5. Asset Mapping & L1 Execution Hysteresis
            # ------------------------------------------------------------------
            final_positions = np.zeros_like(sphere_target)

            if self.prev_signal_series is not None:
                active = (
                    self.prev_signal_series
                    .reindex(tickers, fill_value=0.0)
                    .to_numpy(dtype=np.float64)
                )
            else:
                active = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target = sphere_target[t]
                l1_delta = np.sum(np.abs(target - active))

                if l1_delta < self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    current = 0.20 * target + 0.80 * active
                    current -= current.mean()

                final_positions[t] = current
                active = current.copy()

            # ------------------------------------------------------------------
            # 6. Final Clean Post-Clip Demean Check
            # ------------------------------------------------------------------
            final_df = pd.DataFrame(
                final_positions, index=features.index, columns=tickers
            )

            for _ in range(3):
                final_df = final_df.sub(final_df.mean(axis=1), axis=0)
                final_df = final_df.clip(-self.target_bound, self.target_bound)

            # Cache last signal for next call (stateful hysteresis)
            self.prev_signal_series = final_df.iloc[-1].astype(np.float64)

            # Final safety: ensure no NaNs / Infs and exact demean
            final_df = final_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal
