# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
#     "pyarrow"
# ]
# ///

import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    """
    Close to the -0.0059 version with light improvements.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 0.20
        self.optimal_concentration = 0.27
        self.l1_hysteresis_threshold = 0.95   # slightly more responsive

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is not None and not features.empty:
            try:
                self.feature_names = list(features.columns.get_level_values(0).unique())
            except Exception:
                self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            return pd.DataFrame(dtype=np.float32)

        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if (features is None or features.empty
                or not self.feature_names
                or len(self.feature_names) < 2):
            return zero_signal

        try:
            # 1. Vectorized Cross-Sectional Rank Transform
            ranks = {}
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                r = (block_val.rank(axis=1, pct=True, method='average') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = list(ranks.values())[0].shape

            # 2. Controlled Non-Linear Expansion (closer to your best code)
            interaction_blocks = []

            # Strong emphasis on first feature
            f0 = self.feature_names[0]
            interaction_blocks.append(1.15 * np.tanh(ranks[f0] * 1.8))

            # Mild contribution from second feature (reversal-style)
            if len(self.feature_names) > 1:
                f1 = self.feature_names[1]
                interaction_blocks.append(-0.35 * np.tanh(ranks[f1] * 1.5))

            # A few stable interactions only
            for i in range(min(3, len(self.feature_names))):
                for j in range(i + 1, min(4, len(self.feature_names))):
                    f_a = self.feature_names[i]
                    f_b = self.feature_names[j]
                    interaction_blocks.append(
                        0.25 * ranks[f_a] * np.abs(ranks[f_b])
                    )

            # Average the selected blocks
            raw_velocity = np.mean(interaction_blocks, axis=0)

            # 3. Demean + scale to target concentration
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)

            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # 4. L1 Causal Hysteresis (same style as your best code)
            final_positions = np.zeros_like(sphere_target)

            if (self.prev_signal is not None
                    and self.prev_tickers is not None
                    and self.prev_signal.shape == (J_assets,)
                    and self.prev_tickers.equals(tickers)):
                active_position = self.prev_signal.copy()
            else:
                active_position = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target_position = sphere_target[t]
                l1_delta = np.sum(np.abs(target_position - active_position))

                if l1_delta < self.l1_hysteresis_threshold:
                    current_allocation = active_position.copy()
                else:
                    current_allocation = 0.22 * target_position + 0.78 * active_position
                    current_allocation -= current_allocation.mean()

                final_positions[t] = current_allocation
                active_position = current_allocation.copy()

            # 5. Final compliance
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal
