# /// script
# dependencies = [
#   "numpy",
#   "pandas"
# ]
# ///

import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    """
    AlphaNova Production Strategy
    Cross-Sectional Rank → Non-Linear Expansion → Spherical Projection → L1 Hysteresis
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # Small focused changes
        self.target_bound = 0.19
        self.optimal_concentration = 0.28      # slightly higher concentration
        self.l1_hysteresis_threshold = 0.95    # tighter hysteresis band
        self.smoothing = 0.85                  # slightly more weight on previous position

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
            tickers = pd.Index([])

        zero_signal = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if (features is None or features.empty or 
            self.feature_names is None or len(self.feature_names) < 2):
            return zero_signal

        try:
            # 1. Cross-sectional rank transform
            ranks = {}
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                block_val = features[feat].astype(np.float64)
                r = (block_val.rank(axis=1, pct=True, method='average') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            if len(ranks) < 2:
                return zero_signal

            N_time, J_assets = next(iter(ranks.values())).shape

            # 2. Non-Linear Spatial Expansion
            interaction_blocks = []
            feat_list = list(ranks.keys())

            for i, f1 in enumerate(feat_list):
                interaction_blocks.append(np.tanh(ranks[f1] * 2.0))

                for j in range(i + 1, len(feat_list)):
                    f2 = feat_list[j]
                    interaction_blocks.append(
                        np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25)
                    )
                    interaction_blocks.append(ranks[f1] * np.abs(ranks[f2]))

            raw_velocity = np.mean(interaction_blocks, axis=0)

            # 3. Geometric demean + Spherical projection
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)

            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # 4. L1 Causal Hysteresis
            final_positions = np.zeros_like(sphere_target)

            if (self.prev_signal is not None and 
                self.prev_tickers is not None and 
                len(self.prev_signal) == J_assets and 
                list(self.prev_tickers) == list(tickers)):
                active_position = self.prev_signal.copy()
            else:
                active_position = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target_position = sphere_target[t]
                l1_delta = np.sum(np.abs(target_position - active_position))

                if l1_delta < self.l1_hysteresis_threshold:
                    current = active_position.copy()
                else:
                    current = (1.0 - self.smoothing) * target_position + self.smoothing * active_position
                    current -= current.mean()

                final_positions[t] = current
                active_position = current.copy()

            # 5. Final formatting
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = list(final_df.columns)

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal


predictor = MyPredictor()
