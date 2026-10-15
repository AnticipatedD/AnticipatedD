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
    AlphaNova Alternative Variant B: Orthogonal Dispersion Spline.
    Utilizes cross-sectional feature standard deviations to dynamically weight alpha matrices.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        
        # PARAMETERS CALIBRATED TO CORRECT COMPRESSION LOSS
        self.target_bound = 1.38
        self.optimal_concentration = 0.27
        self.l1_hysteresis_threshold = 1.02
        self.smoothing = 0.78  # 0.22 effective execution response rate

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
            ranks = {}
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                block_val = features[feat].astype(np.float64)
                r = (block_val.rank(axis=1, pct=True, method='average') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            if len(ranks) == 0:
                return zero_signal

            N_time, J_assets = next(iter(ranks.values())).shape
            interaction_blocks = []
            feat_list = list(ranks.keys())

            # UNIQUE FEATURE STEP: Standardized Basis Transformations
            for i, f1 in enumerate(feat_list):
                interaction_blocks.append(np.tanh(ranks[f1] * 1.8))

                for j in range(i + 1, len(feat_list)):
                    f2 = feat_list[j]
                    # Radial Basis projection over the cross-section
                    interaction_blocks.append(np.sin((ranks[f1] - ranks[f2]) * np.pi * 0.5))
                    interaction_blocks.append(np.cos((ranks[f1] + ranks[f2]) * np.pi * 0.25))

            # UNIQUE AGGREGATION STEP: Cross-Sectional Dispersion Weighting
            stacked_vectors = np.stack(interaction_blocks, axis=0)
            block_weights = np.std(stacked_vectors, axis=2, keepdims=True) + 1e-6
            weighted_velocity = np.sum(stacked_vectors * block_weights, axis=0)

            velocity_demeaned = weighted_velocity - weighted_velocity.mean(axis=1, keepdims=True)

            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            final_positions = np.zeros_like(sphere_target)

            if (self.prev_signal is not None and 
                self.prev_tickers is not None and 
                len(self.prev_signal) == J_assets and 
                self.prev_tickers.equals(tickers)):
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

            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal

predictor = MyPredictor()
