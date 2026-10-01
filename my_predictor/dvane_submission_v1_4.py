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
        """
        if features is not None:
            # Safely capture the specific tracking tokens across Level 0 of columns
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        # Ensure we have data and exactly 6 features available to prevent index crashes
        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 6:
            return zero_signal
            
        try:
            # 1. Shuffling-Immune Cross-Sectional Rank Transform
            ranks = {}
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                r = (block_val.rank(axis=1, pct=True, method='max') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            # Read structural dimensions from our generated ranks matrix
            N_time, J_assets = list(ranks.values())[0].shape
            
            # Extract and explicitly map each individual feature tracking layer
            f1_rank = ranks[self.feature_names[0]]
            f2_rank = ranks[self.feature_names[1]]
            f3_rank = ranks[self.feature_names[2]]
            f4_rank = ranks[self.feature_names[3]]
            f5_rank = ranks[self.feature_names[4]]
            f6_rank = ranks[self.feature_names[5]]

            # 2. Complete 6-Feature Non-Linear Spatial Interaction Layout
            interaction_blocks = []
            
            # Map structural tanh activations for base ranks
            for f_name in self.feature_names[:6]:
                interaction_blocks.append(np.tanh(ranks[f_name] * 2.0))
                
            # Cross-interaction mappings across features
            interaction_blocks.append(np.sin(f1_rank * np.pi * 0.25) * np.cos(f2_rank * np.pi * 0.25))
            interaction_blocks.append(f1_rank * np.abs(f2_rank))
            
            interaction_blocks.append(np.sin(f3_rank * np.pi * 0.25) * np.cos(f4_rank * np.pi * 0.25))
            interaction_blocks.append(f3_rank * np.abs(f4_rank))
            
            # Dedicated 5th and 6th feature interaction closure lines
            interaction_blocks.append(np.arctan(f5_rank) * np.tanh(f6_rank))
            interaction_blocks.append(f5_rank * f6_rank * np.sign(f1_rank))

            # Pack structures into a multi-dimensional array [Interactions, Time, Assets]
            tensor_blocks = np.stack(interaction_blocks, axis=0)
            
            # 3. Asymmetric Directional Weight Projection Loop
            raw_velocity = np.zeros((N_time, J_assets), dtype=np.float64)
            for t in range(N_time):
                cross_slice = tensor_blocks[:, t, :]  # Dimension: [M_interactions, J_assets]
                cross_slice -= cross_slice.mean(axis=1, keepdims=True)
                
                # Execute SVD factorization over the snapshot step
                u, s, vh = np.linalg.svd(cross_slice, full_matrices=False)
                
                # Extract first orthogonal vector component and align sign directionally
                raw_velocity[t] = vh[0] * np.sign(np.sum(vh[0]))
                
            # 4. Global Hypersphere Normalization
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration
            
            # 5. Asset Mapping & L1 Execution Hysteresis Matrix Logic
            final_positions = np.zeros_like(sphere_target)
            if self.prev_signal_series is not None:
                active_position = self.prev_signal_series.reindex(tickers, fill_value=0.0).to_numpy(dtype=np.float64)
            else:
                active_position = np.zeros(J_assets, dtype=np.float64)
            
            for t in range(N_time):
                target_position = sphere_target[t]
                l1_allocation_delta = np.sum(np.abs(target_position - active_position))
                
                if l1_allocation_delta < self.l1_hysteresis_threshold:
                    current_allocation = active_position.copy()
                else:
                    current_allocation = 0.20 * target_position + 0.80 * active_position
                    current_allocation -= current_allocation.mean()
                
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()
                
            # 6. Final Clean Post-Clip Demean Check
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            for _ in range(3):
                final_df = final_df.sub(final_df.mean(axis=1), axis=0)
                final_df = final_df.clip(-self.target_bound, self.target_bound)
            
            # Cache ongoing structural vector back into memory
            self.prev_signal_series = final_df.iloc[-1].astype(np.float64)
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
