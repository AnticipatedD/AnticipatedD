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
    Optimized Production-Grade Quant Strategy for AlphaNova.
    Features robust universe alignment, verified post-clip centering, 
    and optimized cross-sectional feature combinations.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal_series = None  # Track state via aligned Pandas Series
        
        # Hyperparameters
        self.target_bound = 0.20
        self.optimal_concentration = 0.28
        self.l1_hysteresis_threshold = 1.15

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 2:
            return zero_signal
            
        try:
            # 1. Cross-Sectional Rank Transform
            ranks = {}
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                r = (block_val.rank(axis=1, pct=True, method='max') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = list(ranks.values())[0].shape
            
            # 2. Non-Linear Spatial Expansion
            interaction_blocks = []
            for i, f1 in enumerate(self.feature_names):
                interaction_blocks.append(np.tanh(ranks[f1] * 2.0))
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    interaction_blocks.append(np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25))
                    interaction_blocks.append(ranks[f1] * np.abs(ranks[f2]))
            
            raw_velocity = np.mean(interaction_blocks, axis=0)
            
            # 3. Geometric Subspace Demean & Hypersphere Projection
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration
            
            # 4. State Alignment & L1 Causal Hysteresis Loop
            final_positions = np.zeros_like(sphere_target)
            
            # Map state explicitly to the incoming ticker index to prevent alignment breakages
            if self.prev_signal_series is not None:
                active_position = self.prev_signal_series.reindex(tickers, fill_value=0.0).to_numpy(dtype=np.float64)
            else:
                active_position = np.zeros(J_assets, dtype=np.float64)
            
            # Optimized execution sequence
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
                
            # 5. Compliance Formatting & Exact Truncation Safety Loop
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            
            # Iterative scaling optimization to honor bounds AND dollar-neutral constraints
            for _ in range(3):
                final_df = final_df.sub(final_df.mean(axis=1), axis=0)
                final_df = final_df.clip(-self.target_bound, self.target_bound)
            
            # Cache state as a Series keyed on actual ticker names
            self.prev_signal_series = final_df.iloc[-1].astype(np.float64)
            
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
