# /// script
# dependencies = [
# "numpy",
# "pandas",
# ]
# ///
import numpy as np
import pandas as pd
from predictor import Predictor

class MyPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        self.target_bound = 0.20
        self.optimal_concentration = 0.35  # Moderate concentration
        
        # FIXED: Lowered threshold allows allocations to update rather than trapping them at 0
        self.l1_hysteresis_threshold = 0.25 
        
        # COMPETITION OVERRIDE: Set to True if validation continues to show a persistently negative IC
        self.sign_flip = False 

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0 or not self.feature_names:
            return zero_signal
            
        try:
            # 1. Rank Transform
            ranks = {}
            for feat in self.feature_names:
                r = (features[feat].rank(axis=1, pct=True, method='max') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = list(ranks.values())[0].shape
            
            # 2. Corrected Block Interaction Generation
            interaction_blocks = []
            for i in range(len(self.feature_names)):
                f1 = self.feature_names[i]
                interaction_blocks.append(np.tanh(ranks[f1] * 3.0))
                
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    interaction_blocks.append(ranks[f1] * np.sign(ranks[f2]) * (np.abs(ranks[f2]) ** 0.5))

            # Stack along a clean front axis to avoid structural compression anomalies
            raw_elements = np.stack(interaction_blocks, axis=0) 
            magnitudes = np.abs(raw_elements)
            row_weights = magnitudes ** 2 
            
            # Cross-sectional element-wise weighted consensus
            raw_velocity = np.sum(raw_elements * row_weights, axis=0) / (np.sum(row_weights, axis=0) + 1e-10)
            raw_velocity = np.nan_to_num(raw_velocity)

            # 3. Geometric Subspace Demean & Hypersphere S^{J-2} Projection
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration 

            # 4. Execution Filter: L1 Causal Hysteresis Loop
            final_positions = np.zeros_like(sphere_target)
            
            if (
                self.prev_signal is not None 
                and self.prev_tickers is not None 
                and self.prev_signal.shape == (J_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                active_position = self.prev_signal.copy()
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

            # 5. Compliance Formatting & Final Re-Centering
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            
            # Apply absolute neutralization boundaries
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            # Optional alpha direction fix
            if self.sign_flip:
                final_df = -final_df
            
            # Update snapshot memory
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()
            
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
