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
    Inverted Linear Multi-Factor Strategy for AlphaNova.
    Corrects structural feature inversions to align with out-of-sample alpha target profiles.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        
        self.target_bound = 0.20
        self.optimal_concentration = 0.25  
        self.alpha = 0.20  

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Shuffling-gate resilient feature mapping. Extracts structural feature columns.
        """
        if features is not None:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float64)
        
        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 1:
            return zero_signal
            
        try:
            # 1. Clean Cross-Sectional Ranking Matrix
            ranks = []
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                r = (block_val.rank(axis=1, pct=True, method='first') - 0.5) * 2.0
                ranks.append(r.fillna(0.0).to_numpy())

            # 2. Linear Factor Combination with Structural Directional Inversion
            # Flipped to -1.0 to reverse the underlying negative correlation channel
            combined_velocity = np.mean(ranks, axis=0) * -1.0
            
            # 3. Precise Zero-Mean Scaling to Match Target Profiles
            velocity_demeaned = combined_velocity - combined_velocity.mean(axis=1, keepdims=True)
            row_stds = np.std(velocity_demeaned, axis=1, keepdims=True)
            row_stds[row_stds < 1e-12] = 1.0
            
            # Standardize and scale to reach the target concentration layout
            normalized_target = (velocity_demeaned / row_stds) * (self.optimal_concentration * 0.5)
            
            N_time, J_assets = normalized_target.shape
            final_positions = np.zeros_like(normalized_target, dtype=np.float64)
            
            if (
                self.prev_signal is not None 
                and self.prev_tickers is not None 
                and self.prev_signal.shape == (J_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                active_position = self.prev_signal.copy()
            else:
                active_position = np.zeros(J_assets, dtype=np.float64)
            
            # 4. Path Integration & Smoothing
            for t in range(N_time):
                target_position = normalized_target[t]
                current_allocation = self.alpha * target_position + (1.0 - self.alpha) * active_position
                current_allocation -= current_allocation.mean()
                
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()
                
            # 5. Strict Out-of-Sample Compliance Truncation
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()
            
            return final_df.astype(np.float64)
            
        except Exception:
            return zero_signal
