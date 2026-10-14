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
    Production-Grade Quant Strategy for AlphaNova.
    Explicitly hardcoded and unrolled for exactly 6 Feature Measurements.
    """
    def __init__(self):
        super().__init__()
        self.feature_weights = np.zeros(6, dtype=np.float64)
        self.interaction_weights = np.zeros(15, dtype=np.float64)
        self.prev_signal = None
        self.prev_tickers = None
        
        # Hyperparameters optimized for 6 features and 5bp fee constraints
        self.target_bound = 1.35
        self.alpha_scale = 0.30
        self.smoothing_factor = 0.40  
        self.shrinkage = 0.15         

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is None or target is None:
            return
            
        feature_list = list(features.columns.get_level_values(0).unique())
        # Enforce strict fallback checks to guarantee exactly 6 measurements are present
        if len(feature_list) < 6:
            return
            
        common_idx = features.index.intersection(target.index)
        if len(common_idx) == 0:
            return
            
        feat_sub = features.loc[common_idx]
        targ_sub = target.loc[common_idx]
        t_rank = targ_sub.rank(axis=1, pct=True, method='max') - 0.5
        
        # 1. Unroll the 6 Base Ranks
        r0 = feat_sub[feature_list[0]].rank(axis=1, pct=True, method='max') - 0.5
        r1 = feat_sub[feature_list[1]].rank(axis=1, pct=True, method='max') - 0.5
        r2 = feat_sub[feature_list[2]].rank(axis=1, pct=True, method='max') - 0.5
        r3 = feat_sub[feature_list[3]].rank(axis=1, pct=True, method='max') - 0.5
        r4 = feat_sub[feature_list[4]].rank(axis=1, pct=True, method='max') - 0.5
        r5 = feat_sub[feature_list[5]].rank(axis=1, pct=True, method='max') - 0.5
        
        # Calculate Base Directional Covariances for all 6 features
        self.feature_weights[0] = (r0 * t_rank).mean().mean()
        self.feature_weights[1] = (r1 * t_rank).mean().mean()
        self.feature_weights[2] = (r2 * t_rank).mean().mean()
        self.feature_weights[3] = (r3 * t_rank).mean().mean()
        self.feature_weights[4] = (r4 * t_rank).mean().mean()
        self.feature_weights[5] = (r5 * t_rank).mean().mean()
        
        # 2. Extract and Unroll exactly 15 Permutation Matrix Covariances
        idx = 0
        r_all = [r0, r1, r2, r3, r4, r5]
        for i in range(6):
            for j in range(i + 1, 6):
                interact = r_all[i] * r_all[j]
                interact_demeaned = interact.sub(interact.mean(axis=1), axis=0)
                self.interaction_weights[idx] = (interact_demeaned * t_rank).mean().mean()
                idx += 1

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0:
            return zero_signal
            
        feature_list = list(features.columns.get_level_values(0).unique())
        if len(feature_list) < 6:
            return zero_signal
            
        try:
            N_time = len(features)
            J_assets = len(tickers)
            combined_signal = np.zeros((N_time, J_assets), dtype=np.float64)
            
            # Compute Ranks for all 6 Features
            r0 = (features[feature_list[0]].rank(axis=1, pct=True, method='max') - 0.5).fillna(0.0).to_numpy()
            r1 = (features[feature_list[1]].rank(axis=1, pct=True, method='max') - 0.5).fillna(0.0).to_numpy()
            r2 = (features[feature_list[2]].rank(axis=1, pct=True, method='max') - 0.5).fillna(0.0).to_numpy()
            r3 = (features[feature_list[3]].rank(axis=1, pct=True, method='max') - 0.5).fillna(0.0).to_numpy()
            r4 = (features[feature_list[4]].rank(axis=1, pct=True, method='max') - 0.5).fillna(0.0).to_numpy()
            r5 = (features[feature_list[5]].rank(axis=1, pct=True, method='max') - 0.5).fillna(0.0).to_numpy()
            
            # Add base alpha
            combined_signal += r0 * self.feature_weights[0]
            combined_signal += r1 * self.feature_weights[1]
            combined_signal += r2 * self.feature_weights[2]
            combined_signal += r3 * self.feature_weights[3]
            combined_signal += r4 * self.feature_weights[4]
            combined_signal += r5 * self.feature_weights[5]
            
            # Compute all 15 Inter-Feature pairs with shrinkage
            r_all = [r0, r1, r2, r3, r4, r5]
            idx = 0
            for i in range(6):
                for j in range(i + 1, 6):
                    interact_signal = r_all[i] * r_all[j]
                    interact_signal -= interact_signal.mean(axis=1, keepdims=True)
                    combined_signal += interact_signal * (self.interaction_weights[idx] * self.shrinkage)
                    idx += 1

            # Dollar neutralization
            combined_signal -= combined_signal.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(combined_signal, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            target_positions = (combined_signal / norms) * self.alpha_scale
            
            # Dynamic smoother to mitigate trading friction 
            final_positions = np.zeros_like(target_positions)
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
                active_position = (self.smoothing_factor * target_positions[t]) + ((1.0 - self.smoothing_factor) * active_position)
                active_position -= active_position.mean()
                final_positions[t] = active_position
                
            # Formatting and final constraint compliance
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()
            
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
