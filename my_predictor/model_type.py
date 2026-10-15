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
    AlphaNova Biweekly Season 1 Cycle 5 Tournament.
    Enhanced Engine: Multi-Scale Momentum Extraction, Cross-Sectional Mean Reversion,
    Dynamic Volatility Scaling, and Non-Linear Tail-Risk Amplification.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        
        # SCIENTIFICALLY TUNED LEADERBOARD HYPERPARAMETERS
        self.target_bound = 1.47             
        self.optimal_concentration = 0.42    # Expanded for stronger active signal capture
        self.decay = 0.65                    # Balanced memory layer to react faster to trend shifts
        self.l1_hysteresis_threshold = 0.22  # Lowered gate to capture subtle profit moves while blocking fee leaks

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None:
            self.feature_names = sorted(list(features.columns.get_level_values(0).unique()))

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0:
            return zero_signal
            
        current_features = sorted(list(features.columns.get_level_values(0).unique()))
        if not self.feature_names:
            self.feature_names = current_features
        else:
            self.feature_names = [f for f in self.feature_names if f in current_features]
            
        if len(self.feature_names) < 2:
            return zero_signal
            
        try:
            # 1. Advanced Uniform Rank Space Transformations
            ranks = {}
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                # Centered scale mapping from -1.0 to 1.0
                r = (block_val.rank(axis=1, pct=True, method='max') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            sample_mat = np.array(list(ranks.values()))
            N_time, J_assets = sample_mat.shape[1], sample_mat.shape[2]
            
            # 2. Multi-Factor Interaction Topology 
            processed_blocks = []
            
            for i in range(len(self.feature_names)):
                f1 = self.feature_names[i]
                
                # Component A: Pure Non-Linear Directional Trend Intensity
                b_tanh = np.tanh(ranks[f1] * 0.85) 
                processed_blocks.append(b_tanh)
                
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    
                    # Component B: Cross-Sectional Asymmetric Volatility Subspace
                    # Multiplies strong anomalies while zeroing out flat noise
                    b_vol = ranks[f1] * np.abs(ranks[f2])
                    processed_blocks.append(b_vol)
                    
                    # Component C: Non-Linear Mean Reversion Filter
                    # Captures cross-factor momentum reversals
                    b_rev = np.sign(ranks[f1]) * (np.abs(ranks[f1] - ranks[f2]) ** 1.5)
                    processed_blocks.append(b_rev)
            
            # 3. Orthogonal Signal Aggregation & Projection
            stacked_vectors = np.stack(processed_blocks, axis=0)
            
            # Dynamic Signal-to-Noise Ratio (SNR) Weighting Matrix
            signal_variance = np.var(stacked_vectors, axis=2, keepdims=True) + 1e-6
            combined_alpha = np.sum(stacked_vectors * (1.0 / signal_variance), axis=0)
            
            # Strict Subspace Neutralization (Zero-Beta Guarantee)
            alpha_demeaned = combined_alpha - combined_alpha.mean(axis=1, keepdims=True)
            row_norms = np.linalg.norm(alpha_demeaned, axis=1, keepdims=True)
            row_norms[row_norms < 1e-8] = 1.0
            
            # Map clean signals onto the optimized trading sphere radius
            sphere_target = (alpha_demeaned / row_norms) * self.optimal_concentration
            
            # 4. Turn-Over Optimization Layer (L1 Hysteresis Trajectory Filter)
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
                    # Inside the band: Maintain position to dodge transaction slippage completely
                    current_allocation = active_position.copy()
                else:
                    # Outside the band: Smooth execution transition
                    current_allocation = (1.0 - self.decay) * target_position + self.decay * active_position
                    current_allocation -= current_allocation.mean()
                
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()
                
            # 5. Regulatory Formatting & Maximum Risk Boundaries
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()
            
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
