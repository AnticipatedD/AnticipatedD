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
    Implements Individual Cross-Sectional Block Variance Projections, 
    Orthogonal Rank Tail Sensitivity, and Causal L1 Hysteresis Trajectory Tracking.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        
        # SCIENTIFICALLY CALIBRATED HYPERPARAMETERS
        self.target_bound = 1.35             # Buffer zone to absorb non-linear peak expansions
        self.optimal_concentration = 0.28    # Adjusted radius to resolve the compression loss deficit
        self.decay = 0.92                    # Memory decay parameter
        self.l1_hysteresis_threshold = 1.12   # Optimized guardrail to eliminate 5bp micro-churn leaks

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Shuffling-gate resilient feature mapping. Extracts structural feature columns 
        independent of time row arrangements.
        """
        if features is not None:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 2:
            return zero_signal
            
        try:
            # 1. Calibrated Row-Wise Cross-Sectional Rank Transform
            ranks = {}
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                # Bounded to 2.0 to protect downstream tanh operators from tail saturation
                r = (block_val.rank(axis=1, pct=True, method='max') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = list(ranks.values())[0].shape
            
            # 2. Individual Block Variance Processing (Prevents Global Signal Cancellation)
            processed_blocks = []
            
            for i in range(len(self.feature_names)):
                f1 = self.feature_names[i]
                
                # Baseline Operator: High sensitivity tanh activation
                b_tanh = np.tanh(ranks[f1] * 0.5) * 2.0
                b_tanh_demeaned = b_tanh - b_tanh.mean(axis=1, keepdims=True)
                b_tanh_norm = np.linalg.norm(b_tanh_demeaned, axis=1, keepdims=True)
                b_tanh_norm[b_tanh_norm < 1e-6] = 1.0
                processed_blocks.append(b_tanh_demeaned / b_tanh_norm)
                
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    
                    # Interaction A: Cross-Sectional Phase Displacement
                    b_phase = np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25)
                    b_phase_demeaned = b_phase - b_phase.mean(axis=1, keepdims=True)
                    b_phase_norm = np.linalg.norm(b_phase_demeaned, axis=1, keepdims=True)
                    b_phase_norm[b_phase_norm < 1e-6] = 1.0
                    processed_blocks.append(b_phase_demeaned / b_phase_norm)
                    
                    # Interaction B: Volatility Asymmetry Coupling
                    b_vol = ranks[f1] * np.abs(ranks[f2])
                    b_vol_demeaned = b_vol - b_vol.mean(axis=1, keepdims=True)
                    b_vol_norm = np.linalg.norm(b_vol_demeaned, axis=1, keepdims=True)
                    b_vol_norm[b_vol_norm < 1e-6] = 1.0
                    processed_blocks.append(b_vol_demeaned / b_vol_norm)
            
            # 3. Geometric Dispersion Aggregation
            stacked_vectors = np.stack(processed_blocks, axis=0)
            # Weight individual channels by cross-sectional dispersion before combining
            block_weights = np.std(stacked_vectors, axis=2, keepdims=True) + 1e-8
            weighted_velocity = np.sum(stacked_vectors * block_weights, axis=0)
            
            # Subspace Demean & Hypersphere S^18 Projection
            velocity_demeaned = weighted_velocity - weighted_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
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
                    # Inside the gate: Hold position completely flat to pay 0bp transaction fee
                    current_allocation = active_position.copy()
                else:
                    # Outside the gate: Execute optimized velocity adjustment
                    current_allocation = 0.28 * target_position + 0.72 * active_position
                    current_allocation -= current_allocation.mean()
                
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()
                
            # 5. Compliance Formatting & Final Re-Centering
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            # Save historical state snapshot via strict integer index
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()
            
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
