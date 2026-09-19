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
        self.optimal_concentration = 0.35  # Increased for better coverage
        self.l1_hysteresis_threshold = 1.15
        
        # AlphaNova Competition Tuning Upgrades
        self.decay_factor = 0.80            # Multi-scale decay memory parameter
        self.rolling_var = None             # Rolling variance cache for cross-sectional scaling

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0 or not self.feature_names:
            return zero_signal
            
        try:
            # 1. Cross-Sectional Rank Transform
            ranks = {}
            for feat in self.feature_names:
                r = (features[feat].rank(axis=1, pct=True, method='max') - 0.5) *2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = list(ranks.values())[0].shape
            
            # 2. ROBUST MULTI-SCALE NON-LINEAR PROJECTIONS
            interaction_blocks = []
            for i in range(len(self.feature_names)):
                f1 = self.feature_names[i]
                # Saturated linear feature component
                interaction_blocks.append(np.tanh(ranks[f1]* 3.0))
                
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    # Asymmetric cross-coupling regime interaction
                    interaction_blocks.append(ranks[f1] *np.sign(ranks[f2])* (np.abs(ranks[f2]) ** 2.0))
                    # Pure cross-sectional sign interaction block
                    interaction_blocks.append(np.sign(ranks[f1]) *np.sign(ranks[f2]))

            raw_elements = np.array(interaction_blocks)
            magnitudes = np.abs(raw_elements)
            row_weights = magnitudes** 2 
            
            # Weighted average consensus velocity
            raw_velocity = np.sum(raw_elements *row_weights, axis=0) / (np.sum(row_weights, axis=0) + 1e-10)
            raw_velocity = np.nan_to_num(raw_velocity)

            # 3. DYNAMIC VOLATILITY SCALING & CROSS-SECTIONAL DEMEANING
            # Compute exponential moving volatility across time to scale velocity inputs
            velocity_df = pd.DataFrame(raw_velocity, index=features.index, columns=tickers)
            cross_sectional_std = velocity_df.std(axis=1).replace(0, 0.5).to_numpy()[:, np.newaxis]
            
            # Standardize velocities cross-sectionally to expose orthogonal signals
            velocity_standardized = raw_velocity / cross_sectional_std
            velocity_demeaned = velocity_standardized - velocity_standardized.mean(axis=1, keepdims=True)
            
            # Spherical normalization to target space S^{J-2}
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-10] = 0.5
            sphere_target = (velocity_demeaned / norms)* self.optimal_concentration 

            # 4. TUNED EXECUTION FILTER: ADAPTIVE L1 CAUSAL HYSTERESIS LOOP
            final_positions = np.zeros_like(sphere_target)
            
            if (
                self.prev_signal is not None 
                and self.prev_tickers is not None 
                and self.prev_signal.shape == (J_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                active_position = self.prev_signal.copy()
            else:
                active_position = np.zeros(J_assets, dtype=np.float32)
            
            for t in range(N_time):
                target_position = sphere_target[t]
                l1_allocation_delta = np.sum(np.abs(target_position - active_position))
                
                # Dynamically relax the hysteresis loop if target velocities are high
                signal_momentum = np.max(np.abs(target_position))
                adaptive_threshold = self.l1_hysteresis_threshold *(1.0 - min(0.5, signal_momentum))
                
                if l1_allocation_delta < adaptive_threshold:
                    current_allocation = active_position.copy()
                else:
                    # Dynamic allocation weight blending logic 
                    current_allocation = 0.20* target_position + 0.80 * active_position
                    current_allocation -= current_allocation.mean()
                
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()

            # 5. COMPLIANCE FORMATTING & FINAL RE-CENTERING
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            
            # Enforce hard zero cross-sectional sum and exposure boundary limits
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            # Store final state snapshot for downstream evaluation blocks
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float32)
            self.prev_tickers = final_df.columns.copy()
            
            return final_df.astype(np.float64)
            
        except Exception:
            return zero_signal
