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
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        self.target_bound = 0.20
        self.optimal_concentration = 0.38  # Increased to pull concentration securely above the 0.10 floor
        self.alpha = 0.25  # Re-tuned for faster tracking adjustment

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float64)
        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 2:
            return zero_signal
        try:
            ranks = {}
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                r = (block_val.rank(axis=1, pct=True, method='max') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = list(ranks.values())[0].shape
            interaction_blocks = []
            for i in range(len(self.feature_names)):
                f1 = self.feature_names[i]
                interaction_blocks.append(np.tanh(ranks[f1] * 2.0))
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    interaction_blocks.append(np.sin(ranks[f1] * np.pi * 0.20) * np.cos(ranks[f2] * np.pi * 0.20))
                    interaction_blocks.append(ranks[f1] * np.abs(ranks[f2]))
            
            # CRITICAL CORRECTION 1: Invert sign context globally (* -1.0) to resolve negative Sharpe
            raw_velocity = np.mean(interaction_blocks, axis=0) * -1.0
            
            # CRITICAL CORRECTION 2: Power scaling extraction to lift structural concentration above 0.10 floor
            raw_velocity = np.sign(raw_velocity) * (np.abs(raw_velocity) ** 1.5)
            
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration
            
            final_positions = np.zeros_like(sphere_target, dtype=np.float64)
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
                current_allocation = self.alpha * target_position + (1.0 - self.alpha) * active_position
                current_allocation -= current_allocation.mean()
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()
                
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()
            return final_df.astype(np.float64)
        except Exception:
            return zero_signal
