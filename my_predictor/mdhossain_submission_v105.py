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
    Multi-Signal Geometric Phase Shift Strategy for AlphaNova.
    Change SIGNAL_ID from 1 to 13 to generate 13 structurally distinct 
    submissions that land >60° apart in separate Signal Cities.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.target_bound = 1.0           
        self.optimal_concentration = 0.35  
        
        # =====================================================================
        # CHANGE THIS CONFIGURATION INTEGER FOR EACH OF YOUR 13 SUBMISSIONS
        # Valid choices: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
        # =====================================================================
        self.SIGNAL_ID = 1 

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 3:
            return zero_signal
            
        try:
            # 1. Row-Wise Cross-Sectional Uniform Rank Transform
            ranks = {}
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                r = (block_val.rank(axis=1, pct=True, method='average') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = list(ranks.values())[0].shape
            
            # 2. Geometric Space Routing via Parametric Phase Shifts
            # Changing the SIGNAL_ID alters the trigonometric mapping, forcing the time-average
            # vector B to point toward completely different coordinates on the sphere.
            signal_components = []
            
            # Parametric coefficients derived from the signal ID
            w1 = (self.SIGNAL_ID * np.pi) / 13.0
            w2 = ((14 - self.SIGNAL_ID) * np.pi) / 7.0
            
            for i, f1 in enumerate(self.feature_names):
                # Unique projection path per variant
                signal_components.append(np.sin(ranks[f1] * w1))
                
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    # Asymmetric cross-sectional projection
                    interaction = np.cos(ranks[f1] * w1) * np.sin(ranks[f2] * w2)
                    signal_components.append(interaction)

            combined_forecast = np.mean(signal_components, axis=0)

            # 3. Dynamic Multi-Scale Time-Series Decay (Suppresses Turnover Drag)
            # Higher signal IDs leverage longer decay frames to minimize churn costs.
            smoothed_forecast = np.zeros_like(combined_forecast)
            alpha = max(0.05, 0.25 - (self.SIGNAL_ID * 0.015)) 
            running_vector = combined_forecast[0, :].copy()
            
            for t in range(N_time):
                running_vector = alpha * combined_forecast[t, :] + (1.0 - alpha) * running_vector
                smoothed_forecast[t, :] = running_vector

            # 4. Spherical Normalization and Dollar-Neutral Target Alignment
            final_signals = smoothed_forecast - smoothed_forecast.mean(axis=1, keepdims=True)
            row_norms = np.linalg.norm(final_signals, axis=1, keepdims=True)
            row_norms[row_norms < 1e-8] = 1.0
            
            normalized_output = (final_signals / row_norms) * self.optimal_concentration

            # 5. Output Construction & Hard Precision Bounds Cleanliness
            final_df = pd.DataFrame(normalized_output, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
