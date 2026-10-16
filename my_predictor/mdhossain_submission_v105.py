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
    Stateless Cross-Sectional Ranking Strategy for AlphaNova.
    Optimized for turnover drag suppression and signal city diversification.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        # Strict structural limits matching platform conditions
        self.target_bound = 1.0           
        self.optimal_concentration = 0.35  

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Extract feature identities dynamically without position dependency.
        """
        if features is not None:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0 or not self.feature_names:
            return zero_signal
            
        try:
            # 1. Compute Cross-Sectional Normalized Ranks
            # Centered around 0.0, bounding values to immune the pipeline from outliers
            ranks = {}
            for feat in self.feature_names:
                block_val = features[feat].astype(np.float64)
                # Percentile rank mapped from -1.0 to 1.0 cross-sectionally
                r = (block_val.rank(axis=1, pct=True, method='average') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = list(ranks.values())[0].shape
            
            # 2. Build Multi-Scale Structural Combinations
            # Mix linear features with residual interactions to avoid crowded baseline spaces
            signal_components = []
            
            for i, f1 in enumerate(self.feature_names):
                # Raw directional structural component
                signal_components.append(ranks[f1])
                
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    # Orthogonal non-linear mapping to steer coordinates away from standard cities
                    interaction = np.sin(ranks[f1] * np.pi * 0.5) * np.cos(ranks[f2] * np.pi * 0.5)
                    signal_components.append(interaction)

            # Generate aggregate raw forecast direction
            combined_forecast = np.mean(signal_components, axis=0)

            # 3. Time-Series Smoothing Filter (Turnover Mitigation Lever)
            # Rebalancing hourly costs ~6% volatility drag. We smooth the signal over the 
            # available batch history to penalize high-frequency churn.
            smoothed_forecast = np.zeros_like(combined_forecast)
            
            # Use a safe decay parameter (e.g., 0.85) to hold structural allocation trends longer
            alpha = 0.15
            running_vector = combined_forecast[0, :].copy()
            
            for t in range(N_time):
                running_vector = alpha * combined_forecast[t, :] + (1.0 - alpha) * running_vector
                smoothed_forecast[t, :] = running_vector

            # 4. Cross-Sectional Demean & Spherical Standardization
            # Guarantees sum(P_j) == 0 perfectly at every single timestamp
            final_signals = smoothed_forecast - smoothed_forecast.mean(axis=1, keepdims=True)
            
            # Scale coordinates into AlphaNova's desired concentration space
            row_norms = np.linalg.norm(final_signals, axis=1, keepdims=True)
            row_norms[row_norms < 1e-8] = 1.0
            
            # Project onto the sphere surface defined by target constraints
            normalized_output = (final_signals / row_norms) * self.optimal_concentration

            # 5. Output Packaging & Formatting Cleanliness
            final_df = pd.DataFrame(normalized_output, index=features.index, columns=tickers)
            
            # Final compliance pass: clear precision residue, clip, and re-center
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
