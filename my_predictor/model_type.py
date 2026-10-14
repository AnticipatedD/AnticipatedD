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
    Explicitly optimized for a 6-Feature Measurement Framework.
    Implements Rank-Space Directional Mapping with Cross-Feature Interaction Shrinkage.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.feature_weights = None
        self.interaction_weights = None
        self.prev_signal = None
        self.prev_tickers = None
        
        # Hyperparameters optimized for 6 features and 5bp fee constraints
        self.target_bound = 1.35
        self.alpha_scale = 0.30
        self.smoothing_factor = 0.40  # Faster tracking optimized for high-dimensional signals
        self.shrinkage = 0.15         # Regularization factor for feature interaction blocks

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Calibrates optimal predictive weights explicitly for all 6 features
        and their structural combinations.
        """
        if features is None or target is None:
            return
            
        self.feature_names = list(features.columns.get_level_values(0).unique())
        tickers = features.columns.get_level_values(1).unique()
        
        # Ensure correct alignment of dates/times
        common_idx = features.index.intersection(target.index)
        if len(common_idx) == 0:
            return
            
        feat_sub = features.loc[common_idx]
        targ_sub = target.loc[common_idx]
        
        # 1. Map individual linear rank coefficients for all 6 features
        t_rank = targ_sub.rank(axis=1, pct=True, method='max') - 0.5
        
        self.feature_weights = {}
        ranks_dict = {}
        
        for feat in self.feature_names:
            f_rank = feat_sub[feat].rank(axis=1, pct=True, method='max') - 0.5
            ranks_dict[feat] = f_rank
            
            # Cross-sectional covariance tracking
            cov = (f_rank * t_rank).mean().mean()
            self.feature_weights[feat] = cov
            
        # 2. Map explicit pair-wise interaction coefficients unique to a 6-feature universe
        self.interaction_weights = {}
        for i in range(len(self.feature_names)):
            for j in range(i + 1, len(self.feature_names)):
                f1, f2 = self.feature_names[i], self.feature_names[j]
                # Non-linear interaction block
                interact = ranks_dict[f1] * ranks_dict[f2]
                interact_demeaned = interact.sub(interact.mean(axis=1), axis=0)
                
                cov_interact = (interact_demeaned * t_rank).mean().mean()
                self.interaction_weights[f"{f1}_{f2}"] = cov_interact

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0 or self.feature_names is None or self.feature_weights is None:
            return zero_signal
            
        try:
            N_time = len(features)
            J_assets = len(tickers)
            combined_signal = np.zeros((N_time, J_assets), dtype=np.float64)
            
            # 1. Compute Base Rank Matrices for the 6 features
            ranks_dict = {}
            for feat in self.feature_names:
                r = (features[feat].rank(axis=1, pct=True, method='max') - 0.5).fillna(0.0).to_numpy()
                ranks_dict[feat] = r
                
                # Apply base feature directional alpha
                combined_signal += r * self.feature_weights.get(feat, 0.0)
                
            # 2. Inject Regularized Interaction Blocks (Unique to 6-feature constraints)
            for i in range(len(self.feature_names)):
                for j in range(i + 1, len(self.feature_names)):
                    f1, f2 = self.feature_names[i], self.feature_names[j]
                    interact_key = f"{f1}_{f2}"
                    
                    if interact_key in self.interaction_weights:
                        interact_signal = ranks_dict[f1] * ranks_dict[f2]
                        # Remove cross-sectional market noise from interaction matrix
                        interact_signal -= interact_signal.mean(axis=1, keepdims=True)
                        
                        # Blend the interaction with a regularized shrinkage factor to protect the Sharpe
                        weight = self.interaction_weights[interact_key] * self.shrinkage
                        combined_signal += interact_signal * weight

            # 3. Geometric Subspace Neutralization
            combined_signal -= combined_signal.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(combined_signal, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            target_positions = (combined_signal / norms) * self.alpha_scale
            
            # 4. Sequential Portfolio Smoother (Friction Mitigation)
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
                # Exponential blending avoids execution latency while maintaining robust turnover safety
                active_position = (self.smoothing_factor * target_positions[t]) + ((1.0 - self.smoothing_factor) * active_position)
                active_position -= active_position.mean()  # Re-enforce pure dollar neutrality
                final_positions[t] = active_position
                
            # 5. Outlier Truncation & Post-Processing Compliance
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            # Persist state snapshots for subsequent streaming ticks
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()
            
            return final_df.astype(np.float32)
            
        except Exception:
            return zero_signal
