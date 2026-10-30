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
    Advanced Cross-Sectional Alpha Engine for AlphaNova.
    Equipped with a targeted temporal rolling risk scalar, adaptive horizon windows,
    and a continuous proportional asymmetric turnover framework to prevent lockups.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_weights = None
        self.prev_columns = None
        
        # Risk & Compliance constraints
        self.allocation_cap = 0.20
        self.scale_factor = 0.28
        
        # Risk scaling properties
        self.vol_floor = 0.01
        self.evaluation_counter = 0

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        """Extract baseline feature definitions if present."""
        if features is not None and not features.empty:
            if isinstance(features.columns, pd.MultiIndex):
                self.feature_names = list(features.columns.get_level_values(0).unique())
            else:
                self.feature_names = list(features.columns)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generates cross-sectional signals natively designed for wide-frame data representation."""
        if isinstance(features.columns, pd.MultiIndex):
            assets = features.columns.get_level_values(1).unique()
        else:
            assets = features.columns
            
        fallback_df = pd.DataFrame(0.0, index=features.index, columns=assets, dtype=np.float32)
        
        if features is None or features.empty or not self.feature_names or len(self.feature_names) < 2:
            return fallback_df
            
        try:
            n_timestamps, n_assets = len(features.index), len(assets)
            self.evaluation_counter += n_timestamps
            
            # --- ADAPTIVE HORIZON WINDOW DETERMINATION ---
            adaptive_window = int(np.clip(10 + (self.evaluation_counter // 20), 5, 40))
            
            # 1. Transform Raw Cross-Sections to Uniform Quantile Vectors [-1, 1]
            transformed_factors = []
            for factor_name in self.feature_names:
                raw_slice = features[factor_name].astype(np.float64)
                ranked_slice = raw_slice.rank(axis=1, pct=True, method='average')
                normalized_slice = (ranked_slice - 0.5) * 2.0
                transformed_factors.append(normalized_slice.fillna(0.0).to_numpy())
                
            # 2. Sequential Orthogonal Component Routing
            orthogonal_blocks = []
            for idx, primary_matrix in enumerate(transformed_factors):
                orthogonal_blocks.append(np.tanh(primary_matrix * 1.5))
                if idx < len(transformed_factors) - 1:
                    secondary_matrix = transformed_factors[idx + 1]
                    residual_mix = np.sign(primary_matrix) * np.log1p(np.abs(primary_matrix * secondary_matrix))
                    orthogonal_blocks.append(residual_mix)

            blended_velocity = np.mean(orthogonal_blocks, axis=0)
            
            # 3. Cross-Sectional Space Centering & Volatility Scaling
            centered_velocity = blended_velocity - blended_velocity.mean(axis=1, keepdims=True)
            row_norms = np.linalg.norm(centered_velocity, axis=1, keepdims=True)
            row_norms = np.where(row_norms < 1e-9, 1.0, row_norms)
            
            ideal_allocations = (centered_velocity / row_norms) * self.scale_factor
            
            # --- TARGETED FACTOR VARIANCE REGIME SCALING ---
            targeted_features = [f for f in self.feature_names if 'qual' in f.lower() or 'vol' in f.lower()]
            if not targeted_features:
                targeted_features = self.feature_names[:2]
                
            raw_matrix = features[targeted_features].fillna(0.0).to_numpy()
            time_series_variance = np.var(raw_matrix, axis=1)
            
            rolling_variance = pd.Series(time_series_variance).rolling(
                window=adaptive_window, min_periods=1
            ).mean().to_numpy()
            
            median_vol = np.median(rolling_variance) if len(rolling_variance) > 0 else 1.0
            median_vol = max(median_vol, self.vol_floor)
            risk_scalars = np.clip(median_vol / (rolling_variance + 1e-8), 0.30, 1.0)
            
            ideal_allocations = ideal_allocations * risk_scalars[:, np.newaxis]
            
            # 4. Continuous Asymmetric Update Loop
            stabilized_positions = np.zeros_like(ideal_allocations)
            
            if (
                self.prev_weights is not None 
                and self.prev_columns is not None 
                and self.prev_weights.shape == (n_assets,)
                and self.prev_columns.equals(assets)
            ):
                running_weights = self.prev_weights.copy()
            else:
                running_weights = np.zeros(n_assets, dtype=np.float64)
                
            for t_idx in range(n_timestamps):
                target_weights = ideal_allocations[t_idx]
                
                # Identify long-to-short or short-to-long side flips
                sign_flip_mask = (np.sign(target_weights) != np.sign(running_weights)) & (running_weights != 0.0)
                
                # CONTINUOUS PENALIZATION: Apply a slower tracking speed (higher friction)
                # exclusively to assets flipping signs, while updating standard allocations normally.
                update_speed = np.where(sign_flip_mask, 0.08, 0.35)
                
                current_weights = (update_speed * target_weights) + ((1.0 - update_speed) * running_weights)
                current_weights -= current_weights.mean()
                
                stabilized_positions[t_idx] = current_weights
                running_weights = current_weights.copy()
                
            # 5. Rigorous Compliance Cleaning & Absolute Float Balancing
            output_df = pd.DataFrame(stabilized_positions, index=features.index, columns=assets)
            
            output_df = output_df.sub(output_df.mean(axis=1), axis=0)
            output_df = output_df.clip(-self.allocation_cap, self.allocation_cap)
            output_df = output_df.sub(output_df.mean(axis=1), axis=0)
            
            self.prev_weights = output_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_columns = output_df.columns.copy()
            
            return output_df.astype(np.float32)
            
        except Exception:
            return fallback_df
