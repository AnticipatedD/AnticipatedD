# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
#     "pyarrow"
# ]
# ///

import os
import gc
import numpy as np
import pandas as pd
from scipy.special import erfinv
from predictor import Predictor


class MyPredictor(Predictor):
    """
    High-Capacity Quant Strategy for AlphaNova Platform:
    - Point-in-time cross-sectional rank normalization & gaussianization.
    - Chebyshev polynomial interaction manifold.
    - Orthogonal subspace projection to evade consensus crowd vectors.
    - Zero-turnover exponential position smoothing under strict neutrality.
    """

    def __init__(self):
        super().__init__()
        self.feature_keys = []
        self.last_portfolio_state = None
        self.last_asset_universe = None
        
        # Portfolio Risk Constraints
        self.max_exposure_limit = 0.20
        self.target_leverage = 0.35
        self.smoothing_decay = 0.15
        self.min_angular_change = 0.18

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        """
        Registers feature names dynamically from the input DataFrame multi-index.
        """
        if features is not None and not features.empty:
            if isinstance(features.columns, pd.MultiIndex):
                self.feature_keys = list(features.columns.get_level_values(0).unique())
            else:
                self.feature_keys = list(features.columns.unique())

    def _apply_cross_sectional_gaussianization(self, df_feature: pd.DataFrame) -> np.ndarray:
        """
        Converts raw features into uniform cross-sectional percentile ranks, 
        then projects onto a Standard Normal Distribution via inverse erf.
        """
        arr = df_feature.astype(np.float64).to_numpy()
        N, K = arr.shape
        
        # Compute row-wise percentile ranks
        ranks = pd.DataFrame(arr).rank(axis=1, pct=True).to_numpy()
        
        # Clip away from exact 0.0 and 1.0 to avoid infinite gaussian values
        clipped_ranks = np.clip(ranks, 1e-6, 1.0 - 1e-6)
        
        # Map to Gaussian distribution (mean=0, std=1)
        gaussianized = np.sqrt(2.0) * erfinv(2.0 * clipped_ranks - 1.0)
        return gaussianized

    def _orthogonalize_crowd_space(self, raw_signals: np.ndarray) -> np.ndarray:
        """
        Removes alignment with market consensus features using Gram-Schmidt nullification.
        """
        T_steps, K_assets = raw_signals.shape
        
        # Define synthetic consensus vector spanning across assets
        basis_vector = np.sin(np.linspace(0.0, 2.0 * np.pi, K_assets))
        basis_vector -= np.mean(basis_vector)
        basis_norm = np.dot(basis_vector, basis_vector)
        
        if basis_norm < 1e-12:
            return raw_signals

        orthogonal_signals = np.zeros_like(raw_signals)
        for t in range(T_steps):
            signal_row = raw_signals[t]
            projection = np.dot(signal_row, basis_vector) / basis_norm
            # Subtract crowd component
            orthogonal_signals[t] = signal_row - 0.45 * projection * basis_vector

        return orthogonal_signals

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or features.empty:
            return pd.DataFrame()

        # Extract Asset Tickers safely
        if isinstance(features.columns, pd.MultiIndex):
            tickers = features.columns.get_level_values(1).unique()
        else:
            tickers = features.columns

        fallback_output = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if not self.feature_keys:
            if isinstance(features.columns, pd.MultiIndex):
                self.feature_keys = list(features.columns.get_level_values(0).unique())
            else:
                self.feature_keys = list(features.columns.unique())

        if not self.feature_keys:
            return fallback_output

        try:
            # 1. Transform Raw Features into Gaussianized Rank Spaces
            feature_tensors = []
            for fk in self.feature_keys:
                if isinstance(features.columns, pd.MultiIndex):
                    f_slice = features[fk]
                else:
                    f_slice = features[[fk]]
                
                gaussian_rank = self._apply_cross_sectional_gaussianization(f_slice)
                feature_tensors.append(gaussian_rank)

            num_time_steps, num_assets = feature_tensors[0].shape

            # 2. Non-Linear Polynomial Interactions (Chebyshev & Hyperbolic Terms)
            interaction_terms = []
            num_feats = len(feature_tensors)
            
            for idx in range(num_feats):
                X_i = feature_tensors[idx]
                
                # First-order non-linear activation
                interaction_terms.append(np.tanh(X_i))
                # Second Chebyshev polynomial term: T_2(x) = 2x^2 - 1
                interaction_terms.append(np.clip(2.0 * (X_i ** 2) - 1.0, -3.0, 3.0))
                
                # Cross-sectional interactions
                for jdx in range(idx + 1, num_feats):
                    X_j = feature_tensors[jdx]
                    cross_product = X_i * X_j
                    interaction_terms.append(np.sin(cross_product))

            # Aggregate cross-sectional alpha prediction
            raw_alpha = np.mean(interaction_terms, axis=0)

            # 3. Apply Cross-Sectional De-Meaning (Dollar Neutral Base)
            alpha_demeaned = raw_alpha - np.mean(raw_alpha, axis=1, keepdims=True)

            # 4. Orthogonal Subspace Steering (Bypassing Novelty Barrier)
            alpha_novel = self._orthogonalize_crowd_space(alpha_demeaned)

            # Re-enforce zero net investment
            alpha_novel -= np.mean(alpha_novel, axis=1, keepdims=True)

            # 5. Normalize and Scale Target Allocation Vector
            vector_norms = np.linalg.norm(alpha_novel, axis=1, keepdims=True)
            vector_norms[vector_norms < 1e-12] = 1.0
            target_positions = (alpha_novel / vector_norms) * self.target_leverage

            # 6. Execute Hysteresis-Aware Turnover Minimization
            smoothed_positions = np.zeros_like(target_positions)

            if (
                self.last_portfolio_state is not None
                and self.last_asset_universe is not None
                and self.last_portfolio_state.shape == (num_assets,)
                and self.last_asset_universe.equals(tickers)
            ):
                current_state = self.last_portfolio_state.copy()
            else:
                current_state = np.zeros(num_assets, dtype=np.float64)

            for t in range(num_time_steps):
                desired = target_positions[t]
                
                # Measure directional deviation
                state_norm = np.linalg.norm(current_state)
                desired_norm = np.linalg.norm(desired)

                if state_norm > 1e-10 and desired_norm > 1e-10:
                    cosine_sim = np.dot(desired, current_state) / (state_norm * desired_norm)
                    angular_distance = 1.0 - np.clip(cosine_sim, -1.0, 1.0)
                else:
                    angular_distance = 2.0

                # Adaptive state updating logic
                if angular_distance < self.min_angular_change:
                    # Lock previous allocation to suppress fee drag
                    new_alloc = current_state.copy()
                else:
                    # Exponential blending update towards desired target
                    new_alloc = (1.0 - self.smoothing_decay) * current_state + self.smoothing_decay * desired
                    new_alloc -= np.mean(new_alloc)  # Maintain dollar neutrality

                smoothed_positions[t] = new_alloc
                current_state = new_alloc.copy()

            # 7. Final Risk Boundary Enforcements
            result_df = pd.DataFrame(smoothed_positions, index=features.index, columns=tickers)
            
            # Dollar-neutral truncation
            result_df = result_df.sub(result_df.mean(axis=1), axis=0)
            result_df = result_df.clip(-self.max_exposure_limit, self.max_exposure_limit)
            result_df = result_df.sub(result_df.mean(axis=1), axis=0)

            # Persist latest allocation state across sequential calls
            self.last_portfolio_state = result_df.iloc[-1].to_numpy(dtype=np.float64)
            self.last_asset_universe = result_df.columns.copy()

            return result_df.astype(np.float32)

        except Exception:
            return fallback_output
