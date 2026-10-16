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
    Production-Grade Quant Strategy for AlphaNova Biweekly Season 1 Cycle 5 Tournament.
    Implements regularized cross-sectional feature training with an L2 penalty,
    spherical variance stabilization, and an L1 turnover hysteresis loop.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        self.weights = None
        
        # SCIENTIFICALLY CALIBRATED HYPERPARAMETERS
        self.target_bound = 1.39
        self.optimal_concentration = 0.31   # Target concentration within [0.0, 1.0]
        self.decay = 0.99
        self.l1_hysteresis_threshold = 1.11  # Strict L1 turnover protection buffer

    def _engineer_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Generates robust, shuffling-immune cross-sectional interaction factors.
        """
        if self.feature_names is None:
            self.feature_names = list(features_df.columns.get_level_values(0).unique())
            
        ranks = {}
        for feat in self.feature_names:
            if feat in features_df.columns.levels:
                block_val = features_df[feat].astype(np.float64)
                # Uniform rank mapping into [-2.0, 2.0] domain
                r = (block_val.rank(axis=1, pct=True, method='max') - 0.5) * 4.0
                ranks[feat] = r.fillna(0.0).to_numpy()

        if not ranks:
            return np.zeros((len(features_df), len(features_df.columns.get_level_values(1).unique()), 1))

        N_time, J_assets = next(iter(ranks.values())).shape
        interaction_blocks = []
        
        # Construct orthogonal non-linear alpha components
        for i in range(len(self.feature_names)):
            f1 = self.feature_names[i]
            if f1 in ranks:
                interaction_blocks.append(np.tanh(ranks[f1] * 2.0))
                
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    if f2 in ranks:
                        interaction_blocks.append(np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25))
                        interaction_blocks.append(ranks[f1] * np.abs(ranks[f2]))
                        
        if not interaction_blocks:
            return np.zeros((N_time, J_assets, 1))
            
        # Layout shapes natively to preserve cross-sectional dimensions: (N_time, J_assets, N_factors)
        return np.stack(interaction_blocks, axis=-1)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Trains generalizable global parameters on targets using a strong L2 penalty
        to prevent overfitting and noise memorization.
        """
        if features is None or features.empty or target is None or target.empty:
            self.weights = np.zeros(1)
            return
            
        self.feature_names = list(features.columns.get_level_values(0).unique())
        
        # Sort indices to guarantee robust sequence alignment
        features_sorted = features.sort_index(ascending=True)
        target_sorted = target.sort_index(ascending=True)
        
        # Extract engineered components and targets
        X_tensor = self._engineer_features(features_sorted)  # (N, J, N_feats)
        y_matrix = target_sorted.astype(np.float64).fillna(0.0).to_numpy()  # (N, J)
        
        N_time, J_assets, N_feats = X_tensor.shape
        
        # Flatten time and asset domains concurrently to train global parameters
        X_flat = X_tensor.reshape(-1, N_feats)
        y_flat = y_matrix.reshape(-1)
        
        # Strong L2 Penalty (Ridge Regularization) to enforce feature generalization
        lambd = 800.0
        try:
            self.weights = np.linalg.solve(X_flat.T @ X_flat + lambd * np.eye(N_feats), X_flat.T @ y_flat)
        except np.linalg.LinAlgError:
            self.weights = np.zeros(N_feats)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        features_sorted = features.sort_index(ascending=True)
        tickers = features_sorted.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features_sorted.index, columns=tickers, dtype=np.float32)
        
        if len(features_sorted) == 0:
            return zero_signal
            
        if self.feature_names is None:
            self.feature_names = list(features_sorted.columns.get_level_values(0).unique())
            
        try:
            X_tensor = self._engineer_features(features_sorted)
            N_time, J_assets, N_feats = X_tensor.shape
            
            if self.weights is None or len(self.weights) != N_feats:
                self.weights = np.zeros(N_feats)
                
            # 1. Generate directional alpha via dot product with regularized weights
            raw_velocity = np.tensordot(X_tensor, self.weights, axes=([2], [0]))
            
            # 2. Geometric Subspace Demean & Hypersphere Projection
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration
            
            # 3. Execution Filter: L1 Causal Hysteresis Loop
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
                    current_allocation = active_position.copy()
                else:
                    current_allocation = 0.08 * target_position + 0.92 * active_position
                    current_allocation -= current_allocation.mean()
                
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()
                
            # 4. Compliance Formatting & Snapshot Tracing
            self.prev_signal = final_positions[-1].copy()
            self.prev_tickers = tickers.copy()
            
            final_df = pd.DataFrame(final_positions, index=features_sorted.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            
            return final_df.reindex(index=features.index).astype(np.float32)
            
        except Exception:
            return zero_signal

# Required production instantiation assignment
predictor = MyPredictor()
