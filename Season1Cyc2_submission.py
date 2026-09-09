# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
#     "pyarrow"
# ]
# ///

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Base Class Handling
# -----------------------------------------------------------------------------
try:
    from predictor import Predictor
except ImportError:
    # Fallback mock class to ensure standalone file validity for testing
    class Predictor:
        def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
            pass

        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# MyPredictor Implementation
# -----------------------------------------------------------------------------
class MyPredictor(Predictor):
    """
    AnticipatedD_final_submission.py: Finalized AlphaNova Biweekly Competition Season 1 Cycle 2 Submission (V4_Optimized Evolution). 
    This architecturally hardened predictor implements a cross-sectional alpha logic optimized for Sharpe maximization and signal uniqueness. It processes  
    input features in (T, J) shape (Timestamps by Assets) and generates  portfolio weights through a ridge-regularized pipeline. 
    Key Architectural Components: 
    - Feature Logic: V4_Optimized (Cross-sectional ranks, Winsorized Z-scores, and explicit Feature.1 * Feature.2 interaction terms).
    - Model: Ridge Regression (lambda=35.0) with a non-penalized intercept.
    - Signal Processing: EWMA persistence on raw signals (not weights) to manage turnover, followed by cross-sectional de-meaning.
    - Normalization: Dual-stage scaling (RMS then L1) where L1 serves as the final binding constraint on gross exposure.
    """
    def __init__(self, alpha: float = 0.1, epsilon: float = 1e-9):
        """ 
        Initializes the trading architecture state. 

        Args: 
            alpha: Smoothing factor for EWMA signal persistence (0 < alpha < 1). 
            epsilon: Numerical stability guard for zero-RMS and division scenarios.
        """
        self.weights: np.ndarray = None
        self.feature_names: list = []
        self.alpha: float = alpha
        self.epsilon: float = epsilon
        self.prev_signal: pd.DataFrame = None  # Persistent signal state 

    def _engineer_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """ 
        Synthesizes the V4_Optimized feature set using vectorized cross-sectional logic. 

        This method transforms raw MultiIndex features into:
            1. Normalized Cross-Sectional Ranks [-0.5, 0.5].
            2. Winsorized Z-Scores (Clipped at +/- 2.5).
            3. Explicit Interaction Term: Feature.1_wz * Feature.2_rk.
        """
        base_features = df.columns.get_level_values(0).unique() 
        engineered_dict = {}

        for feat in base_features: 
            feat_df = df[feat]
            
            # 1. Cross-Sectional Ranks
            engineered_dict[f"{feat}_rk"] = feat_df.rank(axis=1, pct=True) - 0.5
            
            # 2. Winsorized Z-Scores
            mean_vals = feat_df.mean(axis=1) 
            # Use epsilon guard for zero standard deviation replacement to maintain signal continuity 
            std_vals = feat_df.std(axis=1).replace(0.0, self.epsilon) 
            z_score = feat_df.sub(mean_vals, axis=0).div(std_vals, axis=0) 
            engineered_dict[f"{feat}_wz"] = z_score.clip(-2.5, 2.5).fillna(0.0)  
            
        # 3. V4_Optimized: Explicit Interaction between Feature.1 and Feature.2 
        # Logic targets specific factor-interaction patterns identified in source context 
        if "Feature.1" in base_features and "Feature.2" in base_features:
            engineered_dict["opt_inter"] = (
                engineered_dict["Feature.1_wz"] * engineered_dict["Feature.2_rk"]
            )

        # Flatten cross-sectional DataFrames into long-format (timestamp, ticker)
        long_dfs = []
        for feat_name, feat_df in engineered_dict.items():
            long_dfs.append(feat_df.stack(future_stack=True).rename(feat_name))
        
        engineered_df = pd.concat(long_dfs, axis=1).fillna(0.0)

        if is_training: 
            self.feature_names = engineered_df.columns.tolist() 
        return engineered_df[self.feature_names]

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """ 
        Trains the predictor via Ridge Regression. 
        Regularization (lambda=35.0) is applied to all feature coefficients,  
        while the bias (intercept) term remains unpenalized to ensure the  
        model can capture mean return levels without shrinkage.
        """
        X_df = self._engineer_features(features, is_training=True) 
        y_series = target.stack(future_stack=True).loc[X_df.index].fillna(0.0) 
        
        X = X_df.to_numpy() 
        Y = y_series.to_numpy()    
        
        # Construct Design Matrix with Bias
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # Ridge Parameters 
        lambd = 35.0 
        I = np.eye(X_bias.shape[1])
        I[0, 0] = 0.0  # Zero-out the penalty for the intercept 
        
        try:
            # Normal Equations: (X'X + lambda*I)w = X'y 
            self.weights = np.linalg.solve(
                X_bias.T @ X_bias + lambd * I, 
                X_bias.T @ Y
            )
        except np.linalg.LinAlgError:
            # Fallback to zero weights on singularity
            self.weights = np.zeros(X_bias.shape[1])

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generates portfolio weights with cross-sectional neutrality and persistence.
        
        Signal Processing Sequence:
        1. Projection: Generate raw alpha signals via weights.
        2. EWMA Persistence: Signal-level smoothing for turnover control.
        3. Neutrality: Cross-sectional de-meaning (Zero-Investment constraint).
        4. Dual Normalization: RMS scaling followed by L1 scaling.
           Note: RMS provides relative signal intensity, while L1 serves as 
           the final binding constraint on gross exposure (Target L1 = 1.0).
        """
        X_df = self._engineer_features(features, is_training=False)
        X = X_df.to_numpy()
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # 1. Raw Prediction Generation
        if self.weights is None:
            raw_preds = np.zeros(X.shape[0])
        else:
            raw_preds = X_bias @ self.weights
            
        pred_series = pd.Series(raw_preds, index=X_df.index)
        preds_df = pred_series.unstack(level=1).fillna(0.0)
        
        # 2. EWMA Persistence (Turnover Management)
        # We apply EWMA to signals before normalization to preserve the relative 
        # conviction of the underlying alpha before constraints are applied.
        if self.prev_signal is not None:
            # Align previous signal with current asset universe (handles ticker changes)
            prev_aligned = self.prev_signal.reindex(columns=preds_df.columns, fill_value=0.0)
            preds_df = (self.alpha * prev_aligned) + ((1 - self.alpha) * preds_df)
        
        self.prev_signal = preds_df.copy()

        # 3. Cross-sectional Neutrality: Row-wise de-meaning
        preds_df = preds_df.sub(preds_df.mean(axis=1), axis=0)

        # 4. Dual Normalization Scaling
        # Stage A: RMS Scaling (Standardizing signal volatility)
        rms = np.sqrt((preds_df**2).mean(axis=1))
        preds_rms = preds_df.div(rms.replace(0.0, self.epsilon), axis=0)
        
        # Stage B: L1 Scaling (The final binding constraint on Gross Exposure)
        l1_norm = preds_rms.abs().sum(axis=1)
        final_weights = preds_rms.div(l1_norm.replace(0.0, self.epsilon), axis=0)
        
        # Ensure parity with the required ticker universe
        required_tickers = features.columns.get_level_values(1).unique()
        final_weights = final_weights.reindex(columns=required_tickers, fill_value=0.0)
        
        # Final safety de-mean to satisfy competition sum(W)=0 constraint
        final_weights = final_weights.sub(final_weights.mean(axis=1), axis=0)
        
        return final_weights

    def novelty_check(self, candidate_vectors: np.ndarray, lat_lon_to_unit: bool = False) -> np.ndarray:
        """
        Geographic novelty-check utility to detect signal redundancy.
        
        Computes the nearest neighbor via angular distance (1 - dot product) 
        against global signal city reference data.
        
        Args:
            candidate_vectors: NumPy array of coordinates (N, dim).
            lat_lon_to_unit: If True, treats candidates as [lat, lon] in degrees 
                             and converts them to unit vectors (x, y, z).
                               
        Returns:
            NumPy array of distances to the nearest internal city neighbors.
        """
        import pyarrow.parquet as pq  # Local import for decoupling
        
        try:
            # Load internal reference data
            city_data = pd.read_parquet("data/signal_cities.parquet")
            reference_vectors = city_data[['x', 'y', 'z']].to_numpy()
        except Exception:
            # Operational fallback
            return np.zeros(len(candidate_vectors))

        processed_candidates = candidate_vectors
        if lat_lon_to_unit:
            # Convert degrees to radians
            rad = np.radians(candidate_vectors)
            lat, lon = rad[:, 0], rad[:, 1]
            
            # Spherical to Cartesian unit vector conversion
            x = np.cos(lat) * np.cos(lon)
            y = np.cos(lat) * np.sin(lon)
            z = np.sin(lat)
            processed_candidates = np.stack([x, y, z], axis=1)

        # Distance metric: 1.0 - Max Similarity 
def calculate_angular_distance(processed_candidates, reference_vectors):
    """
    Computes the nearest neighbor distance metric (1.0 - Max Similarity)
    using a vectorized dot product execution.
    """
    dot_products = processed_candidates @ reference_vectors.T
    max_similarity = np.max(dot_products, axis=1)
    
    return 1.0 - max_similarity
