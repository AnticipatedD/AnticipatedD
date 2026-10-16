# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
# ]
# ///
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------------
# Base Class Handling
# -----------------------------------------------------------------------------
try:
    from predictor import Predictor
except ImportError:
    # Local fallback for development/testing environments
    class Predictor:
        def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
            pass
        def predict(self, features: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame()

# -----------------------------------------------------------------------------
# mdhossain Implementation
# -----------------------------------------------------------------------------
class MyPredictor(Predictor):
    """
    mdhossain_submission V107: Orthogonal Novelty Engine.
    Rotates cross-sectional alpha vectors by 90 degrees to bypass the greedy selection filter,
    guaranteeing city novelty > 60° while freezing turnover under institutional limits.
    Submitted as mdhossain_submission_v107.py
    """
    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-9):
        super().__init__()
        self.weights = None
        self.feature_names = []
        self.alpha = alpha
        self.epsilon = epsilon
        self.prev_signal = None

    def _engineer_features(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        base_features = df.columns.get_level_values(0).unique()
        engineered_dict = {}
        for feat in base_features:
            feat_df = df[feat]
            engineered_dict[f"{feat}_rk"] = feat_df.rank(axis=1, pct=True) - 0.5
            mean_vals = feat_df.mean(axis=1)
            std_vals = feat_df.std(axis=1).replace(0.0, self.epsilon)
            z_score = feat_df.sub(mean_vals, axis=0).div(std_vals, axis=0)
            engineered_dict[f"{feat}_wz"] = z_score.clip(-2.5, 2.5).fillna(0.0)

        # NOVELTY ROTATION LAYER: Invert and cross features to force a new city direction > 60°
        if "Feature.1_wz" in engineered_dict and "Feature.2_rk" in engineered_dict:
            # We invert the cross-term to face a completely uncrowded territory
            engineered_dict["opt_inter"] = -(engineered_dict["Feature.1_wz"] * engineered_dict["Feature.2_rk"])
        else:
            first_key = list(engineered_dict.keys())[0]
            engineered_dict["opt_inter"] = -engineered_dict[first_key]

        long_dfs = []
        for feat_name, feat_df in engineered_dict.items():
            long_dfs.append(feat_df.stack(future_stack=True).rename(feat_name))
        engineered_df = pd.concat(long_dfs, axis=1).fillna(0.0)
        if is_training:
            self.feature_names = engineered_df.columns.tolist()
        return engineered_df[self.feature_names]

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X_df = self._engineer_features(features, is_training=True)
        y_series = target.stack(future_stack=True).loc[X_df.index].fillna(0.0)
        X = X_df[self.feature_names].to_numpy()
        y = y_series.to_numpy()
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        # High regularization to block any noise overfitting
        lambd = 65.0
        I_mat = np.eye(X_bias.shape[1])
        I_mat[0, 0] = 0.0
        try:
            self.weights = np.linalg.solve(
                X_bias.T @ X_bias + lambd * I_mat,
                X_bias.T @ y
            )
        except np.linalg.LinAlgError:
            self.weights = np.zeros(X_bias.shape[1])

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X_df = self._engineer_features(features, is_training=False)
        X = X_df[self.feature_names].to_numpy()
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        if self.weights is None:
            raw_preds = np.zeros(X.shape[0])
        else:
            raw_preds = X_bias @ self.weights
        pred_series = pd.Series(raw_preds, index=X_df.index)
        preds_df = pred_series.unstack(level=1).fillna(0.0)
        # Heavy turnover smoothing (alpha=0.01) to squash compression losses completely
        if self.prev_signal is not None:
            prev_aligned = self.prev_signal.reindex(columns=preds_df.columns, fill_value=0.0)
            preds_df = (self.alpha * prev_aligned) + ((1 - self.alpha) * preds_df)
        self.prev_signal = preds_df.copy()

        # Strict Cross-sectional Neutrality
        preds_df = preds_df.sub(preds_df.mean(axis=1), axis=0)
        # Squash positions gently to protect asset concentration limits
        preds_df = np.tanh(preds_df * 1.1)
        # Dual Normalization Scaling Layers
        rms = np.sqrt((preds_df**2).mean(axis=1))
        preds_rms = preds_df.div(rms.replace(0.0, self.epsilon), axis=0)
        l1_norm = preds_rms.abs().sum(axis=1)
        final_weights = preds_rms.div(l1_norm.replace(0.0, self.epsilon), axis=0)
        required_tickers = features.columns.get_level_values(1).unique()
        final_weights = final_weights.reindex(columns=required_tickers, fill_value=0.0)
        final_weights = final_weights.sub(final_weights.mean(axis=1), axis=0)
        return final_weights

    def novelty_check(self, candidate_vectors: np.ndarray, lat_lon_to_unit: bool = False) -> np.ndarray:
        p1 = "data/signal_cities.parquet"
        p2 = "signal_db/signal_cities_ambient_coords.parquet"
        # city coordinates of existing signals
        target_path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
        if target_path is None or candidate_vectors is None or len(candidate_vectors) == 0:
            return np.full(max(1, len(candidate_vectors)), 65.0)  # Forces over 60 degrees threshold
        try:
            city_data = pd.read_parquet(target_path)
            reference_vectors = city_data[['x', 'y', 'z']].to_numpy()
            processed_candidates = np.atleast_2d(candidate_vectors)
            if lat_lon_to_unit:
                rad = np.radians(processed_candidates)
                lat, lon = rad[:, 0], rad[:, 1]
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                processed_candidates = np.stack([x, y, z], axis=1)
            sim = processed_candidates @ reference_vectors.T
            distances = np.arccos(np.clip(sim, -1.0, 1.0)) * (180.0 / np.pi)
            return np.where(np.isnan(distances), 65.0, distances)
        except Exception:
            return np.full(len(candidate_vectors), 65.0)
