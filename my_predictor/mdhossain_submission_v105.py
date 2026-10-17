# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
# ]
# ///
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from predictor import Predictor

class MyPredictor(Predictor):
    """
    Stateless market-neutral ridge on cross-sectional z-scores + momentum.
    Guarantees every row sums to zero.
    """
    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.feature_names = []
        self.is_trained = False
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.optimal_concentration = 1.0
        self.target_bound = 0.15

    def _engineer_features(self, features_df: pd.DataFrame) -> np.ndarray:
        if features_df is None or features_df.empty:
            return np.zeros((0, 0, 1), dtype=np.float64)

        if not self.feature_names:
            self.feature_names = list(features_df.columns.get_level_values(0).unique())

        tickers = features_df.columns.get_level_values(1).unique()
        N_time = len(features_df)
        J_assets = len(tickers)
        feature_matrices = []

        for feat in self.feature_names:
            try:
                raw = features_df[feat].astype(np.float64).to_numpy()
            except Exception:
                raw = np.zeros((N_time, J_assets), dtype=np.float64)

            # Cross-sectional z-score
            row_mean = np.nanmean(raw, axis=1, keepdims=True)
            row_std = np.nanstd(raw, axis=1, keepdims=True)
            row_std = np.where(row_std < self.epsilon, self.epsilon, row_std)
            z = (raw - row_mean) / row_std
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            z = np.clip(z, -3.0, 3.0)
            feature_matrices.append(z)

            # Lag-1 momentum
            mom = np.zeros_like(raw)
            if N_time > 1:
                mom[1:] = raw[1:] - raw[:-1]
            mom = np.nan_to_num(mom, nan=0.0, posinf=0.0, neginf=0.0)
            feature_matrices.append(mom)

        if not feature_matrices:
            return np.zeros((N_time, J_assets, 1), dtype=np.float64)

        return np.stack(feature_matrices, axis=-1)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is None or features.empty or target is None or target.empty:
            self.is_trained = False
            return

        self.feature_names = list(features.columns.get_level_values(0).unique())

        features_sorted = features.sort_index(ascending=True)
        target_sorted = target.sort_index(ascending=True)

        common_idx = features_sorted.index.intersection(target_sorted.index)
        if len(common_idx) == 0:
            self.is_trained = False
            return

        features_sorted = features_sorted.loc[common_idx]
        target_sorted = target_sorted.loc[common_idx]

        X_tensor = self._engineer_features(features_sorted)
        y_matrix = target_sorted.astype(np.float64).fillna(0.0).to_numpy()

        if X_tensor.size == 0 or y_matrix.size == 0:
            self.is_trained = False
            return

        N_time, J_assets, N_feats = X_tensor.shape
        X_flat = X_tensor.reshape(-1, N_feats)
        y_flat = y_matrix.reshape(-1)

        mask = np.isfinite(X_flat).all(axis=1) & np.isfinite(y_flat)
        if mask.sum() < 10:
            self.is_trained = False
            return

        self.model.fit(X_flat[mask], y_flat[mask])
        self.is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or features.empty:
            return pd.DataFrame()

        features_sorted = features.sort_index(ascending=True)
        tickers = features_sorted.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(
            0.0, index=features_sorted.index, columns=tickers, dtype=np.float32
        )

        try:
            X_tensor = self._engineer_features(features_sorted)
            if X_tensor.size == 0:
                return zero_signal

            N_time, J_assets, N_feats = X_tensor.shape

            if self.is_trained and hasattr(self.model, "coef_"):
                X_flat = X_tensor.reshape(-1, N_feats)
                raw_preds = self.model.predict(X_flat)
                alpha_raw = raw_preds.reshape(N_time, J_assets)
            else:
                alpha_raw = X_tensor[..., 0] * 0.05

            # Market-neutral demean
            alpha_demeaned = alpha_raw - np.nanmean(alpha_raw, axis=1, keepdims=True)
            alpha_demeaned = np.nan_to_num(alpha_demeaned, nan=0.0)

            # L2 normalize + scale
            row_norms = np.linalg.norm(alpha_demeaned, axis=1, keepdims=True)
            row_norms = np.where(row_norms < self.epsilon, self.epsilon, row_norms)
            target_signals = (alpha_demeaned / row_norms) * self.optimal_concentration

            # Intra-batch exponential smoothing (stateless)
            decay = 0.65
            smoothed = np.zeros_like(target_signals)
            prev = np.zeros(J_assets, dtype=np.float64)
            for t in range(N_time):
                curr = decay * target_signals[t] + (1.0 - decay) * prev
                curr = curr - curr.mean()          # keep neutral
                smoothed[t] = curr
                prev = curr

            final_df = pd.DataFrame(
                smoothed, index=features_sorted.index, columns=tickers
            )

            # Clip (this breaks neutrality)
            final_df = final_df.clip(-self.target_bound, self.target_bound)

            # Re-demean after clip
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            # Align to original call
            final_df = final_df.reindex(index=features.index, columns=tickers).fillna(0.0)

            # Final hard demean – mandatory for the platform check
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal.reindex(index=features.index).fillna(0.0).astype(np.float32)

predictor = MyPredictor()
