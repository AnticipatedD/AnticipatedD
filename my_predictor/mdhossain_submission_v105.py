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

# -----------------------------------------------------------------------------
# Base Class Handling
# -----------------------------------------------------------------------------
try:
    from predictor import Predictor
except ImportError:
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
    MyPredictor V1.1.10: Master Prize-Pool Architecture.
    Submitted as dvane_submission_v5_1.py
    Signal ID: signal_3bdd884cf0150041 | Sharpe: -0.0075 | IC: 0.0051
    """
    def __init__(self, alpha: float = 0.05, epsilon: float = 1e-8):
        self.weights = None
        self.feature_names = []
        self.alpha = alpha
        self.epsilon = epsilon
        self.prev_signal = None
        self.prev_tickers = None
        self.is_trained = False
        self.model = Ridge(alpha=alpha)
        self.optimal_concentration = 1.0
        self.decay = 0.7
        self.target_bound = 0.1

    def _engineer_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Extracts predictive structural features: Rolling Momentum and Cross-Sectional Z-Scores.
        """
        tickers = features_df.columns.get_level_values(1).unique()
        N_time = len(features_df)
        J_assets = len(tickers)

        if not self.feature_names:
            self.feature_names = list(features_df.columns.get_level_values(0).unique())

        feature_matrices = []

        for feat in self.feature_names:
            if feat in features_df.columns.levels[0]:
                raw_mat = features_df[feat].astype(np.float64).fillna(0.0).to_numpy()

                # Feature 1: Cross-Sectional Z-Score (Relative Value)
                row_mean = raw_mat.mean(axis=1, keepdims=True)
                row_std = raw_mat.std(axis=1, keepdims=True) + self.epsilon
                z_score = (raw_mat - row_mean) / row_std
                feature_matrices.append(z_score)

                # Feature 2: Momentum (Directional Change over Time)
                if N_time > 2:
                    mom = np.zeros_like(raw_mat)
                    mom[1:] = raw_mat[1:] - raw_mat[:-1]
                    feature_matrices.append(mom)
                else:
                    feature_matrices.append(np.zeros_like(raw_mat))

        if not feature_matrices:
            return np.zeros((N_time, J_assets, 1))

        # Shape: (N_time, J_assets, N_features)
        return np.stack(feature_matrices, axis=-1)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Trains global predictive parameters across the pooled asset-time universe.
        """
        if features is None or features.empty or target is None or target.empty:
            return

        self.feature_names = list(features.columns.get_level_values(0).unique())

        features_sorted = features.sort_index(ascending=True)
        target_sorted = target.sort_index(ascending=True)

        X_tensor = self._engineer_features(features_sorted)
        y_matrix = target_sorted.astype(np.float64).fillna(0.0).to_numpy()

        N_time, J_assets, N_feats = X_tensor.shape

        # Pool rows and columns together to ensure the model generalizes across assets
        X_flat = X_tensor.reshape(-1, N_feats)
        y_flat = y_matrix.reshape(-1)

        # Fit the regularized ridge model to learn actual predictive weights
        self.model.fit(X_flat, y_flat)
        self.is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        features_sorted = features.sort_index(ascending=True)
        tickers = features_sorted.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features_sorted.index, columns=tickers, dtype=np.float32)

        if len(features_sorted) == 0:
            return zero_signal

        try:
            X_tensor = self._engineer_features(features_sorted)
            N_time, J_assets, N_feats = X_tensor.shape

            # Use trained model predictions if available; fallback to flat weights if not
            if self.is_trained:
                X_flat = X_tensor.reshape(-1, N_feats)
                raw_preds = self.model.predict(X_flat)
                alpha_raw = raw_preds.reshape(N_time, J_assets)
            else:
                alpha_raw = X_tensor[..., 0] * 0.01  # Fallback to a minor scale of the baseline feature

            # Market-Neutral Transformation: Zero out cross-sectional mean
            alpha_demeaned = alpha_raw - alpha_raw.mean(axis=1, keepdims=True)

            # Scale allocations to meet tournament concentration standards
            row_norms = np.linalg.norm(alpha_demeaned, axis=1, keepdims=True) + self.epsilon
            target_signals = (alpha_demeaned / row_norms) * self.optimal_concentration

            # Smooth allocations over time rows to reduce execution churn
            smoothed_positions = np.zeros_like(target_signals)
            if (
                self.prev_signal is not None
                and self.prev_signal.shape == (J_assets,)
                and self.prev_tickers is not None
                and self.prev_tickers.equals(tickers)
            ):
                active_allocation = self.prev_signal.copy()
            else:
                active_allocation = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target_allocation = target_signals[t]
                if np.all(np.abs(target_allocation) < 1e-8):
                    current_allocation = active_allocation.copy()
                else:
                    current_allocation = (self.decay * target_allocation) + ((1.0 - self.decay) * active_allocation)

                current_allocation -= current_allocation.mean()
                smoothed_positions[t] = current_allocation
                active_allocation = current_allocation.copy()

            self.prev_signal = smoothed_positions[-1].copy()
            self.prev_tickers = tickers.copy()

            # Final output formatting with strict clipping boundaries
            final_df = pd.DataFrame(smoothed_positions, index=features_sorted.index, columns=tickers)
            final_df = final_df.clip(-self.target_bound, self.target_bound)

            return final_df.reindex(index=features.index).astype(np.float32)

        except Exception:
            return zero_signal

predictor = MyPredictor()
