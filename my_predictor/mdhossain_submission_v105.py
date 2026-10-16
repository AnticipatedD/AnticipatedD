# /// script
# dependencies = [
#     "numpy",
#     "pandas",
# ]
# ///
import numpy as np
import pandas as pd
from predictor import Predictor

class MyPredictor(Predictor):
    """
    Robust market-neutral signal based on cross-sectional ranks + z-scores.
    Pure numpy/pandas — no external model libraries.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.feature_names = []
        self.is_trained = False
        self.target_bound = 1.35

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # Only need to remember the ordered list of base feature names
        if features is None or features.empty:
            self.feature_names = []
            self.is_trained = False
            return
        try:
            self.feature_names = list(features.columns.get_level_values(0).unique())
            self.is_trained = True
        except Exception:
            self.feature_names = []
            self.is_trained = False

    def _safe_cs_zscore(self, mat: np.ndarray) -> np.ndarray:
        """Cross-sectional z-score, fully NaN-safe."""
        mat = np.asarray(mat, dtype=np.float64)
        row_mean = np.nanmean(mat, axis=1, keepdims=True)
        row_std = np.nanstd(mat, axis=1, keepdims=True)
        row_std = np.where(np.isnan(row_std) | (row_std < self.epsilon), self.epsilon, row_std)
        z = (mat - row_mean) / row_std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _safe_cs_rank(self, mat: np.ndarray) -> np.ndarray:
        """Cross-sectional percentile rank centered at 0."""
        mat = np.asarray(mat, dtype=np.float64)
        # rank along axis=1, then map to [-0.5, 0.5]
        # simple argsort method that tolerates NaNs
        order = np.argsort(np.argsort(np.where(np.isnan(mat), np.inf, mat), axis=1), axis=1)
        n = mat.shape[1]
        ranks = (order.astype(np.float64) / max(n - 1, 1)) - 0.5
        ranks = np.where(np.isnan(mat), 0.0, ranks)
        return ranks

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or features.empty:
            return pd.DataFrame()

        try:
            features_sorted = features.sort_index(ascending=True)
            tickers = features_sorted.columns.get_level_values(1).unique()
            idx = features_sorted.index

            # Determine which base features we can use
            available = list(features_sorted.columns.get_level_values(0).unique())
            feats_to_use = [f for f in self.feature_names if f in available] if self.feature_names else available
            if not feats_to_use:
                feats_to_use = available

            if not feats_to_use:
                return pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

            signals = []
            for feat in feats_to_use:
                try:
                    raw = features_sorted[feat].astype(np.float64).to_numpy()
                except Exception:
                    continue
                # Combine rank and z-score for robustness
                r = self._safe_cs_rank(raw)
                z = self._safe_cs_zscore(raw)
                signals.append(0.6 * r + 0.4 * z)

            if not signals:
                return pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

            # Average the engineered signals
            alpha = np.mean(np.stack(signals, axis=0), axis=0)

            # Market-neutral
            alpha = alpha - np.nanmean(alpha, axis=1, keepdims=True)
            alpha = np.nan_to_num(alpha, nan=0.0)

            # L2 row-normalize then scale
            norms = np.linalg.norm(alpha, axis=1, keepdims=True)
            norms = np.where(norms < self.epsilon, self.epsilon, norms)
            alpha = (alpha / norms) * 1.0

            # Clip
            alpha = np.clip(alpha, -self.target_bound, self.target_bound)

            out = pd.DataFrame(alpha, index=idx, columns=tickers)
            # Align exactly to the original caller's index & columns
            out = out.reindex(index=features.index, columns=tickers).fillna(0.0)
            return out.astype(np.float32)

        except Exception:
            # Last-resort zero signal with correct shape
            try:
                tickers = features.columns.get_level_values(1).unique()
                return pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
            except Exception:
                return pd.DataFrame()

predictor = MyPredictor()
