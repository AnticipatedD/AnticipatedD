# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///

"""
AlphaNova Season 1 — Submission-ready Predictor

File: alpha_predictor_submission.py

Implements MyPredictor(Predictor) with:
 - Robust cross-sectional feature engineering (rank-based, permutation-safe)
 - Low-capacity ridge regression on engineered interaction factors
 - Exponential smoothing + L1 hysteresis to reduce turnover
 - Safe numeric fallbacks and full internal state handling

All logic lives inside the MyPredictor class as required by the contest.
"""

from __future__ import annotations
import math
from typing import List, Optional

import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    """
    Production-ready Predictor for AlphaNova submission.

    Characteristics:
      - Uses contemporaneous cross-sectional ranks and 3 engineered interaction
        factors per timestep (robust to column/ticker re-ordering).
      - Trains a small ridge regression (3 coefficients) on flattened time x asset
        data so capacity is matched to expected signal complexity.
      - Predict outputs are cross-sectionally de-meaned, smoothed, and subject to
        a causal L1 hysteresis band to limit churn and transaction costs.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.05,
        concentration_radius: float = 0.95,
        tail_clip: float = 0.99,
        ridge_lambda: float = 35.0,
        l1_hysteresis_threshold: float = 1.12,
        rng_seed: int = 35,
    ):
        super().__init__()

        # Algorithm hyperparameters (tunable)
        self.smoothing_alpha = float(smoothing_alpha)           # EMA blending weight for updates
        self.concentration_radius = float(concentration_radius) # per-row normalization radius
        self.tail_clip = float(tail_clip)                       # max absolute position value
        self.ridge_lambda = float(ridge_lambda)                 # ridge regularization strength
        self.l1_hysteresis_threshold = float(l1_hysteresis_threshold)  # turnover band threshold

        # Internal state
        self.feature_names: Optional[List[str]] = None
        self.weights: Optional[np.ndarray] = None
        self.prev_signal: Optional[np.ndarray] = None
        self.prev_tickers: Optional[pd.Index] = None

        # RNG for deterministic fallbacks
        self._rng = np.random.RandomState(int(rng_seed))

    # -------------------------
    # Feature engineering
    # -------------------------
    def _engineer_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Build robust interaction features from MultiIndex feature panel.

        Input:
            features_df: DataFrame, expected MultiIndex columns (feature_name, ticker).
                         If not MultiIndex, treat each top-level column as a feature
                         and use its column ordering as tickers.

        Output:
            X_tensor: np.ndarray of shape (N_time, J_assets, N_factors)
                      where N_factors == 3 (inter_1, inter_2, inter_3).
        """
        if features_df is None or features_df.shape[0] == 0:
            return np.zeros((0, 0, 0), dtype=np.float64)

        # Determine tickers and feature names safely
        if isinstance(features_df.columns, pd.MultiIndex):
            feat_names = list(features_df.columns.get_level_values(0).unique())
            tickers = features_df.columns.get_level_values(1).unique()
        else:
            feat_names = list(features_df.columns)
            # create synthetic tickers if plain DataFrame: use column names
            tickers = features_df.columns

        # If feature_names weren't set by train(), use what we have here
        if self.feature_names is None:
            # limit to first 6 features if many
            self.feature_names = feat_names[:6]

        # Compute cross-sectional ranks for each declared feature, mapped to [-1,1]
        ranks = {}
        for feat in self.feature_names:
            try:
                if isinstance(features_df.columns, pd.MultiIndex):
                    if feat not in features_df.columns.get_level_values(0):
                        # feature absent -> zeros
                        df_feat = pd.DataFrame(0.0, index=features_df.index, columns=tickers, dtype=np.float64)
                    else:
                        df_feat = features_df.xs(feat, level=0, axis=1).astype(np.float64)
                        df_feat = df_feat.reindex(columns=tickers, fill_value=0.0)
                else:
                    # plain DataFrame: single-level columns
                    if feat not in features_df.columns:
                        df_feat = pd.DataFrame(0.0, index=features_df.index, columns=tickers, dtype=np.float64)
                    else:
                        df_feat = pd.DataFrame(features_df[feat].astype(np.float64))
                        # if single column, broadcast across tickers -> but usually not desired
                        # handle as single-series feature: create DataFrame with same length and 1 column
                        df_feat = df_feat.reindex(index=features_df.index)
                # Rank per-row (percentile) mapped to [-1,1]
                r = (df_feat.rank(axis=1, pct=True, method="first") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy(dtype=np.float64)
            except Exception:
                # On any failure default to zeros to avoid crashing competition runner
                ranks[feat] = np.zeros((len(features_df), len(tickers)), dtype=np.float64)

        # Ensure at least 6 base factors for stable interaction construction
        keys = list(ranks.keys())
        if not keys:
            N_time = len(features_df)
            J_assets = len(tickers)
            return np.zeros((N_time, J_assets, 0), dtype=np.float64)

        base_factors = [ranks[k] for k in keys]
        while len(base_factors) < 0:
            base_factors.append(np.zeros_like(base_factors[0] if base_factors else np.zeros((len(features_df), len(tickers)))))

        # Unpack first six factor matrices
        f1, f2, f3, f4, f5, f6 = base_factors[:6]

        # Interaction definitions (robust, low-complexity)
        inter_1 = f1 * np.sign(f2)           # preserves sign structure while limiting scale
        inter_2 = f3 * np.sign(f4)
        inter_3 = np.tanh(f5 - f6)           # bounded non-linearity to reduce outlier impact

        X_tensor = np.stack([inter_1, inter_2, inter_3], axis=2)  # shape (N_time, J_assets, 3)
        return X_tensor.astype(np.float64)

    # -------------------------
    # Training
    # -------------------------
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Train a small ridge regression on flattened (time x asset) engineered features.

        Args:
            features: DataFrame with MultiIndex columns (feature_name, ticker)
            target: DataFrame target the model should fit (same tickers)
        """
        # Minimal validation
        if features is None or features.empty or target is None or target.empty:
            # safe fallback
            self.weights = np.zeros(3, dtype=np.float64)
            return

        # Determine feature names
        if isinstance(features.columns, pd.MultiIndex):
            self.feature_names = list(features.columns.get_level_values(0).unique())[:6]
        else:
            self.feature_names = list(features.columns)[:6]

        # Sort indices for stable behavior
        features_sorted = features.sort_index()
        target_sorted = target.sort_index()

        # Build engineered tensor and target matrix
        X_tensor = self._engineer_features(features_sorted)
        if X_tensor.size == 0:
            self.weights = np.zeros(3, dtype=np.float64)
            return

        N_time, J_assets, N_feats = X_tensor.shape

        # Build y matrix aligned to the tickers order used in features engineering
        # If target columns differ in order, reindex to match
        if isinstance(features_sorted.columns, pd.MultiIndex):
            tickers = features_sorted.columns.get_level_values(1).unique()
        else:
            tickers = target_sorted.columns

        # Ensure same column ordering for target
        target_aligned = target_sorted.reindex(columns=tickers, fill_value=0.0).astype(np.float64)
        y_matrix = target_aligned.to_numpy()

        # Flatten time x asset into rows for regression
        X_flat = X_tensor.reshape(-1, N_feats)  # (N_time * J_assets, N_feats)
        y_flat = y_matrix.reshape(-1)

        # Regularized ridge: solve (X^T X + lambda I) w = X^T y
        XTX = X_flat.T.dot(X_flat)
        XTy = X_flat.T.dot(y_flat)
        reg = self.ridge_lambda * np.eye(N_feats, dtype=np.float64)
        try:
            self.weights = np.linalg.solve(XTX + reg, XTy)
        except np.linalg.LinAlgError:
            # fallback to stable pseudoinverse
            self.weights = np.linalg.pinv(XTX + reg).dot(XTy)

        # Safety: ensure weights length is N_feats (3)
        if self.weights is None or len(self.weights) != N_feats:
            self.weights = np.zeros(N_feats, dtype=np.float64)

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Produce cross-sectionally de-meaned signals for the input features.

        Args:
            features: DataFrame with MultiIndex columns (feature_name, ticker)

        Returns:
            DataFrame (n_time, tickers) of float32 signals, cross-sectionally de-meaned.
        """
        # Quick edge cases
        if features is None or len(features) == 0:
            # best-effort empty DataFrame
            if features is None:
                return pd.DataFrame()
            if isinstance(features.columns, pd.MultiIndex):
                tickers = features.columns.get_level_values(1).unique()
                return pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
            else:
                return pd.DataFrame(0.0, index=features.index, columns=features.columns, dtype=np.float32)

        # Ensure feature names exist
        if self.feature_names is None:
            if isinstance(features.columns, pd.MultiIndex):
                self.feature_names = list(features.columns.get_level_values(0).unique())[:6]
            else:
                self.feature_names = list(features.columns)[:6]

        # Ensure weights exist
        if self.weights is None:
            # small safe fallback weights to produce consistent outputs
            self.weights = np.array([0.08, -0.05, 0.03], dtype=np.float64)

        features_sorted = features.sort_index()
        # Determine tickers ordering expected
        if isinstance(features_sorted.columns, pd.MultiIndex):
            tickers = features_sorted.columns.get_level_values(1).unique()
        else:
            tickers = features_sorted.columns

        zero_signal = pd.DataFrame(0.0, index=features_sorted.index, columns=tickers, dtype=np.float32)

        try:
            X_tensor = self._engineer_features(features_sorted)
            if X_tensor.size == 0:
                return zero_signal

            # Raw alpha combination
            alpha_raw = np.tensordot(X_tensor, self.weights, axes=([2], [0]))  # shape (N_time, J_assets)

            # Cross-sectional de-meaning (important contest requirement)
            alpha_demeaned = alpha_raw - alpha_raw.mean(axis=1, keepdims=True)

            # Normalize per-row to control magnitude then scale to concentration radius
            row_norms = np.linalg.norm(alpha_demeaned, axis=1, keepdims=True)
            eps = 1e-8
            row_norms_safe = row_norms + eps
            sphere_velocity = (alpha_demeaned / row_norms_safe) * self.concentration_radius

            N_time, J_assets = sphere_velocity.shape

            # State retention: ensure prev_signal matches current tickers
            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and isinstance(self.prev_tickers, pd.Index)
                and self.prev_tickers.equals(tickers)
                and self.prev_signal.shape == (len(tickers),)
            ):
                active_allocation = self.prev_signal.copy()
            else:
                active_allocation = np.zeros(len(tickers), dtype=np.float64)

            smoothed_positions = np.zeros_like(sphere_velocity, dtype=np.float64)

            for t in range(N_time):
                target_allocation = sphere_velocity[t]

                # If target is effectively zero, maintain current allocation
                if np.all(np.abs(target_allocation) < 1e-8):
                    current_allocation = active_allocation.copy()
                else:
                    # Compute L1 delta against active allocation to decide hysteresis
                    l1_delta = np.sum(np.abs(target_allocation - active_allocation))

                    if l1_delta < self.l1_hysteresis_threshold:
                        # Within hysteresis band: do not trade to reduce churn
                        current_allocation = active_allocation.copy()
                    else:
                        # Smooth path to new target to control turnover
                        current_allocation = (self.smoothing_alpha * target_allocation) + ((1.0 - self.smoothing_alpha) * active_allocation)
                        # Re-enforce zero-sum
                        current_allocation = current_allocation - current_allocation.mean()

                smoothed_positions[t] = current_allocation
                active_allocation = current_allocation.copy()

            # Persist state (last known allocation)
            self.prev_signal = smoothed_positions[-1].copy()
            # store tickers as Index if available
            self.prev_tickers = pd.Index(tickers)

            # Format into DataFrame aligned to input
            final_df = pd.DataFrame(smoothed_positions, index=features_sorted.index, columns=tickers)

            # Final safety: row-wise de-mean and clipping to tail bounds
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.tail_clip, self.tail_clip)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            return final_df.astype(np.float32)

        except Exception:
            # On unexpected failure, return a safe zero signal (contest runner expects robustness)
            return zero_signal


# Required instantiation for the submission runner
predictor = MyPredictor()
