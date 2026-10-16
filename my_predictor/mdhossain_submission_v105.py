# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///

"""
Submission-ready Predictor with deterministic novelty projection.

Filename: predictor.py

Notes:
 - Implements MyPredictor(Predictor) required by the contest runner.
 - Adds a novelty_projection computed from training-time gauge-free trajectory:
   selects the smallest principal component direction, orthogonalizes it to the city,
   and blends it into predictions to push the city's angle away from common directions.
 - All logic is inside the class. Final instance at bottom: predictor = MyPredictor()
"""

from __future__ import annotations
from typing import List, Optional
import math

import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    def __init__(
        self,
        smoothing_alpha: float = 0.20,
        concentration_radius: float = 0.75,
        tail_clip: float = 0.99,
        ridge_lambda: float = 35.0,
        l1_hysteresis_threshold: float = 1.6,
        novelty_blend: float = 0.70,
        rng_seed: int = 42,
    ):
        super().__init__()
        # Hyperparameters
        self.smoothing_alpha = float(smoothing_alpha)
        self.concentration_radius = float(concentration_radius)
        self.tail_clip = float(tail_clip)
        self.ridge_lambda = float(ridge_lambda)
        self.l1_hysteresis_threshold = float(l1_hysteresis_threshold)
        self.novelty_blend = float(novelty_blend)

        # Internal state
        self.feature_names: Optional[List[str]] = None
        self.weights: Optional[np.ndarray] = None
        self.prev_signal: Optional[np.ndarray] = None
        self.prev_tickers: Optional[pd.Index] = None

        # Novelty projection vector (computed at train time)
        self.novelty_vector: Optional[np.ndarray] = None

        # Deterministic RNG for safe fallbacks
        self._rng = np.random.RandomState(int(rng_seed))

    # -------------------------
    # Feature engineering
    # -------------------------
    def _engineer_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Produce a robust (N_time, J_assets, 3) tensor of interaction factors.
        """
        if features_df is None or features_df.shape[0] == 0:
            return np.zeros((0, 0, 0), dtype=np.float64)

        if isinstance(features_df.columns, pd.MultiIndex):
            feat_names = list(features_df.columns.get_level_values(0).unique())
            tickers = features_df.columns.get_level_values(1).unique()
        else:
            feat_names = list(features_df.columns)
            tickers = features_df.columns

        if self.feature_names is None:
            self.feature_names = feat_names[:6]

        # compute per-feature cross-sectional ranks mapped to [-1,1]
        ranks = {}
        for feat in self.feature_names:
            try:
                if isinstance(features_df.columns, pd.MultiIndex):
                    if feat not in features_df.columns.get_level_values(0):
                        df_feat = pd.DataFrame(0.0, index=features_df.index, columns=tickers, dtype=np.float64)
                    else:
                        df_feat = features_df.xs(feat, level=0, axis=1).astype(np.float64)
                        df_feat = df_feat.reindex(columns=tickers, fill_value=0.0)
                else:
                    if feat not in features_df.columns:
                        df_feat = pd.DataFrame(0.0, index=features_df.index, columns=tickers, dtype=np.float64)
                    else:
                        # In contest data, features are MultiIndex — this branch is fallback.
                        df_feat = pd.DataFrame(0.0, index=features_df.index, columns=tickers, dtype=np.float64)

                r = (df_feat.rank(axis=1, pct=True, method="first") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy(dtype=np.float64)
            except Exception:
                ranks[feat] = np.zeros((len(features_df), len(tickers)), dtype=np.float64)

        keys = list(ranks.keys())
        if not keys:
            return np.zeros((len(features_df), len(tickers), 0), dtype=np.float64)

        base_factors = [ranks[k] for k in keys]
        while len(base_factors) < 6:
            base_factors.append(np.zeros_like(base_factors[0]))

        f1, f2, f3, f4, f5, f6 = base_factors[:6]

        inter_1 = f1 * np.sign(f2)
        inter_2 = f3 * np.sign(f4)
        inter_3 = np.tanh(f5 - f6)

        X_tensor = np.stack([inter_1, inter_2, inter_3], axis=2)
        return X_tensor.astype(np.float64)

    # -------------------------
    # Training
    # -------------------------
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Train ridge regression on flattened time-asset engineered features.
        During training compute the novelty_vector from the learned gauge-free
        unit trajectories (derived from alpha on training rows).
        """
        # Safety checks
        if features is None or features.empty or target is None or target.empty:
            self.weights = np.zeros(3, dtype=np.float64)
            # novelty_vector fallback: random orthonormal
            self.novelty_vector = None
            return

        # set feature names
        if isinstance(features.columns, pd.MultiIndex):
            self.feature_names = list(features.columns.get_level_values(0).unique())[:6]
        else:
            self.feature_names = list(features.columns)[:6]

        features_sorted = features.sort_index()
        target_sorted = target.sort_index()

        X_tensor = self._engineer_features(features_sorted)
        if X_tensor.size == 0:
            self.weights = np.zeros(3, dtype=np.float64)
            self.novelty_vector = None
            return

        N_time, J_assets, N_feats = X_tensor.shape

        if isinstance(features_sorted.columns, pd.MultiIndex):
            tickers = features_sorted.columns.get_level_values(1).unique()
        else:
            tickers = target_sorted.columns

        target_aligned = target_sorted.reindex(columns=tickers, fill_value=0.0).astype(np.float64)
        y_matrix = target_aligned.to_numpy()

        X_flat = X_tensor.reshape(-1, N_feats)
        y_flat = y_matrix.reshape(-1)

        XTX = X_flat.T.dot(X_flat)
        XTy = X_flat.T.dot(y_flat)
        reg = self.ridge_lambda * np.eye(N_feats, dtype=np.float64)
        try:
            self.weights = np.linalg.solve(XTX + reg, XTy)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.pinv(XTX + reg).dot(XTy)

        if self.weights is None or len(self.weights) != N_feats:
            self.weights = np.zeros(N_feats, dtype=np.float64)

        # --- compute training-time gauge-free unit trajectories to derive novelty vector
        try:
            alpha_raw = np.tensordot(X_tensor, self.weights, axes=([2], [0]))  # (N_time, J_assets)
            alpha_dm = alpha_raw - alpha_raw.mean(axis=1, keepdims=True)      # zero-sum per row

            # per-row unit-normalized vectors (avoid divide-by-zero)
            norms = np.linalg.norm(alpha_dm, axis=1, keepdims=True)
            eps = 1e-8
            norms_safe = norms + eps
            gf_train = alpha_dm / norms_safe  # (N_time, J_assets)

            # Compute city (time mean) and its normalized form
            B = gf_train.mean(axis=0)
            B_norm = np.linalg.norm(B)
            if B_norm < 1e-12:
                city = None
            else:
                city = B / B_norm

            # Covariance across assets of gf_train; use smallest-eigenvector as candidate
            # (direction with minimal variance across time)
            cov = np.cov(gf_train, rowvar=False)  # shape J_assets x J_assets
            # Numerical robustness: small jitter
            cov = cov + 1e-8 * np.eye(cov.shape[0])
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                # pick eigenvector corresponding to smallest eigenvalue
                v = eigvecs[:, 0]
            except Exception:
                # fallback: random vector
                v = self._rng.normal(size=(J_assets,))
                v = v / (np.linalg.norm(v) + 1e-12)

            # orthogonalize v against city if city exists
            if city is not None:
                proj = np.dot(v, city) * city
                v_orth = v - proj
                vnorm = np.linalg.norm(v_orth)
                if vnorm < 1e-8:
                    # perturb slightly and renormalize
                    v_orth = v + 1e-3 * self._rng.normal(size=v.shape)
                    vnorm = np.linalg.norm(v_orth)
                novelty_vec = v_orth / (vnorm + 1e-12)
            else:
                novelty_vec = v / (np.linalg.norm(v) + 1e-12)

            self.novelty_vector = novelty_vec  # length J_assets

        except Exception:
            # If anything fails, clear novelty vector; prediction will fall back to learned alpha
            self.novelty_vector = None

    # -------------------------
    # Prediction
    # -------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict cross-sectionally de-meaned signals with novelty-blended projection.
        """
        if features is None or len(features) == 0:
            if features is None:
                return pd.DataFrame()
            if isinstance(features.columns, pd.MultiIndex):
                tickers = features.columns.get_level_values(1).unique()
                return pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
            else:
                return pd.DataFrame(0.0, index=features.index, columns=features.columns, dtype=np.float32)

        if self.feature_names is None:
            if isinstance(features.columns, pd.MultiIndex):
                self.feature_names = list(features.columns.get_level_values(0).unique())[:6]
            else:
                self.feature_names = list(features.columns)[:6]

        if self.weights is None:
            # safe fallback coefficients
            self.weights = np.array([0.08, -0.05, 0.03], dtype=np.float64)

        features_sorted = features.sort_index()
        if isinstance(features_sorted.columns, pd.MultiIndex):
            tickers = features_sorted.columns.get_level_values(1).unique()
        else:
            tickers = features_sorted.columns

        zero_signal = pd.DataFrame(0.0, index=features_sorted.index, columns=tickers, dtype=np.float32)

        try:
            X_tensor = self._engineer_features(features_sorted)
            if X_tensor.size == 0:
                return zero_signal

            alpha_raw = np.tensordot(X_tensor, self.weights, axes=([2], [0]))  # (N_time, J_assets)
            alpha_dm = alpha_raw - alpha_raw.mean(axis=1, keepdims=True)

            # If novelty_vector exists and dimensions match, blend original alpha with its projection
            N_time, J_assets = alpha_dm.shape
            if self.novelty_vector is not None and len(self.novelty_vector) == J_assets:
                nv = self.novelty_vector.reshape(-1)  # (J_assets,)
                # per-row scalar projection onto novelty vector
                proj_scalar = alpha_dm.dot(nv)               # shape (N_time,)
                proj_component = np.outer(proj_scalar, nv)   # shape (N_time, J_assets)
                blended = (self.novelty_blend * alpha_dm) + ((1.0 - self.novelty_blend) * proj_component)
            else:
                blended = alpha_dm

            # Cross-sectional renormalize: convert to per-row unit-scale then apply concentration radius
            norms = np.linalg.norm(blended, axis=1, keepdims=True)
            eps = 1e-8
            norms_safe = norms + eps
            sphere_velocity = (blended / norms_safe) * self.concentration_radius

            # Smoothing + L1 hysteresis (causal)
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

                # flat warmup check
                if np.all(np.abs(target_allocation) < 1e-12):
                    current_allocation = active_allocation.copy()
                else:
                    l1_delta = np.sum(np.abs(target_allocation - active_allocation))
                    if l1_delta < self.l1_hysteresis_threshold:
                        current_allocation = active_allocation.copy()
                    else:
                        current_allocation = (self.smoothing_alpha * target_allocation) + ((1.0 - self.smoothing_alpha) * active_allocation)
                        # re-enforce dollar neutrality
                        current_allocation = current_allocation - current_allocation.mean()

                smoothed_positions[t] = current_allocation
                active_allocation = current_allocation.copy()

            self.prev_signal = smoothed_positions[-1].copy()
            self.prev_tickers = pd.Index(tickers)

            final_df = pd.DataFrame(smoothed_positions, index=features_sorted.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.tail_clip, self.tail_clip)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal


# Required instantiation for contest runner
predictor = MyPredictor()
