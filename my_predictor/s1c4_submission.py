# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn"
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from predictor import Predictor


class MyPredictor(Predictor):
    """
    AlphaNova Season 1 - Hardened Robust Predictor
    Designed to survive padding, pickle reload, shape changes, and edge cases.
    """

    def __init__(self):
        super().__init__()
        self.n_assets = 1
        self.n_features = 6
        self.tickers = ["asset_0"]
        self.feat_names = [f"f_{i}" for i in range(6)]
        self.feature_means = np.zeros(6)
        self.feature_stds = np.ones(6)
        self.model = None
        self.is_trained = False

    def train(self, features, target):
        np.random.seed(42)

        try:
            self._discover_structure(features, target)
            X_raw, y_arr, T, J = self._extract(features, target)
            self.n_assets = J
            T, J, F = X_raw.shape
            self.n_features = F

            if y_arr is None or y_arr.size == 0:
                self.is_trained = False
                return

            X_flat = X_raw.reshape(-1, F)
            y_flat = y_arr.reshape(-1)

            # Keep only valid targets
            valid = np.isfinite(y_flat)
            if not np.any(valid):
                self.is_trained = False
                return

            X_train = X_flat[valid]
            y_train = y_flat[valid]

            # Impute
            medians = np.nanmedian(X_train, axis=0)
            medians = np.where(np.isfinite(medians), medians, 0.0)
            X_train = np.where(np.isfinite(X_train), X_train, medians)

            # Standardize
            self.feature_means = np.mean(X_train, axis=0)
            self.feature_stds = np.std(X_train, axis=0) + 1e-8

            X_norm = (X_train - self.feature_means) / self.feature_stds
            X_norm = np.clip(X_norm, -4.0, 4.0)

            self.model = Ridge(alpha=50.0, fit_intercept=True, random_state=42)
            self.model.fit(X_norm, y_train)
            self.is_trained = True

        except Exception:
            self.is_trained = False

    def predict(self, features):
        # Always return a valid DataFrame — never raise
        try:
            T = len(features) if hasattr(features, "__len__") else 1
            index = getattr(features, "index", pd.RangeIndex(T))
            columns = self.tickers if self.tickers else [f"asset_{i}" for i in range(self.n_assets)]

            if T == 0:
                return pd.DataFrame(index=index, columns=columns, dtype=float)

            X_raw, _, T, J = self._extract(features, target=None)
            T, J, F = X_raw.shape

            # Safety: if model not trained or shape mismatch → zeros
            if (not self.is_trained or 
                self.model is None or 
                self.feature_means is None or 
                len(self.feature_means) != F):
                return pd.DataFrame(0.0, index=index, columns=columns[:J] if J <= len(columns) else columns)

            X_flat = X_raw.reshape(-1, F)
            X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

            means = self.feature_means
            stds = self.feature_stds

            X_norm = (X_flat - means) / stds
            X_norm = np.clip(X_norm, -4.0, 4.0)

            raw = self.model.predict(X_norm).reshape(T, J)

            # Strict cross-sectional demean + L1 scale
            out = np.zeros_like(raw)
            for t in range(T):
                row = raw[t]
                if not np.isfinite(row).any():
                    continue
                row = np.nan_to_num(row, nan=np.nanmedian(row) if np.isfinite(row).any() else 0.0)
                row = row - np.mean(row)
                l1 = np.sum(np.abs(row))
                if l1 > 1e-12:
                    row /= l1
                out[t] = row - np.mean(row)

            # Align columns safely
            final_columns = columns[:J] if J <= len(columns) else [f"asset_{i}" for i in range(J)]
            return pd.DataFrame(out, index=index, columns=final_columns)

        except Exception:
            # Ultimate fallback
            T = len(features) if hasattr(features, "__len__") else 1
            index = getattr(features, "index", pd.RangeIndex(T))
            columns = self.tickers if self.tickers else [f"asset_{i}" for i in range(max(1, self.n_assets))]
            return pd.DataFrame(0.0, index=index, columns=columns)

    def _discover_structure(self, features, target):
        try:
            if isinstance(features, pd.DataFrame):
                if isinstance(features.columns, pd.MultiIndex):
                    self.feat_names = sorted(features.columns.get_level_values(0).unique().tolist())
                    self.tickers = sorted(features.columns.get_level_values(1).unique().tolist())
                else:
                    self.tickers = [str(c) for c in features.columns]
                    self.feat_names = [f"f_{i}" for i in range(6)]
            elif hasattr(target, "columns"):
                self.tickers = [str(c) for c in target.columns]
                self.feat_names = [f"f_{i}" for i in range(6)]
            else:
                n = 1
                if hasattr(target, "shape") and len(target.shape) > 1:
                    n = target.shape[1]
                self.tickers = [f"asset_{i}" for i in range(n)]
                self.feat_names = [f"f_{i}" for i in range(6)]

            self.n_assets = max(1, len(self.tickers))
            self.n_features = max(1, len(self.feat_names))
        except Exception:
            self.n_assets = 1
            self.n_features = 6
            self.tickers = ["asset_0"]
            self.feat_names = [f"f_{i}" for i in range(6)]

    def _extract(self, features, target=None):
        """Extremely tolerant tensor extraction"""
        try:
            if isinstance(features, pd.DataFrame):
                T = len(features)
                if isinstance(features.columns, pd.MultiIndex):
                    feat_names = self.feat_names
                    tickers = self.tickers
                    J, F = len(tickers), len(feat_names)
                    X = np.zeros((T, J, F), dtype=np.float64)
                    for fi, f in enumerate(feat_names):
                        for ji, tk in enumerate(tickers):
                            key = (f, tk)
                            if key in features.columns:
                                X[:, ji, fi] = np.nan_to_num(features[key].to_numpy(dtype=float), nan=0.0)
                else:
                    arr = np.nan_to_num(features.to_numpy(dtype=float), nan=0.0)
                    J = self.n_assets
                    F = self.n_features
                    needed = J * F
                    fixed = np.zeros((T, needed), dtype=np.float64)
                    fixed[:, :min(arr.shape[1], needed)] = arr[:, :min(arr.shape[1], needed)]
                    X = fixed.reshape(T, J, F)
            else:
                arr = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0)
                if arr.ndim == 3:
                    X = arr
                else:
                    T = arr.shape[0] if arr.ndim >= 1 else 1
                    J = self.n_assets
                    F = self.n_features
                    needed = J * F
                    fixed = np.zeros((T, needed), dtype=np.float64)
                    if arr.ndim == 2:
                        fixed[:, :min(arr.shape[1], needed)] = arr[:, :min(arr.shape[1], needed)]
                    elif arr.ndim == 1:
                        fixed[:, 0] = arr
                    X = fixed.reshape(T, J, F)

            # Target handling
            y = None
            if target is not None:
                tarr = np.asarray(target, dtype=float).ravel()
                expected = X.shape[0] * X.shape[1]
                if tarr.size >= expected:
                    y = tarr[:expected]
                elif tarr.size == X.shape[0]:
                    y = np.repeat(tarr, X.shape[1])
                else:
                    y = np.zeros(expected)

            return X, y, X.shape[0], X.shape[1]

        except Exception:
            T = len(features) if hasattr(features, "__len__") else 1
            J = self.n_assets
            F = self.n_features
            return np.zeros((T, J, F)), None, T, J


# Required by the platform
predictor = MyPredictor()
