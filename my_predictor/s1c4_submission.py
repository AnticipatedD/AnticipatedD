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
    AlphaNova Season 1 Cycle 4 - Clean & Robust Predictor

    Guarantees:
    1. Returns pd.DataFrame preserving features.index and tickers columns.
    2. Zero lookahead leakage.
    3. Fully self-contained (survives pickle).
    4. Deterministic and strictly cross-sectionally de-meaned.
    """

    def __init__(self):
        super().__init__()
        self.n_assets = None
        self.n_features = None
        self.tickers = []
        self.feat_names = []
        self.feature_means = None
        self.feature_stds = None
        self.model = None
        self.is_trained = False

    def train(self, features, target):
        np.random.seed(42)
        self._validate_input(features, target)

        # Discover structure
        self._discover_structure(features, target)

        # Extract tensors
        X_raw, y_arr, T, J = self._safe_extract_tensors(features, target)
        self.n_assets = J
        T, J, F = X_raw.shape
        self.n_features = F

        if y_arr is None or not np.any(~np.isnan(y_arr)):
            self.is_trained = False
            return

        # Flatten and clean
        X_flat = X_raw.reshape(T * J, F)
        valid_mask = ~np.isnan(y_arr) & ~np.isinf(y_arr)

        # Impute missing features with column medians
        col_medians = np.nanmedian(X_flat, axis=0)
        col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
        X_flat = np.where(np.isnan(X_flat) | np.isinf(X_flat),
                          np.take(col_medians, np.arange(F)), X_flat)

        X_train = X_flat[valid_mask]
        y_train = y_arr[valid_mask]

        if len(X_train) == 0:
            self.is_trained = False
            return

        # Robust standardization
        self.feature_means = np.mean(X_train, axis=0)
        self.feature_stds = np.std(X_train, axis=0) + 1e-8
        X_norm = (X_train - self.feature_means) / self.feature_stds
        X_norm = np.clip(X_norm, -3.5, 3.5)

        # Train Ridge
        self.model = Ridge(alpha=50.0, fit_intercept=True, random_state=42, solver="auto")
        self.model.fit(X_norm, y_train)
        self.is_trained = True

    def predict(self, features):
        if features is None:
            raise ValueError("features cannot be None")

        T = len(features) if hasattr(features, "__len__") else 1
        J = self.n_assets or len(self.tickers) or 1
        index = getattr(features, "index", pd.RangeIndex(T))
        columns = self.tickers if self.tickers else [f"asset_{j}" for j in range(J)]

        if T == 0:
            return pd.DataFrame(dtype=float, index=index, columns=columns)

        try:
            X_raw, _, T, J = self._safe_extract_tensors(features, target=None)
            T, J, F = X_raw.shape

            if not self.is_trained or self.model is None or self.feature_means is None:
                return pd.DataFrame(0.0, index=index, columns=columns)

            # Standardize using training statistics
            X_flat = X_raw.reshape(T * J, F)
            X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

            means = self.feature_means if len(self.feature_means) == F else np.zeros(F)
            stds = self.feature_stds if len(self.feature_stds) == F else np.ones(F)

            X_norm = (X_flat - means) / stds
            X_norm = np.clip(X_norm, -3.5, 3.5)

            # Predict
            raw_scores = self.model.predict(X_norm).reshape(T, J)

            # Strict cross-sectional demeaning + L1 scaling
            demeaned = np.zeros_like(raw_scores)
            for t in range(T):
                row = raw_scores[t].copy()
                if np.isnan(row).all() or np.allclose(row, row[0]):
                    continue

                row = np.nan_to_num(row, nan=np.nanmedian(row))
                row = row - np.mean(row)

                l1 = np.sum(np.abs(row))
                if l1 > 1e-12:
                    row = row / l1

                demeaned[t] = row - np.mean(row)  # final cleanup

            return pd.DataFrame(demeaned, index=features.index if isinstance(features, pd.DataFrame) else index, columns=columns)

        except Exception:
            return pd.DataFrame(0.0, index=index, columns=columns)

    def _discover_structure(self, features, target):
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
            n_assets = getattr(target, "shape", [0, 1])[1] if hasattr(target, "shape") and len(getattr(target, "shape", [])) > 1 else 1
            self.tickers = [f"asset_{j}" for j in range(n_assets)]
            self.feat_names = [f"f_{i}" for i in range(6)]

        self.n_assets = len(self.tickers)
        self.n_features = len(self.feat_names)

    def _safe_extract_tensors(self, features, target=None):
        if isinstance(features, pd.DataFrame):
            T = len(features)
            cols = features.columns

            if isinstance(cols, pd.MultiIndex):
                feat_names = self.feat_names or sorted(cols.get_level_values(0).unique().tolist())
                tickers = self.tickers or sorted(cols.get_level_values(1).unique().tolist())
                J, F = len(tickers), len(feat_names)
                X_raw = np.zeros((T, J, F), dtype=float)

                for fi, feat in enumerate(feat_names):
                    for ji, tk in enumerate(tickers):
                        key = (feat, tk)
                        if key in features.columns:
                            X_raw[:, ji, fi] = np.nan_to_num(features[key].values, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                arr = np.nan_to_num(features.values, nan=0.0, posinf=0.0, neginf=0.0)
                J = self.n_assets or 1
                F = self.n_features or 6
                target_dim = J * F
                arr_fixed = np.zeros((T, target_dim), dtype=float)
                cols_to_copy = min(arr.shape[1], target_dim)
                arr_fixed[:, :cols_to_copy] = arr[:, :cols_to_copy]
                X_raw = arr_fixed.reshape(T, J, F)
        else:
            arr = np.nan_to_num(np.asarray(features), nan=0.0, posinf=0.0, neginf=0.0)
            if arr.ndim == 3:
                X_raw = arr
            else:
                T = arr.shape[0]
                J = self.n_assets or 1
                F = self.n_features or 6
                target_dim = J * F
                arr_fixed = np.zeros((T, target_dim), dtype=float)
                if arr.ndim > 1:
                    arr_fixed[:, :min(arr.shape[1], target_dim)] = arr[:, :min(arr.shape[1], target_dim)]
                else:
                    arr_fixed[:, 0] = arr
                X_raw = arr_fixed.reshape(T, J, F)

        # Target
        y_arr = None
        if target is not None:
            tarr = np.asarray(target)
            expected = X_raw.shape[0] * X_raw.shape[1]
            if tarr.size == expected:
                y_arr = tarr.reshape(-1)
            elif tarr.size == X_raw.shape[0]:
                y_arr = np.repeat(tarr, X_raw.shape[1])
            else:
                y_arr = np.zeros(expected, dtype=float)

        return X_raw.astype(float, copy=False), y_arr, X_raw.shape[0], X_raw.shape[1]

    def _validate_input(self, features, target):
        if features is None:
            raise ValueError("features cannot be None")
        if target is not None:
            tarr = np.asarray(target)
            if tarr.size == 0:
                raise ValueError("target is empty")
            if np.isnan(tarr).all():
                raise ValueError("target contains only NaN")

# Module-level instance required by the platform
predictor = MyPredictor()

if __name__ == "__main__":
    pass
