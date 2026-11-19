# /// script
# dependencies = ["xgboost", "numpy", "pandas"]
# ///

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V14Predictor(Predictor):
    """
    13-Signal Orthogonal Alpha Predictor with 70% XGBoost + 30% Closed-Form Ridge Blend,
    EMA Smoothing/De-tilting, and Cross-Sectional Standardization.
    """

    def __init__(self):
        self.gbdt_model = None
        self.ridge_w = None
        self.feat_mean = None
        self.feat_std = None
        self.num_signals = 13
        self.ridge_alpha = 5000.0
        self.ema_span = 45
        self.tilt_span = 450

    def _unpack_features(self, X):
        """Extracts F1..F6 as 2D numpy arrays of shape (T, N) from dict, list/tuple, or ndarray."""
        if isinstance(X, dict):
            keys = ["F1", "F2", "F3", "F4", "F5", "F6"]
            if all(k in X for k in keys):
                mats = [np.asarray(X[k], dtype=np.float64) for k in keys]
            else:
                sorted_keys = sorted(X.keys())[:6]
                mats = [np.asarray(X[k], dtype=np.float64) for k in sorted_keys]
            return mats

        if isinstance(X, (list, tuple)) and len(X) == 6:
            return [np.asarray(m, dtype=np.float64) for m in X]

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 3:
            if arr.shape[2] == 6:
                return [arr[:, :, i] for i in range(6)]
            if arr.shape[0] == 6:
                return [arr[i, :, :] for i in range(6)]

        raise ValueError("Unsupported feature container format; expected 6 feature matrices (F1..F6).")

    @staticmethod
    def _csrank(M):
        """Normalized cross-sectional rank in [-1, 1] matching MATLAB tiedrank / pandas rank(axis=1)."""
        df = pd.DataFrame(M)
        counts = df.notna().sum(axis=1)
        rk = df.rank(axis=1, method="average", na_option="keep")
        denom = np.maximum(counts - 1.0, 1.0)
        norm_rk = rk.sub(0.5 * (counts + 1.0), axis=0).div(0.5 * denom, axis=0)
        return norm_rk.to_numpy(dtype=np.float64)

    @staticmethod
    def _cs_robust_z(M):
        """Cross-sectional median/MAD robust z-score clipped to [-4, 4]."""
        med = np.nanmedian(M, axis=1, keepdims=True)
        mad_val = np.nanmedian(np.abs(M - med), axis=1, keepdims=True)
        mad_val = np.where((mad_val == 0.0) | ~np.isfinite(mad_val), 1.0, mad_val)
        z = (M - med) / (mad_val * 1.4826)
        return np.clip(z, -4.0, 4.0)

    @staticmethod
    def _ewm_mean(M, span):
        """Matches pandas .ewm(span=span, min_periods=1).mean()."""
        return pd.DataFrame(M).ewm(span=span, min_periods=1).mean().to_numpy(dtype=np.float64)

    def _build_13_signals(self, F1, F2, F3, F4, F5, F6):
        """Constructs the 13 orthogonal alpha signals of shape (T, N, 13)."""
        T, N = F1.shape
        R1 = self._csrank(F1)
        R2 = self._csrank(F2)
        R3 = self._csrank(F3)
        R4 = self._csrank(F4)
        R5 = self._csrank(F5)
        R6 = self._csrank(F6)

        s24 = F2 - F4
        s12 = F1 - F2
        s13 = F1 - F3

        # Signal 12: 5-bar difference of s24
        diff5_s24 = np.zeros_like(s24)
        if T > 5:
            diff5_s24[5:, :] = s24[5:, :] - s24[:-5, :]

        # Signal 13: Mean-reversion against 12-bar EMA of F1
        ema12_F1 = self._ewm_mean(F1, span=12)

        X_3d = np.zeros((T, N, self.num_signals), dtype=np.float64)
        X_3d[:, :, 0] = self._csrank(s24)
        X_3d[:, :, 1] = self._csrank(s12)
        X_3d[:, :, 2] = self._csrank(s13)
        X_3d[:, :, 3] = (R2 - R3) * R6
        X_3d[:, :, 4] = (R2 - R4) * R5
        X_3d[:, :, 5] = R5 * R6
        X_3d[:, :, 6] = R6
        X_3d[:, :, 7] = R5
        X_3d[:, :, 8] = self._csrank(F1 + F4) * self._csrank(np.abs(F6))
        X_3d[:, :, 9] = np.tanh(2.0 * R3) * (R5 - R2)
        X_3d[:, :, 10] = self._csrank(self._cs_robust_z(F1)) - R6
        X_3d[:, :, 11] = self._csrank(diff5_s24)
        X_3d[:, :, 12] = -self._csrank(F1 - ema12_F1) * R3

        return X_3d

    def fit(self, X, y):
        F1, F2, F3, F4, F5, F6 = self._unpack_features(X)
        X_3d = self._build_13_signals(F1, F2, F3, F4, F5, F6)

        X_flat = X_3d.reshape(-1, self.num_signals)
        y_flat = np.asarray(y, dtype=np.float64).reshape(-1)

        valid_mask = np.isfinite(y_flat) & np.all(np.isfinite(X_flat), axis=1)
        X_train = X_flat[valid_mask]
        y_train = y_flat[valid_mask]

        # 1. Gradient Boosted Trees (70% weight)
        self.gbdt_model = xgb.XGBRegressor(
            n_estimators=65,
            max_depth=3,
            learning_rate=0.06,
            min_child_weight=350,
            colsample_bytree=0.55,
            subsample=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self.gbdt_model.fit(X_train, y_train)

        # 2. Closed-Form L2 Ridge Regression (30% weight)
        self.feat_mean = np.mean(X_train, axis=0)
        self.feat_std = np.std(X_train, axis=0, ddof=1)
        self.feat_std = np.where(self.feat_std == 0.0, 1.0, self.feat_std)

        X_train_std = (X_train - self.feat_mean) / self.feat_std
        y_train_ctr = y_train - np.mean(y_train)

        gram = (X_train_std.T @ X_train_std) + self.ridge_alpha * np.eye(self.num_signals)
        self.ridge_w = np.linalg.solve(gram, X_train_std.T @ y_train_ctr)

        return self

    def predict(self, X):
        F1, F2, F3, F4, F5, F6 = self._unpack_features(X)
        T, N = F1.shape

        X_3d = self._build_13_signals(F1, F2, F3, F4, F5, F6)
        X_val = X_3d.reshape(-1, self.num_signals)
        X_val = np.where(np.isfinite(X_val), X_val, 0.0)

        # Out-of-sample predictions
        raw_gbdt = self.gbdt_model.predict(X_val)
        X_val_std = (X_val - self.feat_mean) / self.feat_std
        raw_ridge = X_val_std @ self.ridge_w

        # Scale-match before 70/30 blend
        gbdt_norm = (raw_gbdt - np.mean(raw_gbdt)) / (np.std(raw_gbdt) + 1e-8)
        ridge_norm = (raw_ridge - np.mean(raw_ridge)) / (np.std(raw_ridge) + 1e-8)
        raw_blend = 0.70 * gbdt_norm + 0.30 * ridge_norm

        pred = raw_blend.reshape(T, N)

        # EMA Smoothing (span = 45) & Lagged Drift De-tilting (span = 450)
        pred_smooth = self._ewm_mean(pred, span=self.ema_span)
        pred_lagged = np.vstack([np.zeros((1, N), dtype=np.float64), pred_smooth[:-1, :]])
        tilt = self._ewm_mean(pred_lagged, span=self.tilt_span)
        pred = pred_smooth - tilt

        # Cross-sectional Z-Score, [-4, 4] Clip, and Mandatory De-meaning
        cs_mean = np.mean(pred, axis=1, keepdims=True)
        cs_std = np.std(pred, axis=1, ddof=1, keepdims=True)
        cs_std = np.where((cs_std == 0.0) | ~np.isfinite(cs_std), 1.0, cs_std)

        pred = (pred - cs_mean) / cs_std
        pred = np.clip(pred, -4.0, 4.0)
        pred = pred - np.mean(pred, axis=1, keepdims=True)

        return pred
