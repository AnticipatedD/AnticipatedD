# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy"
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

from predictor import Predictor   # provided by the competition


class MyPredictor(Predictor):
    """
    Cross-sectional momentum + nonlinear interactions + EMA turnover control.
    Target: positive Sharpe (0.15–0.25 range under realistic costs).
    """

    def __init__(self):
        self.is_trained = False
        self.n_assets = None
        self.n_features = None

        self.feature_scaler = RobustScaler(quantile_range=(5.0, 95.0))
        self.coefficients = None
        self.intercept = 0.0
        self.target_mean = 0.0
        self.target_std = 1.0

        # EMA smoothing – critical for positive Sharpe after 5 bp costs
        self.alpha_smooth = 0.15

    def train(self, features, target):
        """Train in < 240 s."""
        self._validate_input(features, target)
        X_raw, y_raw = self._extract_tensors(features, target)

        X_eng = self._engineer_features(X_raw)
        X_norm = self.feature_scaler.fit_transform(X_eng)
        X_norm = np.clip(X_norm, -10.0, 10.0)

        self._fit_ridge(X_norm, y_raw)
        self.is_trained = True

    def predict(self, features):
        """Produce de-meaned signal in < 60 s."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predict()")

        X_raw, _ = self._extract_tensors(features, np.zeros(len(features)))
        X_eng = self._engineer_features(X_raw)
        X_norm = self.feature_scaler.transform(X_eng)
        X_norm = np.clip(X_norm, -10.0, 10.0)

        # Linear prediction
        pred_std = X_norm @ self.coefficients + self.intercept
        pred = pred_std * self.target_std + self.target_mean

        T, J = X_raw.shape[0], X_raw.shape[1]
        signal = pred.reshape(T, J)

        # First mandatory de-meaning
        signal = signal - signal.mean(axis=1, keepdims=True)

        # Turnover control (EMA) – the main source of positive Sharpe
        signal = self._apply_ema_smoothing(signal)

        # Second mandatory de-meaning + numerical guards
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        signal = np.clip(signal, -100.0, 100.0)
        signal = signal - signal.mean(axis=1, keepdims=True)

        return signal

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_input(self, features, target):
        if features is None or target is None:
            raise ValueError("features / target cannot be None")
        if len(features) != len(target):
            raise ValueError("Length mismatch between features and target")
        if len(features) < 50:
            raise ValueError("Need at least 50 samples")
        if np.isnan(np.asarray(target)).any():
            raise ValueError("target contains NaN")

    def _extract_tensors(self, features, target):
        """Return X_raw of shape (T, J, F) and y of shape (T,)."""
        if isinstance(features, pd.DataFrame):
            if isinstance(features.columns, pd.MultiIndex):
                feat_names = sorted(features.columns.get_level_values(0).unique())
                X_list = [features[f].values for f in feat_names]
                X_raw = np.stack(X_list, axis=-1)          # (T, J, F)
            else:
                X_flat = features.values
                if X_flat.shape[1] % 6 != 0:
                    raise ValueError("Number of columns not divisible by 6")
                J = X_flat.shape[1] // 6
                X_raw = X_flat.reshape(-1, J, 6)
        else:
            X_raw = np.asarray(features)

        y = np.asarray(target).ravel()
        self.n_assets = X_raw.shape[1]
        self.n_features = X_raw.shape[2]
        return X_raw, y

    def _engineer_features(self, X_raw):
        """
        ~50-60 engineered features:
        - level, cross-sectional deviation, rank
        - pairwise products & ratios
        - short-term volatility & momentum
        """
        T, J, F = X_raw.shape
        feats = []

        # ---- Layer 1: base transforms (3 × F) ----
        for f in range(F):
            x = X_raw[:, :, f]
            feats.append(x)                                      # level
            feats.append(x - x.mean(axis=1, keepdims=True))      # deviation
            ranks = np.array([stats.rankdata(x[t]) for t in range(T)])
            feats.append(ranks / J - 0.5)                        # centered rank

        # ---- Layer 2: pairwise interactions (limited to keep D reasonable) ----
        for f1 in range(F):
            for f2 in range(f1 + 1, min(f1 + 3, F)):
                a, b = X_raw[:, :, f1], X_raw[:, :, f2]
                feats.append(a * b)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(np.abs(b) > 1e-8, a / (np.abs(b) + 1e-8), a)
                feats.append(ratio)

        # ---- Layer 3: temporal features ----
        for f in range(F):
            x = X_raw[:, :, f]
            # 3-period rolling std
            vol = np.zeros_like(x)
            for t in range(2, T):
                vol[t] = np.std(x[t-2:t+1], axis=0)
            feats.append(vol)
            # 1-period momentum
            mom = np.diff(x, axis=0, prepend=0)
            feats.append(mom)

        X_eng = np.stack(feats, axis=2)               # (T, J, D_eng)
        X_flat = X_eng.reshape(T, -1)                 # (T, J*D_eng)
        return np.nan_to_num(X_flat, nan=0.0, posinf=1e3, neginf=-1e3)

    def _fit_ridge(self, X_norm, y_raw):
        """Ridge with adaptive L2 strength."""
        self.target_mean = float(np.mean(y_raw))
        self.target_std = float(np.std(y_raw) + 1e-8)
        y_std = (y_raw - self.target_mean) / self.target_std

        D = X_norm.shape[1]
        alpha = 10.0 / np.sqrt(D)          # adaptive regularisation

        model = Ridge(alpha=alpha, fit_intercept=True, max_iter=10000)
        model.fit(X_norm, y_std)

        self.coefficients = model.coef_
        self.intercept = model.intercept_

    def _apply_ema_smoothing(self, signal):
        """
        Exponential smoothing that dramatically reduces turnover.
        This step is usually the difference between Sharpe ≈ 0 and Sharpe > 0.15.
        """
        T, J = signal.shape
        smooth = np.zeros_like(signal)
        smooth[0] = signal[0]

        a = self.alpha_smooth
        for t in range(1, T):
            smooth[t] = a * signal[t] + (1.0 - a) * smooth[t - 1]

        # Optional: re-scale each cross-section to keep volatility similar
        for t in range(T):
            s_raw = np.std(signal[t])
            s_sm = np.std(smooth[t])
            if s_sm > 1e-8 and s_raw > 1e-8:
                smooth[t] *= s_raw / s_sm

        return smooth
