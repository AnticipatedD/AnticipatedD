# /// script
# dependencies = [
#   "numpy>=1.19.0",
#   "pandas>=1.2.0",
#   "scipy>=1.6.0",
#   "scikit-learn>=0.24.0",
# ]
# ///

"""
AlphaNova Biweekly Competition — Season 1, Cycle 1
Cross-sectional momentum + simple feature interactions + Ridge
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from scipy import stats

warnings.filterwarnings("ignore")


class Predictor:
    def train(self, features, target):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError


class MyPredictor(Predictor):
    def __init__(self):
        self.is_trained = False
        self.feature_scaler = None
        self.coefficients = None
        self.intercept = None
        self.target_mean = 0.0
        self.target_std = 1.0
        self.n_assets = None
        self.alpha_smooth = 0.20          # light EMA to reduce turnover
        self.last_signal = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, features, target):
        X_raw, y_raw = self._extract_tensors(features, target)
        X_eng = self._engineer_features(X_raw)
        X_norm = self._normalize_features(X_eng, fit=True)
        self._fit_ridge(X_norm, y_raw)
        self.is_trained = True

    def predict(self, features):
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        X_raw, _ = self._extract_tensors(features, None)
        X_eng = self._engineer_features(X_raw)
        X_norm = self._normalize_features(X_eng, fit=False)

        pred = X_norm @ self.coefficients + self.intercept
        pred = pred * self.target_std + self.target_mean

        T, J = X_raw.shape[0], X_raw.shape[1]
        signal = pred.reshape(T, J)

        # Strict cross-sectional demeaning
        signal = signal - signal.mean(axis=1, keepdims=True)

        # Light turnover control
        signal = self._smooth(signal)

        # Final safety
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        signal = signal - signal.mean(axis=1, keepdims=True)
        return signal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_tensors(self, features, target):
        """Return X of shape (T, J, 6) and y of shape (T,)."""
        if isinstance(features, pd.DataFrame):
            if isinstance(features.columns, pd.MultiIndex):
                # MultiIndex: level 0 = feature name, level 1 = ticker
                feat_names = sorted(features.columns.get_level_values(0).unique())
                X_list = [features[f].values for f in feat_names]
                X = np.stack(X_list, axis=-1)          # (T, J, F)
            else:
                vals = features.values
                if vals.shape[1] % 6 != 0:
                    raise ValueError(f"Columns {vals.shape[1]} not divisible by 6")
                J = vals.shape[1] // 6
                X = vals.reshape(-1, J, 6)
        else:
            X = np.asarray(features)
            if X.ndim == 2:
                if X.shape[1] % 6 != 0:
                    raise ValueError(f"Columns {X.shape[1]} not divisible by 6")
                J = X.shape[1] // 6
                X = X.reshape(-1, J, 6)
            elif X.ndim != 3 or X.shape[2] != 6:
                raise ValueError(f"Expected (T, J, 6), got {X.shape}")

        self.n_assets = X.shape[1]

        if target is None:
            y = np.zeros(X.shape[0])
        else:
            y = np.asarray(target).ravel()
            if len(y) != X.shape[0]:
                raise ValueError("features and target length mismatch")

        return X.astype(np.float64), y.astype(np.float64)

    def _engineer_features(self, X):
        """Simple, fast cross-sectional features. Output shape (T, J * n_eng)."""
        T, J, F = X.shape
        feats = []

        for f in range(F):
            x = X[:, :, f]                              # (T, J)

            # raw
            feats.append(x)

            # demeaned (relative strength)
            feats.append(x - x.mean(axis=1, keepdims=True))

            # rank (centered)
            ranks = np.apply_along_axis(
                lambda row: stats.rankdata(row) / J - 0.5, 1, x
            )
            feats.append(ranks)

            # 1-period momentum
            mom = np.diff(x, axis=0, prepend=x[:1])
            feats.append(mom)

        # a few cheap interactions
        for i in range(min(3, F)):
            for j in range(i + 1, min(i + 2, F)):
                feats.append(X[:, :, i] * X[:, :, j])

        X_eng = np.stack(feats, axis=-1)                # (T, J, n_eng)
        X_flat = X_eng.reshape(T, -1)
        return np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_features(self, X, fit=False):
        if fit or self.feature_scaler is None:
            self.feature_scaler = RobustScaler(quantile_range=(5.0, 95.0))
            Xn = self.feature_scaler.fit_transform(X)
        else:
            Xn = self.feature_scaler.transform(X)
        return np.clip(Xn, -8.0, 8.0)

    def _fit_ridge(self, X, y):
        self.target_mean = float(np.mean(y))
        self.target_std = float(np.std(y) + 1e-8)
        y_std = (y - self.target_mean) / self.target_std

        # moderate regularization
        alpha = max(1.0, 5.0 / np.sqrt(X.shape[1]))
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X, y_std)

        self.coefficients = model.coef_
        self.intercept = float(model.intercept_)

    def _smooth(self, signal):
        """Simple EMA across time to cut turnover."""
        T, J = signal.shape
        out = np.empty_like(signal)
        out[0] = signal[0]
        a = self.alpha_smooth
        for t in range(1, T):
            out[t] = a * signal[t] + (1.0 - a) * out[t - 1]

        # re-scale each cross-section to keep roughly the same magnitude
        for t in range(T):
            s = np.std(out[t])
            if s > 1e-8:
                out[t] *= np.std(signal[t]) / s
        return out


# Optional local smoke-test helper (not required by the portal)
def validate_submission(features, target):
    pred = MyPredictor()
    pred.train(features, target)
    signal = pred.predict(features)
    assert signal.shape[0] == len(features)
    assert np.allclose(signal.mean(axis=1), 0.0, atol=1e-6)
    print("Local validation OK – shape", signal.shape)
    return pred, signal


if __name__ == "__main__":
    print("MyPredictor ready. Use runner.py or the portal to score.")
