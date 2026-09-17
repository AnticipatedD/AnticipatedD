# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from predictor import Predictor


class MyPredictor(Predictor):
    """
    AlphaNova Elite Trading Signal 
    Strategy: Cross-Sectional Momentum + Interaction + Smoothing   
    """
    
    def __init__(self):
        self.is_trained = False
        self.n_assets = None
        self.n_features = None
        
        self.feature_scaler = RobustScaler(quantile_range=(5.0, 95.0))
        
        self.coefficients = None
        self.intercept = None
        self.target_mean = None
        self.target_std = None
        
        self.alpha_smooth = 0.15  # EMA parameter for turnover control
    
    def train(self, features, target):
        try:
            self._validate_input(features, target)
            X_raw, y_raw = self._extract_tensors(features, target)
            X_engineered = self._engineer_features(X_raw)
            X_normalized = self.feature_scaler.fit_transform(X_engineered)
            X_normalized = np.clip(X_normalized, -10, 10)
            self._fit_ridge_regression(X_normalized, y_raw)
            self.is_trained = True
        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}") from e
    
    def _validate_input(self, features, target):
        if features is None or target is None:
            raise ValueError("features and target cannot be None")
        if len(features) != len(target):
            raise ValueError(f"Length mismatch: {len(features)} vs {len(target)}")
        if len(features) < 50:
            raise ValueError(f"Insufficient data: {len(features)} samples")
        if np.isnan(target).any():
            raise ValueError("target contains NaN")
    
    def _extract_tensors(self, features, target):
        if isinstance(features, pd.DataFrame):
            if isinstance(features.columns, pd.MultiIndex):
                feature_names = sorted(features.columns.get_level_values(0).unique().tolist())
                X_list = [features[feat].values for feat in feature_names]
                X_raw = np.stack(X_list, axis=1)
                X_raw = np.transpose(X_raw, (0, 2, 1))
            else:
                X_flat = features.values
                if X_flat.shape[1] % 6 != 0:
                    raise ValueError(f"Column count {X_flat.shape[1]} not divisible by 6")
                J = X_flat.shape[1] // 6
                X_raw = X_flat.reshape(-1, J, 6)
        else:
            X_raw = np.array(features)
        
        y_raw = np.array(target).flatten()
        self.n_assets = X_raw.shape[1]
        self.n_features = X_raw.shape[2]
        return X_raw, y_raw
    
    def _engineer_features(self, X_raw):
        T, J, F = X_raw.shape
        engineered = []
        
        # Base features
        for f in range(F):
            feat = X_raw[:, :, f]
            engineered.append(feat)
            engineered.append(feat - feat.mean(axis=1, keepdims=True))
            rank_pct = np.array([stats.rankdata(feat[t]) / J for t in range(T)])
            engineered.append(rank_pct - 0.5)
        
        # Interactions
        for f1 in range(F):
            for f2 in range(f1 + 1, min(f1 + 3, F)):
                feat1, feat2 = X_raw[:, :, f1], X_raw[:, :, f2]
                engineered.append(feat1 * feat2)
                with np.errstate(divide='ignore', invalid='ignore'):
                    engineered.append(np.where(np.abs(feat2) > 1e-8, feat1 / (np.abs(feat2) + 1e-8), feat1))
        
        # Temporal
        for f in range(F):
            feat = X_raw[:, :, f]
            vol = np.full_like(feat, np.nan)
            for t in range(2, T):
                vol[t] = np.std(feat[max(0, t-2):t+1], axis=0)
            engineered.append(np.nan_to_num(vol, nan=0.0))
            engineered.append(np.diff(feat, axis=0, prepend=0))
        
        X_eng = np.stack(engineered, axis=2)
        X_flat = X_eng.reshape(T * J, -1)
        return np.nan_to_num(X_flat, nan=0.0, posinf=1e3, neginf=-1e3)
    
    def _fit_ridge_regression(self, X_norm, y_raw):
        T_times_J, D = X_norm.shape
        self.target_mean = np.mean(y_raw)
        self.target_std = np.std(y_raw) + 1e-8
        
        # Target expansion across all cross-sectional units if needed
        if len(y_raw) != T_times_J:
            y_expanded = np.repeat(y_raw, self.n_assets) if len(y_raw) * self.n_assets == T_times_J else y_raw
        else:
            y_expanded = y_raw
            
        y_std = (y_expanded - self.target_mean) / self.target_std
        
        lambda_ridge = 10.0 / np.sqrt(D)
        ridge = Ridge(alpha=lambda_ridge, fit_intercept=True, max_iter=10000)
        ridge.fit(X_norm, y_std)
        
        self.coefficients = ridge.coef_
        self.intercept = ridge.intercept_
    
    def predict(self, features):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        try:
            X_raw, _ = self._extract_tensors(features, np.zeros(len(features)))
            T, J = X_raw.shape[0], X_raw.shape[1]
            
            X_eng = self._engineer_features(X_raw)
            X_norm = self.feature_scaler.transform(X_eng)
            X_norm = np.clip(X_norm, -10, 10)
            
            # Correct matrix product for 1D coefficients array
            pred_std = X_norm @ self.coefficients + self.intercept
            pred_raw = pred_std * self.target_std + self.target_mean
            
            signal_raw = pred_raw.reshape(T, J)
            
            # [MANDATORY] First De-meaning
            signal_demeaned = signal_raw - signal_raw.mean(axis=1, keepdims=True)
            residual_mean = np.abs(signal_demeaned.mean(axis=1)).max()
            if residual_mean > 1e-6:
                signal_demeaned -= signal_demeaned.mean(axis=1, keepdims=True)
            
            # Turnover optimization via EMA
            signal_smooth = self._apply_turnover_control(signal_demeaned)
            
            # [MANDATORY] Second De-meaning Enforcement
            signal_final = np.nan_to_num(signal_smooth, nan=0.0)
            signal_final = np.clip(signal_final, -100, 100)
            signal_final -= signal_final.mean(axis=1, keepdims=True)
            
            return signal_final
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}") from e
    
    def _apply_turnover_control(self, signal_raw):
        T, J = signal_raw.shape
        signal_smooth = np.zeros_like(signal_raw)
        signal_smooth[0] = signal_raw[0]
        
        alpha = self.alpha_smooth
        for t in range(1, T):
            signal_smooth[t] = alpha * signal_raw[t] + (1 - alpha) * signal_smooth[t - 1]
        
        for t in range(T):
            std_t = np.std(signal_smooth[t])
            if std_t > 1e-8:
                signal_smooth[t] *= np.std(signal_raw[t]) / std_t
        
        return signal_smooth
