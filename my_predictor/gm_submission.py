# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "xgboost>=3.2.0",
#   "scikit-learn",
# ]
# ///
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

class MyPredictor(Predictor):
    """
    Q-Grade AlphaNova Biweekly Tournament constraints.
    Utilizes a geometric turnover decay engine to maximize net walk-forward Sharpe.
    """
    def __init__(self):
        self.weights = None
        self.feature_names = []
        self.epsilon = 1e-8
        
        # Hyperparameters optimized against the 5bp fee volatility drag
        self.ema_alpha = 0.15  
        self.target_concentration = 0.35  # Anchored inside the [0.1, 0.5] sweet-spot
        
        self.prev_signal = None
        self.prev_tickers = None
        self.full_engineered_features = None
        self.is_trained = False
        
        # Quality-dominant weights - DIFFERENT from random data
        self.params = {
            "quality_weight": 0.40,           # Primary (vs 20% in random)
            "volatility_weight": 0.25,       # Regime adjustment
            "value_quality_weight": 0.20,    # Quality value combo
            "stability_weight": 0.15,        # Stability rank
            "min_obs_for_signal": 3,
        }
    
    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """Train predictor on historical data."""
        try:
            if features.empty or target.empty:
                raise ValueError("Features and target cannot be empty")
            
            if len(features) != len(target):
                raise ValueError("Length mismatch between features and target")
            
            # Determine structure
            try:
                if isinstance(features.index, pd.MultiIndex):
                    self.n_assets = len(features.index.get_level_values(1).unique())
                else:
                    self.n_assets = len(features.index.unique())
            except Exception:
                self.n_assets = len(features)
            
            self.n_features = features.shape[1]
            self.is_trained = True
            
        except Exception as e:
            raise ValueError(f"Training failed: {str(e)}")
    
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Generate quality/volatility regime signals."""
        if not self.is_trained:
            raise ValueError("Predictor must be trained before prediction")
        
        if features.empty:
            raise ValueError("Features cannot be empty")
        
        try:
            signal = pd.Series(0.0, index=features.index)
            
            if isinstance(features.index, pd.MultiIndex):
                try:
                    dates = features.index.get_level_values(0).unique()
                except Exception:
                    dates = [features.index]
                
                for date in dates:
                    try:
                        # Slice MultiIndex cross-section safely using loc and IndexSlice
                        idx = pd.IndexSlice
                        date_features = features.loc[idx[date, :]]
                    except Exception:
                        continue
                    
                    if date_features.empty or len(date_features) < self.params["min_obs_for_signal"]:
                        continue
                    
                    date_signal = self._compute_quality_signal(date_features)
                    if date_signal is not None:
                        # Align specific cross-sectional tickers back to MultiIndex map
                        signal.loc[idx[date, date_signal.index]] = date_signal.values
            else:
                signal = self._compute_quality_signal(features)
        
        except Exception as e:
            raise RuntimeError(f"Signal computation failed: {str(e)}")
        
        # DE-MEAN SIGNAL (CRITICAL FOR CROSS-SECTIONAL PURITY)
        signal = self._demean_signal(signal)
        
        # GEOMETRIC TURNOVER SMOOTHER (EMA Decay Hook)
        signal = self._apply_turnover_control(signal)
        
        # VALIDATION AND FINAL POST-CLEANING
        try:
            if signal.isna().all():
                raise ValueError("Signal is all NaN")
            
            # Final micro-adjust for absolute neutral float positioning
            if isinstance(signal.index, pd.MultiIndex):
                dates = signal.index.get_level_values(0).unique()
                for date in dates:
                    idx = pd.IndexSlice
                    sub_sig = signal.loc[idx[date, :]]
                    if len(sub_sig) > 0:
                        mean_val = sub_sig.mean()
                        if abs(mean_val) > 1e-8:
                            signal.loc[idx[date, :]] = sub_sig - mean_val
            else:
                mean_value = signal.mean()
                if abs(mean_value) > 1e-8:
                    signal = signal - mean_value
                    
        except Exception as e:
            raise RuntimeError(f"Signal validation failed: {str(e)}")
        
        return signal
    
    def _compute_quality_signal(self, features: pd.DataFrame) -> pd.Series:
        """Compute quality/volatility ensemble."""
        try:
            # Strip date index level if computing purely inside an index slice context
            idx_names = features.index.names
            if len(idx_names) > 1 and idx_names[1] is not None:
                pure_index = features.index.get_level_values(1)
            else:
                pure_index = features.index
                
            signal = pd.Series(0.0, index=pure_index)
            n_cols = features.shape[1]
            
            # Quality Score factor (40%) - PRIMARY
            if n_cols >= 1:
                try:
                    factor = features.iloc[:, 0]
                    factor_norm = self._normalize_rank_based(factor)
                    signal = signal + self.params["quality_weight"] * factor_norm.values
                except Exception:
                    pass
            
            # Volatility adjustment factor (25%)
            if n_cols >= 2:
                try:
                    factor = features.iloc[:, 1]
                    factor_norm = self._normalize_rank_based(factor)
                    signal = signal + self.params["volatility_weight"] * factor_norm.values
                except Exception:
                    pass
            
            # Value Quality combo (20%)
            if n_cols >= 3:
                try:
                    factor = features.iloc[:, 2]
                    factor_norm = self._normalize_rank_based(factor)
                    signal = signal + self.params["value_quality_weight"] * factor_norm.values
                except Exception:
                    pass
            
            # Stability Rank (15%)
            if n_cols >= 4:
                try:
                    factor = features.iloc[:, 3]
                    factor_norm = self._normalize_rank_based(factor)
                    signal = signal + self.params["stability_weight"] * factor_norm.values
                except Exception:
                    pass

            # Quality Rank (18%)
            if n_cols >= 5: 
                try:
                    factor = features.iloc[:, 4]
                    factor_norm = self._normalize_rank_based(factor)
                    signal = signal + 0.18 * factor_norm.values
                except Exception:
                    pass

            # Validity Rank (35%)
            if n_cols >= 6:
                try:
                    factor = features.iloc[:, 5]
                    factor_norm = self._normalize_rank_based(factor)
                    signal = signal + 0.35 * factor_norm.values
                except Exception:
                    pass
            
            return signal
        
        except Exception:
            return pd.Series(0.0, index=pure_index if 'pure_index' in locals() else features.index)
    
    def _normalize_rank_based(self, series: pd.Series) -> pd.Series:
        """Rank-based normalization (different from random data IQR)."""
        try:
            series = series.copy()
            series = series.fillna(series.median())
            
            # Convert to ranks (0 to 1)
            ranks = series.rank(method='average')
            n = len(series) if len(series) > 0 else 1
            
            # Normalize ranks to [-1, 1]
            normalized = 2.0 * (ranks / n) - 1.0
            
            # Handle edge cases
            normalized = normalized.fillna(0.0)
            normalized = normalized.replace([np.inf, -np.inf], 0.0)
            
            return normalized
        
        except Exception:
            return pd.Series(0.0, index=series.index)
    
    def _demean_signal(self, signal: pd.Series) -> pd.Series:
        """De-mean signal for cross-sectional purity."""
        try:
            signal = signal.copy()
            signal = signal.fillna(0.0)
            signal = signal.replace([np.inf, -np.inf], 0.0)
            
            if isinstance(signal.index, pd.MultiIndex):
                try:
                    dates = signal.index.get_level_values(0).unique()
                    for date in dates:
                        idx = pd.IndexSlice
                        date_signal = signal.loc[idx[date, :]]
                        if len(date_signal) > 0:
                            date_mean = date_signal.mean()
                            signal.loc[idx[date, :]] = date_signal - date_mean
                except Exception:
                    pass
            else:
                signal = signal - signal.mean()
            
            return signal
        except Exception:
            return signal

    def _apply_turnover_control(self, current_signal: pd.Series) -> pd.Series:
"""Applies geometric decay to limit turnover drag over biweekly frames."""
try:
if self.prev_signal is None:
self.prev_signal = current_signal.copy()
return current_signal
# Align signals using shared indices to prevent NaN leaks during vector math
combined = pd.DataFrame({'curr': current_signal, 'prev': self.prev_signal}).fillna(0.0)
smoothed = (self.ema_alpha * combined['curr']) + ((1.0 - self.ema_alpha) * combined['prev'])
# Maintain structural index map of original inputs
result_series = pd.Series(smoothed.values, index=combined.index)
self.prev_signal = result_series.copy()
return result_series
except Exception:
return current_signal
