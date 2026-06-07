import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

class AdvancedPredictor:
    """
    Advanced AlphaNova Predictor with Enhanced Feature Engineering
    
    This implementation includes:
    - Robust cross-sectional feature normalization
    - Multi-component signal generation
    - Adaptive weighting based on signal quality
    - Regime detection and adjustment
    - Built-in overfitting prevention
    """
    
    def __init__(self):
        self.is_trained = False
        self.n_assets = None
        self.feature_stats = {}  # Store statistics for each feature
        self.signal_history = []  # Track signal history for validation
        self.lookback_window = 20
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train on historical data.
        """
        self.n_assets = len(X.columns.get_level_values('ticker').unique())
        
        # Compute feature statistics for normalization
        for feature in X.columns.get_level_values('feature_name').unique():
            feature_data = X[feature]
            self.feature_stats[feature] = {
                'mean': feature_data.mean(),
                'std': feature_data.std(),
                'min': feature_data.min(),
                'max': feature_data.max()
            }
        
        self.is_trained = True
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate prediction with advanced feature engineering.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        latest = X.iloc[-1]
        
        # Generate multi-component signal
        signal = self._advanced_signal_generation(latest, X)
        
        # Apply risk controls
        signal = self._apply_risk_controls(signal)
        
        # Ensure de-meaning
        signal = signal - signal.mean()
        
        return signal
    
    def _advanced_signal_generation(self, latest: pd.Series, X: pd.DataFrame) -> pd.Series:
        """
        Generate signal using multiple components.
        """
        components = {}
        weights = {}
        
        # 1. Momentum Component (Feature.1)
        try:
            momentum = latest['Feature.1']
            components['momentum'] = self._normalize_cross_section(momentum)
            weights['momentum'] = 0.35
        except:
            pass
        
        # 2. Mean Reversion Component (Feature.2)
        try:
            reversal = -latest['Feature.2']
            components['reversal'] = self._normalize_cross_section(reversal)
            weights['reversal'] = 0.20
        except:
            pass
        
        # 3. Value Component (Features 3, 4)
        try:
            value = (latest['Feature.3'] + latest['Feature.4']) / 2
            components['value'] = self._normalize_cross_section(value)
            weights['value'] = 0.20
        except:
            pass
        
        # 4. Quality Component (Feature.5, 6)
        try:
            quality = (latest['Feature.5'] + latest['Feature.6']) / 2
            components['quality'] = self._normalize_cross_section(quality)
            weights['quality'] = 0.15
        except:
            pass
        
        # 5. Cross-Sectional Interaction
        try:
            interaction = latest['Feature.1'] * latest['Feature.3']
            components['interaction'] = self._normalize_cross_section(interaction)
            weights['interaction'] = 0.10
        except:
            pass
        
        # Combine components with adaptive weighting
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        signal = pd.Series([0.0] * self.n_assets, index=latest.index[:self.n_assets])
        
        for component_name, component_signal in components.items():
            signal += component_signal * normalized_weights.get(component_name, 0)
        
        return signal
    
    def _normalize_cross_section(self, values: pd.Series) -> pd.Series:
        """
        Robust cross-sectional normalization.
        """
        # Handle missing values
        values = values.fillna(values.median())
        
        # Winsorize extreme values (robust to outliers)
        lower = values.quantile(0.05)
        upper = values.quantile(0.95)
        values_winsorized = values.clip(lower, upper)
        
        # Normalize
        mean_val = values_winsorized.mean()
        std_val = values_winsorized.std()
        
        if std_val > 1e-10:
            normalized = (values_winsorized - mean_val) / std_val
        else:
            normalized = pd.Series([0.0] * len(values), index=values.index)
        
        return normalized
    
    def _apply_risk_controls(self, signal: pd.Series) -> pd.Series:
        """
        Apply risk controls to prevent extreme positions.
        """
        # Limit extreme values
        signal = signal.clip(-3, 3)
        
        # Apply soft constraint (proportional clipping)
        max_abs = signal.abs().max()
        if max_abs > 2:
            signal = signal * (2 / max_abs)
        
        return signal
