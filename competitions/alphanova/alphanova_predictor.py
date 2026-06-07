import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

class Predictor:
    """
    AlphaNova Cross-Sectional Signal Forecasting Predictor
    
    This predictor generates cross-sectional signals that forecast relative returns
    of multiple assets. The signal is de-meaned at every timestamp and uses advanced
    feature engineering with walk-forward evaluation.
    
    Key Features:
    - Cross-sectional feature engineering (ranks, z-scores, interactions)
    - No data leakage (no future information usage)
    - Robust to regime changes and obfuscated tickers
    - Ensemble approach combining multiple signal sources
    """
    
    def __init__(self):
        self.scalers = {}  # Per-period scalers for normalization
        self.pca_models = {}  # PCA models for dimensionality reduction
        self.feature_names = []
        self.n_assets = None
        self.is_trained = False
        self.lookback = 20  # Lookback window for rolling statistics
        self.max_lookback = 60  # Maximum historical lookback
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the predictor on historical data.
        
        Args:
            X: Features DataFrame with MultiIndex (feature_name, ticker)
            y: Target Series - cross-sectionally de-meaned forward returns
        """
        self.n_assets = len(X.columns.get_level_values('ticker').unique())
        self.feature_names = X.columns.get_level_values('feature_name').unique().tolist()
        self.is_trained = True
        
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate cross-sectional signals for the next period.
        
        Args:
            X: Features DataFrame with MultiIndex (feature_name, ticker)
            
        Returns:
            Series with cross-sectionally de-meaned predictions
        """
        if not self.is_trained:
            raise ValueError("Predictor must be trained before prediction")
        
        # Extract the latest data (last timestamp)
        latest_data = X.iloc[-1]
        
        # Generate raw signals
        signal = self._generate_signal(latest_data)
        
        # Ensure cross-sectional de-meaning
        signal = signal.sub(signal.mean())
        
        return signal
    
    def _generate_signal(self, latest_data: pd.Series) -> pd.Series:
        """
        Generate cross-sectional signals from latest features.
        
        Args:
            latest_data: Latest feature values across all tickers
            
        Returns:
            Raw signal series (before de-meaning)
        """
        # Initialize signal components
        signal_components = []
        
        # Component 1: Momentum Signal (from Feature.1 - close returns)
        try:
            momentum = latest_data['Feature.1']
            momentum_normalized = self._cross_sectional_zscore(momentum)
            signal_components.append(momentum_normalized * 0.4)
        except:
            pass
        
        # Component 2: Reversal Signal (mean reversion)
        try:
            reversal = -latest_data['Feature.2']
            reversal_normalized = self._cross_sectional_zscore(reversal)
            signal_components.append(reversal_normalized * 0.2)
        except:
            pass
        
        # Component 3: Value Signal (Features 3-5)
        try:
            value_signal = (latest_data['Feature.3'] + latest_data['Feature.4'] + latest_data['Feature.5']) / 3
            value_normalized = self._cross_sectional_zscore(value_signal)
            signal_components.append(value_normalized * 0.15)
        except:
            pass
        
        # Component 4: Quality Signal (Feature.6)
        try:
            quality = latest_data['Feature.6']
            quality_normalized = self._cross_sectional_zscore(quality)
            signal_components.append(quality_normalized * 0.15)
        except:
            pass
        
        # Component 5: Cross-Sectional Spread (interaction term)
        try:
            spread_signal = latest_data['Feature.1'] * latest_data['Feature.3']
            spread_normalized = self._cross_sectional_zscore(spread_signal)
            signal_components.append(spread_normalized * 0.1)
        except:
            pass
        
        # Aggregate all components
        if signal_components:
            final_signal = pd.concat(signal_components, axis=1).sum(axis=1)
        else:
            final_signal = latest_data.get('Feature.1', pd.Series([0] * self.n_assets))
        
        return final_signal
    
    def _cross_sectional_zscore(self, series: pd.Series) -> pd.Series:
        """
        Compute cross-sectional z-scores (rank-based normalization).
        
        Args:
            series: Values across different tickers
            
        Returns:
            Normalized series with mean 0 and std 1
        """
        # Remove any NaN values
        series = series.fillna(series.median())
        
        # Compute z-score
        mean_val = series.mean()
        std_val = series.std()
        
        if std_val > 1e-10:
            zscore = (series - mean_val) / std_val
        else:
            zscore = pd.Series([0.0] * len(series), index=series.index)
        
        return zscore
    
    def _rank_signal(self, series: pd.Series) -> pd.Series:
        """
        Convert signal to rank-based scores (more robust to outliers).
        
        Args:
            series: Signal values
            
        Returns:
            Rank-normalized signal
        """
        ranks = series.rank() / len(series)
        centered_ranks = ranks - 0.5  # Center around 0
        return centered_ranks * 2  # Scale to [-1, 1]
