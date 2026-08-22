# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy"
# ]
# ///

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from predictor import Predictor

class MyPredictor(Predictor):
"""
AlphaNova Predictor Base Class
==============================
Abstract base class that all submissions must inherit from.
Defines the interface contract for train() and predict() methods.

DO NOT MODIFY THIS FILE.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Predictor(ABC):
    """
    Abstract base class for AlphaNova cross-sectional signal predictors.
    
    All submissions must:
    1. Inherit from this class
    2. Implement train(features, target)
    3. Implement predict(features)
    4. Ensure predict() returns cross-sectionally de-meaned signals
    5. Place all logic inside the class (no global functions/state)
    
    Interface Contract:
    - train(features, target): Learn from historical data
      * features: pd.DataFrame with MultiIndex (feature, ticker) or shape (T, J*6)
      * target: pd.Series of shape (T,), z-scored and clipped at ±5
      
    - predict(features): Generate forward-looking signal
      * features: pd.DataFrame, same format as training
      * return: np.ndarray or pd.DataFrame, shape (T, J), where sum per row ≈ 0
    """
    
    @abstractmethod
    def train(self, features, target):
        """
        Train the predictor on historical cross-sectional data.
        
        Args:
            features: pd.DataFrame
                Input features. Can be:
                - MultiIndex: columns = (feature_name, ticker), rows = timestamps
                - Flat: shape (T, J*F) where T=periods, J=assets=20, F=features=6
                
            target: pd.Series
                Forward-looking target, z-scored and clipped at ±5.
                Shape: (T,), one scalar per timestamp (cross-sectional, not per-asset).
        
        Returns:
            None (trains in-place)
        
        Constraints:
            - Must complete within 4 minutes (240 seconds)
            - Memory must stay under 8 GB
            - Must not raise exceptions on valid input
        """
        pass
    
    @abstractmethod
    def predict(self, features):
        """
        Generate cross-sectionally de-meaned trading signal.
        
        Args:
            features: pd.DataFrame
                Same format as training data.
        
        Returns:
            Signal: np.ndarray or pd.Series
                Shape: (T, J) for array, or (T,) for Series with MultiIndex
                MUST satisfy: ∑_j P_j(t) ≈ 0 for every timestamp t
                (cross-sectional de-meaning is mandatory)
        
        Constraints:
            - Must complete within 60 seconds
            - Memory must stay under 8 GB
            - Must return de-meaned output (checked by overfitting gate)
            - Must handle NaN/missing values gracefully
            - No look-ahead bias (use only t and past, not future)
        """
        pass
