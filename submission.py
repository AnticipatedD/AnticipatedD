"""
AlphaNova Submission: submission.py
===================================
Strategy: Multi-factor Cross-Sectional Signal (Momentum Dominant)
Approach: Ensemble of momentum, mean-reversion, and quality signals
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import RobustScaler


class Predictor:
    """
    Multi-Factor Cross-Sectional Signal Generator for AlphaNova
    Strategy Name: amonRa Momentum Ensemble
    """

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self):
        """
        Initialize the predictor with default parameters and state.
        """
        self.is_trained = False
        self.n_assets = None
        self.n_features = None
        self.feature_names = None
        self.scaler = RobustScaler()  # Robust to outliers using IQR

        # Factor weights (sum to 1.0 for proper weighting)
        self.params = {
            "momentum_weight": 0.30,      # Primary signal
            "reversal_weight": 0.25,       # Contrarian signal
            "value_weight": 0.25,          # Value signal
            "quality_weight": 0.20,        # Quality signal
            "min_obs_for_signal": 3,       # Minimum assets for valid cross-section
        }

    def _compute_feature_statistics(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Helper method to handle missing entries and structure feature data
        without creating look-ahead bias.
        """
        df_filled = features.copy()
        for col in df_filled.columns:
            median_val = df_filled[col].median()
            if pd.isna(median_val):
                df_filled[col] = df_filled[col].fillna(0.0)
            else:
                df_filled[col] = df_filled[col].fillna(median_val)
        return df_filled

    # =====================================================
    # TRAINING METHOD
    # =====================================================

    def train(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Train the predictor on historical cross-sectional data.
        """
        # ==================== DATA VALIDATION ====================
        if features.empty or target.empty:
            raise ValueError("Features and target cannot be empty")

        if len(features) != len(target):
            raise ValueError(
                f"Features length ({len(features)}) != "
                f"target length ({len(target)})"
            )

        # Parse index structure to understand data layout
        try:
            if isinstance(features.index, pd.MultiIndex):
                self.n_assets = len(features.index.get_level_values(1).unique())
            else:
                self.n_assets = 1
        except Exception as e:
            raise ValueError(f"Cannot parse index structure: {e}") from e

        self.n_features = features.shape[1]
        self.feature_names = features.columns.tolist()

        # ==================== FEATURE ENGINEERING ====================
        feature_data = self._compute_feature_statistics(features)

        # ==================== SCALER FITTING ====================
        valid_mask = ~feature_data.isna().any(axis=1)
        if valid_mask.sum() > 0:
            self.scaler.fit(feature_data[valid_mask])
        else:
            self.scaler.fit(feature_data)

        self.is_trained = True

    # =====================================================
    # PREDICTION METHOD
    # =====================================================

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate de-meaned cross-sectional signals for prediction period.
        """
        # Create a default zero signal series matching the exact input index shape
        if features.empty:
            return pd.Series(dtype=np.float64)
            
        # 1. Clean data and apply our fitted scaling
        cleaned_features = self._compute_feature_statistics(features)
        scaled_values = self.scaler.transform(cleaned_features)
        scaled_df = pd.DataFrame(scaled_values, index=features.index, columns=features.columns)

        # 2. Extract specific features for our 4-factor ensemble dynamically
        cols = scaled_df.columns
        num_cols = len(cols)

        f_mom = scaled_df.iloc[:, 0] if num_cols > 0 else pd.Series(0.0, index=features.index)
        f_rev = scaled_df.iloc[:, 1] if num_cols > 1 else pd.Series(0.0, index=features.index)
        f_val = scaled_df.iloc[:, 2] if num_cols > 2 else pd.Series(0.0, index=features.index)
        f_qly = scaled_df.iloc[:, 3] if num_cols > 3 else pd.Series(0.0, index=features.index)

        # 3. Calculate raw blended score using the strategy weights
        raw_signal = (
            (f_mom * self.params["momentum_weight"]) +
            (f_rev * self.params["reversal_weight"]) +
            (f_val * self.params["value_weight"]) +
            (f_qly * self.params["quality_weight"])
        )

        # 4. CRITICAL MANDATORY STEP: Cross-Sectional De-Meaning
        if isinstance(raw_signal.index, pd.MultiIndex):
            # Group by the date/timestamp layer to subtract the mean
            group_mean = raw_signal.groupby(level=0).transform('mean')
            demeaned_signal = raw_signal - group_mean
        else:
            # For single timestamp evaluation blocks
            demeaned_signal = raw_signal - raw_signal.mean()

        # Final safety cleanup for any leftover NaNs from empty slices
        demeaned_signal = demeaned_signal.fillna(0.0)

        return demeaned_signal


# Standalone execution guard to prevent root execution issues on server
if __name__ == "__main__":
    pass
