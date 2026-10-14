# /// script
# dependencies = [
#   "numpy",
#   "pandas"
# ]
# ///

import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    """
    AlphaNova Season 1 - New Approach
    Robust Cross-Sectional Z-Score Ensemble 
    + Inverse-Volatility Scaling 
    + Soft Exponential Smoothing
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_position = None
        self.prev_tickers = None

        # Parameters
        self.ema_alpha = 0.35          # light temporal smoothing
        self.target_vol = 0.18         # target cross-sectional volatility
        self.clip_bound = 0.22

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is not None and not features.empty:
            try:
                self.feature_names = list(features.columns.get_level_values(0).unique())
            except Exception:
                self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        zero = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if (features is None or features.empty or 
            self.feature_names is None or len(self.feature_names) < 1):
            return zero

        try:
            # -------------------------------------------------
            # 1. Robust cross-sectional z-score for every feature
            # -------------------------------------------------
            z_list = []
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                block = features[feat].astype(np.float64)
                
                # Robust z-score (median + MAD)
                med = block.median(axis=1)
                mad = (block.sub(med, axis=0)).abs().median(axis=1) + 1e-8
                z = block.sub(med, axis=0).div(mad * 1.4826, axis=0)
                z = z.fillna(0.0).clip(-4.0, 4.0)
                z_list.append(z.to_numpy())

            if len(z_list) == 0:
                return zero

            # Average all feature z-scores
            raw = np.mean(z_list, axis=0)          # shape (T, J)

            # -------------------------------------------------
            # 2. Inverse-volatility scaling (target specific vol)
            # -------------------------------------------------
            cs_vol = np.std(raw, axis=1, keepdims=True) + 1e-8
            scaled = raw * (self.target_vol / cs_vol)

            # -------------------------------------------------
            # 3. Soft exponential moving average (temporal)
            # -------------------------------------------------
            T, J = scaled.shape
            out = np.zeros_like(scaled)

            if (self.prev_position is not None and 
                self.prev_tickers is not None and 
                len(self.prev_position) == J and 
                self.prev_tickers.equals(tickers)):
                prev = self.prev_position.copy()
            else:
                prev = np.zeros(J)

            for t in range(T):
                current = self.ema_alpha * scaled[t] + (1.0 - self.ema_alpha) * prev
                current -= current.mean()          # strict demean
                out[t] = current
                prev = current.copy()

            # -------------------------------------------------
            # 4. Final formatting
            # -------------------------------------------------
            df = pd.DataFrame(out, index=features.index, columns=tickers)
            df = df.clip(-self.clip_bound, self.clip_bound)
            df = df.sub(df.mean(axis=1), axis=0)   # final demean

            # Store state
            self.prev_position = df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = df.columns.copy()

            return df.astype(np.float32)

        except Exception:
            return zero
            
predictor = MyPredictor()
