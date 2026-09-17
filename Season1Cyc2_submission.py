# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from predictor import Predictor

class MyPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.model = Ridge(alpha=100.0)
        self.prev_signal = None
        self.alpha_smooth = 0.15

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is None or target is None or features.empty:
            return

        try:
            # 1. Vectorized Rank Normalization [-1, 1] across assets (level=1)
            # Grouping by time (level=0) and ranking across assets
            f_rank = features.groupby(level=0).rank(pct=True, method='max')
            f_rank = (f_rank - 0.5) * 2.0
            
            X = f_rank.fillna(0.0).to_numpy().astype(np.float32)
            y = target.fillna(0.0).to_numpy().flatten().astype(np.float32)

            self.model.fit(X, y)
        except Exception:
            pass

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = features.columns

        zero_df = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        if features.empty:
            return zero_df

        try:
            # 1. Fast Vectorized Cross-Sectional Ranking
            if isinstance(features.columns, pd.MultiIndex):
                f_rank = features.groupby(level=0, axis=0).rank(pct=True, method='max')
            else:
                f_rank = features.rank(axis=1, pct=True, method='max')
            
            f_rank = (f_rank - 0.5) * 2.0
            f_arr = f_rank.fillna(0.0).to_numpy()

            # 2. Vectorized Batch Prediction
            if hasattr(self.model, "coef_"):
                raw_preds = self.model.predict(f_arr)
            else:
                raw_preds = f_arr.mean(axis=1)

            # Reshape into (time, assets) DataFrame
            out_df = pd.DataFrame(raw_preds.reshape(len(features.index), len(tickers)), 
                                  index=features.index, columns=tickers, dtype=np.float32)

            # 3. Vectorized Zero-Sum De-meaning
            out_df = out_df.sub(out_df.mean(axis=1), axis=0)

            # 4. Vectorized Neutralization against Market Mean Factor
            mkt_mean = f_rank.groupby(level=0).mean() if isinstance(features.columns, pd.MultiIndex) else f_rank.mean(axis=1)
            cov_m = (out_df.values * mkt_mean.values).mean(axis=1, keepdims=True)
            var_m = np.var(mkt_mean.values, axis=1, keepdims=True) + 1e-8
            
            out_df -= (cov_m / var_m) * mkt_mean.values
            out_df = out_df.sub(out_df.mean(axis=1), axis=0)

            # 5. EMA Turnover Control
            if self.prev_signal is not None and self.prev_signal.shape == out_df.shape:
                out_df = self.alpha_smooth * out_df + (1.0 - self.alpha_smooth) * self.prev_signal

            self.prev_signal = out_df.copy()
            return out_df.clip(-0.20, 0.20).fillna(0.0).astype(np.float32)

        except Exception:
            return zero_df
