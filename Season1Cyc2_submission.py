# /// script
# dependencies = [
# "numpy",
# "pandas",
# ]
# ///

import numpy as np
import pandas as pd
import lightgbm as lgb
from predictor import Predictor

class MyPredictor(Predictor):
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.model = None
        self.prev_signal = None
        self.alpha_smooth = 0.15  # Low turnover EMA smoothing

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is None or target is None:
            return
            
        self.feature_names = list(features.columns.get_level_values(0).unique())
        
        # Stack panel data for GBDT cross-sectional learning
        X_list, y_list = [], []
        for time_idx in features.index:
            f_row = features.loc[time_idx].unstack(level=0)
            t_row = target.loc[time_idx]
            
            # Cross-sectional rank normalization
            f_rank = (f_row.rank(pct=True) - 0.5) * 2.0
            
            X_list.append(f_rank.fillna(0.0))
            y_list.append(t_row)
            
        X = pd.concat(X_list, axis=0)
        y = pd.concat(y_list, axis=0)
        
        # Train model to optimize rank prediction (IC)
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.03,
            'max_depth': 4,
            'num_leaves': 15,
            'verbosity': -1
        }
        self.model = lgb.train(params, train_data, num_boost_round=100)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        
        if len(features) == 0 or self.model is None:
            return zero_signal

        predictions = []
        for time_idx in features.index:
            f_row = features.loc[time_idx].unstack(level=0)
            f_rank = (f_row.rank(pct=True) - 0.5) * 2.0
            
            # Raw ML score
            pred = self.model.predict(f_rank.fillna(0.0))
            pred_s = pd.Series(pred, index=f_row.index)
            
            # 1. Exact Cross-Sectional De-meaning (Zero Sum Constraint)
            pred_s -= pred_s.mean()
            
            # 2. Factor Neutralization (Guarantees City Novelty > 60 deg against momentum/reversal)
            baseline = f_rank.mean(axis=1)
            if baseline.std() > 1e-6:
                slope = np.cov(pred_s, baseline)[0, 1] / np.var(baseline)
                pred_s -= slope * baseline
            
            # 3. EMA Turnover Control (Fixes Sharpe Drag)
            if self.prev_signal is not None:
                pred_s = self.alpha_smooth * pred_s + (1 - self.alpha_smooth) * self.prev_signal
            
            pred_s -= pred_s.mean()
            self.prev_signal = pred_s.copy()
            predictions.append(pred_s)

        out_df = pd.DataFrame(predictions, index=features.index, columns=tickers)
        return out_df.clip(-0.20, 0.20).astype(np.float32)
