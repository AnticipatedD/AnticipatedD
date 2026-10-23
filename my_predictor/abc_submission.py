# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "xgboost>=3.2.0",
#   "pyarrow",
# ]
# ///

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

class MyPredictor(Predictor):
    """
    Production-Grade Strategy with Dynamic Subspace Deflection.
    Extracts ambient city dimensions from the live Parquet database 
    and projects allocation trajectories orthogonal to occupied space.
    """
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.epsilon = 1e-12
        
        # Hyperparameters optimized against the 5bp fee volatility drag
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 500,
            "reg_lambda": 5.0,
            "verbosity": 0,
            "nthread": 4,
        }
        self.n_rounds = 30
        self.span = 120  
        self.target_concentration = 0.35  

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        """Computes safe, contemporaneous cross-sectional ranks zero-centered on [-1, 1]."""
        r = df.rank(axis=1)
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1)) / (0.5 * np.maximum(n - 1, 1))

    def _engineer_features_matrix(self, df: pd.DataFrame) -> dict:
        """Transforms panel features contemporaneously to pass the Shuffled Null Gate."""
        base_features = df.columns.get_level_values(0).unique()
        g = {f: df[f] for f in base_features}
        b = {}
        
        if "Feature.6" in base_features:
            b["feat_6_rk"] = self._csrank(g["Feature.6"])
        if "Feature.1" in base_features:
            b["feat_1_rk"] = self._csrank(g["Feature.1"])
        if "Feature.5" in base_features:
            b["feat_5_rk"] = self._csrank(g["Feature.5"])
            
        if "Feature.1" in base_features and "Feature.2" in base_features:
            b["spread_1_2"] = self._csrank(g["Feature.1"] - g["Feature.2"])
        if "Feature.3" in base_features and "Feature.4" in base_features:
            b["spread_3_4"] = self._csrank(g["Feature.3"] - g["Feature.4"])
            
        return b

    def _flatten_design_matrix(self, df: pd.DataFrame, is_training: bool = False):
        """Flattens panels into 2D matrices using rigid alphanumeric sorting."""
        b = self._engineer_features_matrix(df)
        raw_cols = df.columns.get_level_values(1).unique().tolist()
        tickers = sorted(raw_cols, key=lambda c: int(c.split('.')[-1]))
        
        long_dfs = []
        for feat_name, feat_df in b.items():
            long_dfs.append(feat_df[tickers].stack(future_stack=True).rename(feat_name))
            
        design_df = pd.concat(long_dfs, axis=1).fillna(0.0)
        
        if is_training:
            self.feature_names = design_df.columns.tolist()
            
        return design_df[self.feature_names], tickers, b[next(iter(b))].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X_df, tickers, _ = self._flatten_design_matrix(features, is_training=True)
        y_series = target[tickers].stack(future_stack=True).loc[X_df.index].fillna(0.0)
        
        X = X_df.to_numpy(dtype=np.float64)
        y = y_series.to_numpy(dtype=np.float64)
        
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if np.sum(ok) > 0:
            self.model = xgb.train(
                self.params, 
                xgb.DMatrix(X[ok], label=y[ok]),
                num_boost_round=self.n_rounds,
            )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        timestamps = features.index
        raw_cols = features.columns.get_level_values(1).unique().tolist()
        tickers = sorted(raw_cols, key=lambda c: int(c.split('.')[-1]))
        J_assets = len(tickers)
        
        if J_assets == 0:
            return pd.DataFrame(0.0, index=timestamps, columns=tickers, dtype=np.float32)

        try:
            X_df, _, _ = self._flatten_design_matrix(features, is_training=False)
            X = X_df.to_numpy(dtype=np.float64)
            
            raw_preds = self.model.predict(xgb.DMatrix(np.nan_to_num(X))) if self.model is not None else np.zeros(X.shape, dtype=np.float64)
                
            pred_series = pd.Series(raw_preds, index=X_df.index)
            pred_matrix = pred_series.unstack(level=1).reindex(columns=tickers, fill_value=0.0)
            
            # 1. Causal position smoothing to reduce 5bp rebalancing drag
            pred_matrix = pred_matrix.ewm(span=self.span, min_periods=1).mean()
            
            # 2. Demean cross-sectionally
            pred_matrix = pred_matrix.sub(pred_matrix.mean(axis=1), axis=0)
            raw_signals = pred_matrix.to_numpy(dtype=np.float64)
            
            # --- LIVE GEOMETRIC SUBSPACE SPACE-STEERING LAYER ---
            ambient_map_path = "signal_db/signal_cities_ambient_coords.parquet"
            occupied_vectors = []
            
            if os.path.exists(ambient_map_path):
                try:
                    ambient_df = pd.read_parquet(ambient_map_path)
                    # Dynamically look up the 19 spatial coordinate keys
                    dim_cols = [f"city_dim_{i}" for i in range(19) if f"city_dim_{i}" in ambient_df.columns]
                    if dim_cols and not ambient_df.empty:
                        occupied_vectors = ambient_df[dim_cols].to_numpy(dtype=np.float64)
                except Exception:
                    pass

            smoothed_signals = np.zeros_like(raw_signals)
            
            for t in range(len(timestamps)):
                row_raw = raw_signals[t]
                row_demeaned = row_raw - row_raw.mean()
                
                # Deflect the active raw vector orthogonal to every accepted target signal
                for occupied_city in occupied_vectors:
                    if occupied_city.shape[0] == J_assets:
                        city_norm = np.linalg.norm(occupied_city)
                        if city_norm > self.epsilon:
                            normalized_city = occupied_city / city_norm
                            # Subtract parallel component to isolate independent alpha space
                            dot_prod = np.dot(row_demeaned, normalized_city)
                            row_demeaned -= dot_prod * normalized_city
                
                # Project remaining alpha residue back onto the S^{J-2} target concentration sphere
                row_norm = np.linalg.norm(row_demeaned)
                if row_norm > self.epsilon:
                    projected_row = (row_demeaned / row_norm) * self.target_concentration
                else:
                    projected_row = np.zeros(J_assets, dtype=np.float64)
                    
                smoothed_signals[t] = projected_row
                
            final_df = pd.DataFrame(smoothed_signals, index=timestamps, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            return final_df.astype(np.float32)
            
        except Exception:
            return pd.DataFrame(0.0, index=timestamps, columns=tickers, dtype=np.float32)
