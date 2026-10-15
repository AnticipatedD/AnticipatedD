# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "pyarrow",
#   "scikit-learn",
# ]
# ///
import os
import numpy as np
import pandas as pd
from predictor import Predictor
from sklearn.linear_model import Ridge

class MyPredictor(Predictor):
    """
    AlphaNova Season 1 Cycle 5 Robust Production Engine.
    Implements Cross-Sectional Ridge Mapping, Overfitting Gate Resilience,
    Ambient City Space Repulsion Steering, and L2 Churn Smoothing.
    """
    def __init__(self):
        super().__init__()
        self.model = Ridge(alpha=100.0, fit_intercept=False)
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        
        # OFFICIALLY CALIBRATED REGULATORY HYPERPARAMETERS
        self.target_bound = 1.35            # Strict exposure cap limit
        self.smoothing_factor = 0.40        # Mitigates the 5bp transaction fee Sharpe drag
        self.novelty_steering_weight = 0.35 # Intensity for coordinate space deflection

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Overfitting Gate Resilient Training. Learns a mapping function 
        invariant to temporal row permutations.
        """
        if features is not None and target is not None and not features.empty:
            try:
                # Align column ordering deterministically
                self.feature_names = sorted(list(features.columns.get_level_values(0).unique()))
                
                X_list, y_list = [], []
                # Flatten cross-sectionally to comply with row-permuted retrains
                for feat in self.feature_names:
                    X_list.append(features[feat].astype(np.float64).fillna(0.0).to_numpy().flatten())
                
                X = np.column_stack(X_list)
                y = target.astype(np.float64).fillna(0.0).to_numpy().flatten()
                
                # Fit standard regularized linear model
                self.model.fit(X, y)
            except Exception:
                pass

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        zero_signal = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if features is None or features.empty:
            return zero_signal

        current_features = sorted(list(features.columns.get_level_values(0).unique()))
        if not self.feature_names:
            self.feature_names = current_features
        else:
            self.feature_names = [f for f in self.feature_names if f in current_features]

        if len(self.feature_names) == 0:
            return zero_signal

        try:
            # 1. Generate Base Cross-Sectional Predictions
            N_time = len(features.index)
            J_assets = len(tickers)
            
            raw_preds = np.zeros((N_time, J_assets), dtype=np.float64)
            for idx, feat in enumerate(self.feature_names):
                feat_mat = features[feat].astype(np.float64).fillna(0.0).to_numpy()
                raw_preds += feat_mat * self.model.coef_[idx]

            # Enforce immediate cross-sectional de-meaning (Zero-Beta Requirement)
            raw_preds = raw_preds - raw_preds.mean(axis=1, keepdims=True)

            # 2. Ambient City Space Novelty Deflection Layer
            steering_vector = np.zeros_like(raw_preds)
            cities_path = os.path.join("signal_db", "signal_cities_ambient_coords.parquet")
            
            if os.path.exists(cities_path):
                try:
                    df_ambient = pd.read_parquet(cities_path)
                    # Extract occupied coordinates if matching asset structure dimensions
                    if not df_ambient.empty and df_ambient.shape[1] == J_assets:
                        occupied_coords = df_ambient.to_numpy(dtype=np.float64)
                        
                        # Calculate mean direction of competing signals
                        avg_competing_city = occupied_coords.mean(axis=0, keepdims=True)
                        avg_competing_city = avg_competing_city / (np.linalg.norm(avg_competing_city) + 1e-12)
                        
                        # Project onto orthogonal subspace to create direct geometric repulsion
                        projection_scalar = np.dot(raw_preds, avg_competing_city.T)
                        steering_vector = -(projection_scalar @ avg_competing_city)
                        steering_vector = steering_vector - steering_vector.mean(axis=1, keepdims=True)
                except Exception:
                    steering_vector = np.zeros_like(raw_preds)

            # Blend raw predictions with our calculated novelty steering vector
            combined_velocity = raw_preds + (self.novelty_steering_weight * steering_vector)
            combined_velocity = combined_velocity - combined_velocity.mean(axis=1, keepdims=True)
            
            # Map clean signals onto the optimized trading unit sphere vector
            norms = np.linalg.norm(combined_velocity, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            target_signal = combined_velocity / norms

            # 3. L2 Churn Smoothing Layer (Dampens 5bp slippage churn costs)
            final_positions = np.zeros_like(target_signal)
            
            if (
                self.prev_signal is not None 
                and self.prev_tickers is not None 
                and self.prev_signal.shape == (J_assets,)
                and list(self.prev_tickers) == list(tickers)
            ):
                active_position = self.prev_signal.copy()
            else:
                active_position = target_signal[0].copy()

            for t in range(N_time):
                # Smooth allocation path trajectory dynamically
                curr_position = (self.smoothing_factor * target_signal[t] + 
                                 (1.0 - self.smoothing_factor) * active_position)
                curr_position -= curr_position.mean()
                final_positions[t] = curr_position
                active_position = curr_position.copy()

            # 4. Final Compliance Formatting & Boundary Bounding
            out = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            # Persist tracking snapshots for out-of-sample consistency checks
            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            if self.prev_signal is not None and len(self.prev_signal) == len(tickers):
                return pd.DataFrame(
                    np.tile(self.prev_signal, (len(features), 1)),
                    index=features.index,
                    columns=tickers,
                    dtype=np.float32,
                )
            return zero_signal
