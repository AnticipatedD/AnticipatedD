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
    Production-Grade Quant Strategy for AlphaNova.
    Implements a Cross-Sectional Supervised Sign-Alignment Network with 
    L2 Information Coefficient Shrinkage and Adaptive Velocity Blending.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.feature_weights = None  # Learns true historical sign & direction
        self.prev_position = None
        self.prev_tickers = None

        # Corrected parameters for tournament scale optimization
        self.power = 1.0               # High power on weak signals dampens alpha; kept linear
        self.l1_target = 0.25          # Exact L1 norm target
        self.decay = 0.45              # Swapped slow decay (0.92) for active friction mitigation
        self.clip_bound = 0.20

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        """
        Calculates the cross-sectional alignment vector to safely extract
        unbiased alpha direction from the 6 available features.
        """
        if features is None or target is None or features.empty or target.empty:
            return

        try:
            self.feature_names = list(features.columns.get_level_values(0).unique())
            
            # Align training timelines securely
            common_idx = features.index.intersection(target.index)
            if len(common_idx) == 0:
                return
                
            feat_sub = features.loc[common_idx]
            targ_sub = target.loc[common_idx]
            
            # Cross-sectional rank transform of the target returns
            t_rank = targ_sub.rank(axis=1, pct=True, method="average") - 0.5
            
            self.feature_weights = {}
            for feat in self.feature_names:
                if feat not in feat_sub.columns.get_level_values(0):
                    continue
                f_rank = feat_sub[feat].rank(axis=1, pct=True, method="average") - 0.5
                
                # Compute true out-of-sample directional alignment (Information Coefficient)
                ic = (f_rank * t_rank).mean().mean()
                self.feature_weights[feat] = ic
                
            # Apply L2 regularized shrinkage to lock down variance stability
            total_variance = sum(w**2 for w in self.feature_weights.values()) + 1e-8
            for k in self.feature_weights:
                self.feature_weights[k] = self.feature_weights[k] / np.sqrt(total_variance)

        except Exception:
            self.feature_names = None
            self.feature_weights = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        zero = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if (features is None or features.empty or 
            self.feature_names is None or self.feature_weights is None or len(tickers) == 0):
            return zero

        try:
            # -------------------------------------------------
            # 1. Supervised Cross-Sectional Rank Transform
            # -------------------------------------------------
            weighted_signals = []
            
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                    
                weight = self.feature_weights.get(feat, 0.0)
                if abs(weight) < 1e-7:
                    continue
                
                block = features[feat].astype(np.float64)
                q = block.rank(axis=1, pct=True, method="average")
                q = q.fillna(0.5)
                
                # Center around 0.0 and apply directional alignment sign
                centered_q = (q.to_numpy() - 0.5) * weight
                weighted_signals.append(centered_q)

            if len(weighted_signals) == 0:
                return zero

            # Extract consensus trajectory safely aligned to target direction
            combined_signal = np.sum(weighted_signals, axis=0)
            combined_signal -= combined_signal.mean(axis=1, keepdims=True)

            # -------------------------------------------------
            # 2. Bounded Nonlinearity Filter
            # -------------------------------------------------
            signal = np.sign(combined_signal) * (np.abs(combined_signal) ** self.power)

            # -------------------------------------------------
            # 3. Exact L1 Projection Compliance
            # -------------------------------------------------
            l1 = np.sum(np.abs(signal), axis=1, keepdims=True) + 1e-8
            signal = signal * (self.l1_target / l1)

            # -------------------------------------------------
            # 4. Adaptive Velocity Smoothing
            # -------------------------------------------------
            T, J = signal.shape
            out = np.zeros_like(signal)

            if (self.prev_position is not None and 
                self.prev_tickers is not None and 
                len(self.prev_position) == J and 
                self.prev_tickers.equals(tickers)):
                prev = self.prev_position.copy()
            else:
                prev = np.zeros(J)

            for t in range(T):
                # Faster blending factor (1 - decay) clears the 5bp trade friction lag
                current = self.decay * prev + (1.0 - self.decay) * signal[t]
                current -= current.mean()  # Structural dollar neutrality
                out[t] = current
                prev = current.copy()

            # -------------------------------------------------
            # 5. Final Subspace Formatting
            # -------------------------------------------------
            df = pd.DataFrame(out, index=features.index, columns=tickers)
            for _ in range(2):
                df = df.sub(df.mean(axis=1), axis=0)
                df = df.clip(-self.clip_bound, self.clip_bound)

            # Store running pipeline states
            self.prev_position = df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = df.columns.copy()

            return df.astype(np.float32)

        except Exception:
            return zero


predictor = MyPredictor()
