# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
import numpy as np
import pandas as pd
from predictor import Predictor

class MyPredictor(Predictor):
    """
    AlphaNova Final Cycle Production Strategy.
    Robust Data-Gate Validation with Calibrated Cross-Sectional Scaling,
    Hyperbolic Outlier Protection, and Causal L1 Churn Control.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        
        # HYPERPARAMETERS OPTIMIZED FOR SEASON 1 FINAL CYCLE
        self.target_bound = 1.35
        self.optimal_concentration = 0.28   # Provides geometric breathing room on S^18
        self.l1_hysteresis_threshold = 1.05  # Strong gatekeeper to neutralize 5bp fee drag
        self.smoothing = 0.75                # Balanced alpha adoption velocity (0.25 effective shift)

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        """
        Shuffling-gate resilient feature validation.
        """
        if features is not None and not features.empty:
            try:
                self.feature_names = list(features.columns.get_level_values(0).unique())
            except Exception:
                self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # Establish default tickers safely
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        # Fallback benchmark baseline to prevent catastrophic execution crashes
        zero_signal = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        # COMPLIANCE CHECK A: Catch empty/malformed data slices before processing
        if (features is None or features.empty or 
            self.feature_names is None or len(self.feature_names) < 2):
            return zero_signal

        try:
            # 1. Clean Row-Wise Cross-Sectional Rank Transform
            ranks = {}
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                block_val = features[feat].astype(np.float64)
                # Map to uniform space [-1.0, 1.0] to prevent downstream saturation
                r = (block_val.rank(axis=1, pct=True, method='average') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            # COMPLIANCE CHECK B: Prevent StopIteration if features mismatch during live testing
            if len(ranks) == 0:
                return zero_signal

            N_time, J_assets = next(iter(ranks.values())).shape

            # 2. Balanced Non-Linear Spatial Expansion
            interaction_blocks = []
            feat_list = list(ranks.keys())

            for i, f1 in enumerate(feat_list):
                # Bounded hyperbolic baseline
                interaction_blocks.append(np.tanh(ranks[f1] * 1.5))

                for j in range(i + 1, len(feat_list)):
                    f2 = feat_list[j]
                    # Orthogonal phase combination
                    interaction_blocks.append(
                        np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25)
                    )
                    # Modulated volatility tail mapping
                    interaction_blocks.append(ranks[f1] * np.abs(ranks[f2]))

            # Combine interactions safely without phase-cancellation
            raw_velocity = np.mean(interaction_blocks, axis=0)

            # 3. Geometric Subspace Demean & Hypersphere Projection
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)

            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # 4. Execution Filter: L1 Causal Hysteresis Loop
            final_positions = np.zeros_like(sphere_target)

            # Re-establish running state link if asset dimensions align perfectly
            if (
                self.prev_signal is not None 
                and self.prev_tickers is not None 
                and len(self.prev_signal) == J_assets 
                and self.prev_tickers.equals(tickers)
            ):
                active_position = self.prev_signal.copy()
            else:
                active_position = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target_position = sphere_target[t]
                l1_delta = np.sum(np.abs(target_position - active_position))

                if l1_delta < self.l1_hysteresis_threshold:
                    # Inside the band: Hold position flat to pay 0bp transaction fee
                    current = active_position.copy()
                else:
                    # Outside the band: Smooth execution tracking adjustment
                    current = (1.0 - self.smoothing) * target_position + self.smoothing * active_position
                    current -= current.mean()  # Structural dollar-neutral verification

                final_positions[t] = current
                active_position = current.copy()

            # 5. Compliance Formatting & Boundary Constraint Allocation
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            # Track strict integer location state for consecutive streaming lookbehinds
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal

# Core initialization handle for tournament execution runner
predictor = MyPredictor()
