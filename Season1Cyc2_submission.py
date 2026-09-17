# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
import numpy as np
import pandas as pd
from typing import List, Optional
from predictor import Predictor

class MyPredictor(Predictor):
    """
    AlphaNova Elite Production Signal Generator
    
    Architecture:
    1. Cross-Sectional Percentile Rank Mapping [-1, 1]
    2. Phase-Shifted Orthogonal Nonlinear Interaction Expansion
    3. Row-Wise Hypersphere S^{J-2} Variance Projection
    4. Adaptive Exponential Turnover Control (EMA alpha=0.15)
    5. Dual Cross-Sectional De-Meaning Enforcement
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None  # type: Optional[List[str]]
        self.prev_signal = None    # type: Optional[np.ndarray]
        self.prev_tickers = None   # type: Optional[pd.Index]

        # Tuned hyper-parameters
        self.target_bound = 0.20
        self.optimal_concentration = 0.28
        self.alpha_smooth = 0.15  # Continuous EMA parameter to control turnover

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Extract structural feature names from input DataFrame.
        Works across row permutations and shuffling gate tests.
        """
        if features is not None and isinstance(features.columns, pd.MultiIndex):
            self.feature_names = sorted(list(features.columns.get_level_values(0).unique()))
        elif features is not None:
            self.feature_names = list(features.columns)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-sectionally de-meaned signals adhering to competition rules.
        """
        if not isinstance(features.columns, pd.MultiIndex):
            return pd.DataFrame(0.0, index=features.index, columns=features.columns, dtype=np.float32)

        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 2:
            return zero_signal

        try:
            # -----------------------------------------------------------------
            # 1. Cross-Sectional Rank Transformation
            # -----------------------------------------------------------------
            ranks = []
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                # Map cross-sectional ranks onto [-1.0, 1.0]
                r = (block.rank(axis=1, pct=True, method="max") - 0.5) * 2.0
                ranks.append(r.fillna(0.0).to_numpy())

            N_time, J_assets = ranks[0].shape

            # -----------------------------------------------------------------
            # 2. Phase-Shifted Orthogonal Nonlinear Expansion
            # -----------------------------------------------------------------
            interaction_blocks = []
            num_feats = len(self.feature_names)

            for i in range(num_feats):
                f1_rank = ranks[i]
                # Primary non-linear activation
                interaction_blocks.append(np.tanh(f1_rank * 2.0))

                for j in range(i + 1, min(i + 3, num_feats)):
                    f2_rank = ranks[j]
                    # Dynamic phase-shifted combinations for ambient spatial coordinates
                    ps_comb = np.sin(f1_rank * np.pi * 0.25) * np.cos(f2_rank * np.pi * 0.25)
                    interaction_blocks.append(ps_comb)
                    interaction_blocks.append(f1_rank * np.abs(f2_rank))

            # Compute consensus raw signal velocity
            raw_velocity = np.mean(interaction_blocks, axis=0)

            # -----------------------------------------------------------------
            # 3. Geometric Subspace Demean & Hypersphere S^{J-2} Projection
            # -----------------------------------------------------------------
            # Primary De-Meaning
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)

            # Spherical Projection
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # -----------------------------------------------------------------
            # 4. Continuous EMA Turnover Control
            # -----------------------------------------------------------------
            final_positions = np.zeros((N_time, J_assets), dtype=np.float64)

            # Determine initial state
            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (J_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                active_position = self.prev_signal.copy()
            else:
                active_position = sphere_target[0].copy()

            # Process continuous path smoothing
            for t in range(N_time):
                target_pos = sphere_target[t]
                active_position = self.alpha_smooth * target_pos + (1.0 - self.alpha_smooth) * active_position
                active_position -= active_position.mean()
                final_positions[t] = active_position.copy()

            # -----------------------------------------------------------------
            # 5. Strict Double De-Meaning & Boundary Compliance
            # -----------------------------------------------------------------
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)

            # First pass de-meaning
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            # Boundary clipping
            final_df = final_df.clip(-self.target_bound, self.target_bound)

            # Final pass de-meaning (Strict dollar neutrality guarantee)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            # Store snapshot for streaming and consecutive evaluations
            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal
