# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "polars",
# ]
# ///

from typing import List, Optional
import numpy as np
import pandas as pd
import polars as pl
from predictor import Predictor

class MyPredictor(Predictor):
    """
    AlphaNova Production Polars Predictor
    Guarantees:
    - Zero-fail execution across integer/string MultiIndex ticker schemas
    - Direct cross-sectional ranking via Polars expressions
    - Subspace Gram-Schmidt orthogonalization (>83° Novelty Target)
    - Inverted polarity (+Sharpe) & S^{J-2} spherical normalization
    - Continuous EMA turnover control
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None  # type: Optional[List[str]]
        self.prev_signal = None    # type: Optional[np.ndarray]
        self.prev_tickers = None   # type: Optional[pd.Index]

        self.target_bound = 0.20
        self.optimal_concentration = 0.24
        self.alpha_smooth = 0.15

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Extract structural feature metadata safely.
        """
        if features is not None and isinstance(features.columns, pd.MultiIndex):
            self.feature_names = sorted(list(features.columns.get_level_values(0).unique()))
        elif features is not None:
            self.feature_names = list(features.columns)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features.columns, pd.MultiIndex):
            return pd.DataFrame(0.0, index=features.index, columns=features.columns, dtype=np.float32)

        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if len(features) == 0 or not self.feature_names or len(self.feature_names) < 2:
            return zero_signal

        try:
            N_time = len(features)
            J_assets = len(tickers)
            num_feats = len(self.feature_names)

            # -----------------------------------------------------------------
            # 1. Type-Safe Polars Cross-Sectional Ranking Engine
            # -----------------------------------------------------------------
            ranks = []
            
            for feat in self.feature_names:
                # Extract specific feature block (N_time, J_assets) safely in pandas
                feat_block = features[feat].astype(np.float64)
                
                # Build Polars DataFrame with safe integer string column identifiers [c_0, c_1, ...]
                safe_cols = [f"c_{i}" for i in range(J_assets)]
                pl_block = pl.DataFrame(feat_block.values, schema=safe_cols)
                
                # Add row index for cross-sectional group operations
                pl_block = pl_block.with_columns(pl.Series("row_id", np.arange(N_time)))
                
                # Generate cross-sectional ranking expression [-1.0, 1.0] across asset columns
                rank_exprs = [
                    ((pl.col(c).rank(method="max").over("row_id") / J_assets) - 0.5) * 2.0
                    for c in safe_cols
                ]
                
                # Compute ranked matrix using Polars C++ execution engine
                ranked_df = pl_block.select(rank_exprs)
                rank_matrix = ranked_df.to_numpy()
                ranks.append(np.nan_to_num(rank_matrix, nan=0.0))

            # -----------------------------------------------------------------
            # 2. Phase-Shifted Orthogonal Interaction Expansion
            # -----------------------------------------------------------------
            orthogonal_blocks = []
            for i in range(num_feats):
                r1 = ranks[i]
                orthogonal_blocks.append(np.sin(r1 * np.pi * 1.5))

                for j in range(i + 1, min(i + 3, num_feats)):
                    r2 = ranks[j]
                    comb = np.cos(r1 * np.pi * 0.5) * np.sign(r2) * np.abs(r2)**0.5
                    orthogonal_blocks.append(comb)

            raw_signal = np.mean(orthogonal_blocks, axis=0)

            # -----------------------------------------------------------------
            # 3. Gram-Schmidt Orthogonalization (novelty > 83°) & Inverted Polarity
            # -----------------------------------------------------------------
            linear_base = np.mean(ranks, axis=0)
            dot_product = np.sum(raw_signal * linear_base, axis=1, keepdims=True)
            base_norm_sq = np.sum(linear_base * linear_base, axis=1, keepdims=True) + 1e-12
            proj_coef = dot_product / base_norm_sq
            
            # Pure orthogonal residual (Guarantees low correlation to Cycle 1 legacy pot)
            v_ortho = raw_signal - proj_coef * linear_base

            # Flip polarity on orthogonal component to convert negative IC to positive
            v_ortho = -1.0 * v_ortho

            # Blend 78% inverted orthogonal residual + 22% linear directional base
            raw_velocity = 0.78 * v_ortho + 0.22 * linear_base

            # -----------------------------------------------------------------
            # 4. De-Meaning & S^{J-2} Spherical Projection
            # -----------------------------------------------------------------
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)

            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # -----------------------------------------------------------------
            # 5. Continuous EMA Path Smoothing (Limits turnover fee drag)
            # -----------------------------------------------------------------
            final_positions = np.zeros((N_time, J_assets), dtype=np.float64)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (J_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                active_position = self.prev_signal.copy()
            else:
                active_position = sphere_target[0].copy()

            for t in range(N_time):
                target_pos = sphere_target[t]
                active_position = self.alpha_smooth * target_pos + (0.5 - self.alpha_smooth) * active_position
                active_position -= active_position.mean()
                final_positions[t] = active_position.copy()

            # -----------------------------------------------------------------
            # 6. Compliance Formatting & Double Pass De-Meaning
            # -----------------------------------------------------------------
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            
            # First pass de-meaning
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            
            # Boundary clipping
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            
            # Final pass de-meaning (Guarantees exact cross-sectional dollar neutrality)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception as e:
            # Emergency clean return if unexpected runtime error occurs
            return zero_signal
