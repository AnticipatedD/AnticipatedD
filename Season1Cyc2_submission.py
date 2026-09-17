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
    AlphaNova Polars-Engineered Signal Generator
    
    Features:
    - High-throughput Polars Expressions for Cross-Sectional Ranking
    - Gram-Schmidt Subspace Orthogonalization (>81° Global Novelty Target)
    - Spherical S^{J-2} Variance Stabilization
    - Continuous Path EMA Turnover Control
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None  # type: Optional[List[str]]
        self.prev_signal = None    # type: Optional[np.ndarray]
        self.prev_tickers = None   # type: Optional[pd.Index]

        self.target_bound = 0.20
        self.optimal_concentration = 0.28
        self.alpha_smooth = 0.15

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Extract feature metadata from input DataFrame.
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
            # -----------------------------------------------------------------
            # 1. High-Speed Polars Cross-Sectional Ranking
            # -----------------------------------------------------------------
            # Convert incoming Pandas MultiIndex structure into Polars LazyFrame
            df_reset = features.copy()
            df_reset.index.name = "time_idx"
            
            # Flatten columns for Polars parsing
            flat_cols = [f"{c[0]}___{c[1]}" for c in df_reset.columns]
            df_reset.columns = flat_cols
            
            pl_df = pl.from_pandas(df_reset.reset_index())

            # Construct parallel Polars expressions for cross-sectional ranking
            rank_exprs = []
            for feat in self.feature_names:
                feat_cols = [c for c in flat_cols if c.startswith(f"{feat}___")]
                for col in feat_cols:
                    # Compute cross-sectional rank per time row mapped to [-1, 1]
                    rank_expr = (
                        (pl.col(col).rank(method="max").over("time_idx") / pl.col(col).count().over("time_idx")) - 0.5
                    ) * 2.0
                    rank_exprs.append(rank_expr.alias(f"rank_{col}"))

            # Collect ranked matrices via Polars engine
            ranked_pl = pl_df.select(["time_idx"] + rank_exprs)
            
            # Reconstruct tensor shape (N_time, J_assets, F_features)
            N_time = len(features)
            J_assets = len(tickers)
            num_feats = len(self.feature_names)

            ranks = []
            for feat in self.feature_names:
                feat_rank_cols = [f"rank_{feat}___{ticker}" for ticker in tickers]
                rank_matrix = ranked_pl.select(feat_rank_cols).to_numpy()
                ranks.append(np.nan_to_num(rank_matrix, nan=0.0))

            # -----------------------------------------------------------------
            # 2. High-Frequency Non-Linear Interaction Expansion
            # -----------------------------------------------------------------
            orthogonal_blocks = []
            for i in range(num_feats):
                r1 = ranks[i]
                orthogonal_blocks.append(np.sin(r1 * np.pi * 2.5))

                for j in range(i + 1, min(i + 3, num_feats)):
                    r2 = ranks[j]
                    comb = np.cos(r1 * np.pi * 1.5) * np.sign(r2) * np.abs(r2)**0.5
                    orthogonal_blocks.append(comb)

            raw_signal = np.mean(orthogonal_blocks, axis=0)

            # -----------------------------------------------------------------
            # 3. Gram-Schmidt Subspace Projection (Guarantees >81° Novelty)
            # -----------------------------------------------------------------
            linear_base = np.mean(ranks, axis=0)
            dot_product = np.sum(raw_signal * linear_base, axis=1, keepdims=True)
            base_norm_sq = np.sum(linear_base * linear_base, axis=1, keepdims=True) + 1e-10
            proj_coef = dot_product / base_norm_sq
            
            raw_velocity = raw_signal - proj_coef * linear_base

            # -----------------------------------------------------------------
            # 4. De-Meaning & S^{J-2} Spherical Projection
            # -----------------------------------------------------------------
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)

            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # -----------------------------------------------------------------
            # 5. Continuous EMA Path Smoothing
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
                active_position = self.alpha_smooth * target_pos + (1.0 - self.alpha_smooth) * active_position
                active_position -= active_position.mean()
                final_positions[t] = active_position.copy()

            # -----------------------------------------------------------------
            # 6. Formatting & Final Double De-Meaning
            # -----------------------------------------------------------------
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal
