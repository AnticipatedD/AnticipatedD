# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///

from typing import List, Optional
import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    """
    AlphaNova Ultra-Robust Production Signal Generator
    
    Guarantees:
    - Zero silent error masking (explicit execution path)
    - Full point-in-time universe alignment via explicit reindexing
    - Robust handling for both MultiIndex and flat Index column formats
    - State-mapped EMA smoothing resilient to changing ticker universes
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None  # type: Optional[List[str]]
        self.prev_signal_series = None  # type: Optional[pd.Series]

        self.target_bound = 0.20
        self.optimal_concentration = 0.28
        self.alpha_smooth = 0.15

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Extract structural feature names safely across varying column structures.
        """
        if features is not None and isinstance(features.columns, pd.MultiIndex):
            self.feature_names = sorted(list(features.columns.get_level_values(0).unique()))
        elif features is not None:
            self.feature_names = list(features.columns)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if len(features) == 0:
            return pd.DataFrame()

        # -----------------------------------------------------------------
        # 1. Flexible Column Index Parsing & Master Ticker Extraction
        # -----------------------------------------------------------------
        if isinstance(features.columns, pd.MultiIndex):
            tickers = features.columns.get_level_values(1).unique()
            if self.feature_names is None:
                self.feature_names = sorted(list(features.columns.get_level_values(0).unique()))
        else:
            # Fallback for flat Index during sanity/smoke checks
            tickers = features.columns
            if self.feature_names is None:
                self.feature_names = list(features.columns)

        N_time = len(features)
        J_assets = len(tickers)

        if J_assets == 0 or len(self.feature_names) < 2:
            return pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        # -----------------------------------------------------------------
        # 2. Reindexed Feature Extraction (Eliminates Shape Mismatches)
        # -----------------------------------------------------------------
        ranks = []
        for feat in self.feature_names:
            if isinstance(features.columns, pd.MultiIndex):
                # Extract slice and reindex to master tickers to guarantee exact shape (N_time, J_assets)
                try:
                    feat_df = features.xs(feat, axis=1, level=0)
                except KeyError:
                    feat_cols = [c for c in features.columns if c[0] == feat]
                    feat_df = features[feat_cols]
                    feat_df.columns = [c[1] for c in feat_cols]
            else:
                feat_df = features[[feat]]

            # Guarantee shape match regardless of missing asset columns
            feat_df = feat_df.reindex(columns=tickers)
            feat_data = np.nan_to_num(feat_df.to_numpy(dtype=np.float64), nan=0.0)

            # Vectorized cross-sectional percentile ranking along axis=1 [-1.0, 1.0]
            argsort_indices = np.argsort(feat_data, axis=1)
            rank_matrix = np.empty_like(argsort_indices, dtype=np.float64)
            
            rows = np.arange(N_time)[:, None]
            rank_matrix[rows, argsort_indices] = np.arange(J_assets)
            
            denominator = max(J_assets - 1, 1)
            rank_normalized = ((rank_matrix / denominator) - 0.5) * 2.0
            ranks.append(rank_normalized)

        num_feats = len(ranks)

        # -----------------------------------------------------------------
        # 3. Phase-Shifted Interaction Expansion
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
        # 4. Gram-Schmidt Orthogonalization & Inverted Polarity
        # -----------------------------------------------------------------
        linear_base = np.mean(ranks, axis=0)
        dot_product = np.sum(raw_signal * linear_base, axis=1, keepdims=True)
        base_norm_sq = np.sum(linear_base * linear_base, axis=1, keepdims=True) + 1e-10
        proj_coef = dot_product / base_norm_sq

        # Isolate orthogonal component
        v_ortho = raw_signal - proj_coef * linear_base

        # Reverse polarity on orthogonal residual to flip negative IC to positive
        v_ortho = -1.0 * v_ortho

        # Blend 65% inverted orthogonal residual + 35% linear directional base
        raw_velocity = 0.65 * v_ortho + 0.35 * linear_base

        # -----------------------------------------------------------------
        # 5. Cross-Sectional De-Meaning & Spherical S^{J-2} Projection
        # -----------------------------------------------------------------
        velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)

        norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

        # -----------------------------------------------------------------
        # 6. Stateful EMA Path Smoothing (Ticker-Mapped)
        # -----------------------------------------------------------------
        final_positions = np.zeros((N_time, J_assets), dtype=np.float64)

        if self.prev_signal_series is not None:
            # Reindex previous state to current ticker universe to handle asset membership shifts
            active_series = self.prev_signal_series.reindex(tickers, fill_value=0.0)
            active_position = active_series.to_numpy(dtype=np.float64)
        else:
            active_position = sphere_target[0].copy()

        for t in range(N_time):
            target_pos = sphere_target[t]
            active_position = self.alpha_smooth * target_pos + (1.0 - self.alpha_smooth) * active_position
            active_position -= active_position.mean()
            final_positions[t] = active_position.copy()

        # Update persistent state Series for streaming / consecutive calls
        self.prev_signal_series = pd.Series(final_positions[-1], index=tickers)

        # -----------------------------------------------------------------
        # 7. Final Formatting & Double Pass Dollar-Neutrality
        # -----------------------------------------------------------------
        final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
        
        final_df = final_df.sub(final_df.mean(axis=1), axis=0)
        final_df = final_df.clip(-self.target_bound, self.target_bound)
        final_df = final_df.sub(final_df.mean(axis=1), axis=0)

        return final_df.astype(np.float32)
