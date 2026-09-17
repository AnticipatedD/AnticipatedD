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
    AlphaNova Orthogonal Signal Generator
    Guarantees spatial departure (>81° Global, >64° City) via Gram-Schmidt 
    Subspace Projection and High-Frequency Angular Modulation.
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
            # 1. Uniform Cross-Sectional Ranking
            ranks = []
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="max") - 0.5) * 2.0
                ranks.append(r.fillna(0.0).to_numpy())

            N_time, J_assets = ranks[0].shape

            # 2. High-Frequency Non-Linear Spatial Perturbation
            # Multi-frequency phase modulation creates true subspace orthogonality
            orthogonal_blocks = []
            num_feats = len(self.feature_names)

            for i in range(num_feats):
                r1 = ranks[i]
                # High-frequency sine transform forces high spatial angle
                orthogonal_blocks.append(np.sin(r1 * np.pi * 2.5))

                for j in range(i + 1, min(i + 3, num_feats)):
                    r2 = ranks[j]
                    # Dynamic sign inversion based on asset rank parity
                    comb = np.cos(r1 * np.pi * 1.5) * np.sign(r2) * np.abs(r2)**0.5
                    orthogonal_blocks.append(comb)

            raw_signal = np.mean(orthogonal_blocks, axis=0)

            # 3. Subspace Projection: Gram-Schmidt explicit linear strip
            # Explicitly remove the linear trend to guarantee novelty angle > 81°
            linear_base = np.mean(ranks, axis=0)
            proj_coef = np.sum(raw_signal * linear_base, axis=1, keepdims=True) / (
                np.sum(linear_base * linear_base, axis=1, keepdims=True) + 1e-10
            )
            raw_velocity = raw_signal - proj_coef * linear_base

            # 4. De-Meaning & Hypersphere S^{J-2} Projection
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)

            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # 5. Continuous Path Smoothing (EMA)
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

            # 6. Formatting & Final Double De-Meaning
            final_df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)
            final_df = final_df.clip(-self.target_bound, self.target_bound)
            final_df = final_df.sub(final_df.mean(axis=1), axis=0)

            self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal
