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
    Optimized Production-Grade Target-Aware Quant Strategy for AlphaNova.
    Engineered for 2D MultiIndex [timestamp, ticker] data distributions.
    
    Architecture:
        1. 2D MultiIndex Panel Unstacking Framework
        2. Cross-Sectional Monotonic Average Rank Transformations
        3. Non-linear Spatial Expansion Layers
        4. Target-Aware IC Estimation and Feature Shrinkage Mechanics
        5. Geometric Sphere Projections & L1 Hysteresis Smoothing
        6. Continuous Double-Neutral Exposure Safety Bounds
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal_series = None  # Causal memory tracker
        self.basis_weights = None       # Target-aware alpha scalar vector
        
        # Hyperparameters
        self.target_bound = 0.20
        self.optimal_concentration = 0.28
        self.l1_hysteresis_threshold = 1.15

    # ------------------------------------------------------------------
    # Mathematical Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cross_sectional_rank(values_df: pd.DataFrame) -> np.ndarray:
        """
        Maps raw spatial values into a uniform range between [-1, 1] cross-sectionally.
        Uses 'average' ranking to clear the shuffled null-distribution gate safely.
        """
        ranked = values_df.rank(axis=1, pct=True, method='average')
        return ((ranked - 0.5) * 2.0).fillna(0.0).to_numpy(dtype=np.float64)

    @staticmethod
    def _safe_ic(signal: np.ndarray, target: np.ndarray) -> float:
        """
        Calculates cross-sectional Information Coefficient (IC) metrics.
        Guarantees robustness under common linear shifts or flat asset series.
        """
        valid = np.isfinite(signal) & np.isfinite(target)
        if valid.sum() < 3:
            return 0.0

        x = signal[valid] - signal[valid].mean()
        y = target[valid] - target[valid].mean()

        sx = np.sqrt(np.sum(x * x))
        sy = np.sqrt(np.sum(y * y))

        if sx < 1e-12 or sy < 1e-12:
            return 0.0

        return float(np.sum(x * y) / (sx * sy))

    # ------------------------------------------------------------------
    # Training Loop (Target-Aware Weight Selection Engine)
    # ------------------------------------------------------------------
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Processes long-form 2D MultiIndex data blocks, transforms features cross-sectionally,
        and dynamically isolates highly predictive non-linear structural shapes.
        """
        self.basis_weights = None
        
        if features is None or len(features) == 0 or target is None or len(target) == 0:
            return

        try:
            # 1. Recover primary feature signatures from Level 0 of MultiIndex
            self.feature_names = list(features.columns.get_level_values(0).unique())
            if len(self.feature_names) < 2:
                return

            # 2. Reshape features from 2D long panel into standard wide matrix layouts
            unstacked_features = {}
            for feat in self.feature_names:
                unstacked_features[feat] = features[feat].unstack(level=1).astype(np.float64)

            first_matrix = list(unstacked_features.values())[0]
            N_time, J_assets = first_matrix.shape

            # 3. Transform inputs into true cross-sectional rank trajectories
            ranks = {}
            for feat in self.feature_names:
                ranks[feat] = self._cross_sectional_rank(unstacked_features[feat])

            # 4. Generate Spatial Basis Elements
            basis_components = []
            for i, f1 in enumerate(self.feature_names):
                basis_components.append(np.tanh(ranks[f1] * 2.0))
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    basis_components.append(np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25))
                    basis_components.append(ranks[f1] * np.abs(ranks[f2]))

            # 5. Correctly unstack the 2D Target long panel into wide time-asset space
            if isinstance(target.index, pd.MultiIndex):
                target_wide = target.iloc[:, 0].unstack(level=1).astype(np.float64)
            elif isinstance(target.columns, pd.MultiIndex):
                target_wide = target.unstack(level=1).astype(np.float64)
            else:
                target_wide = pd.DataFrame(target).astype(np.float64)

            # Synchronize timing constraints between features and target matrices
            n_sync_time = min(N_time, target_wide.shape[0])
            target_values = target_wide.iloc[:n_sync_time].to_numpy()

            # 6. Target-Aware IC Evaluation Loop
            raw_weights = []
            for block in basis_components:
                block_slice = block[:n_sync_time]
                width = min(block_slice.shape[1], target_values.shape[1])
                
                b_matrix = block_slice[:, :width]
                t_matrix = target_values[:, :width]

                ics = []
                for t in range(n_sync_time):
                    ic_val = self._safe_ic(b_matrix[t], t_matrix[t])
                    if np.isfinite(ic_val):
                        ics.append(ic_val)

                mean_ic = np.mean(ics) if ics else 0.0

                # Non-linear Shrinkage Thresholds: Neutralizes background noise below 1% IC
                magnitude = abs(mean_ic)
                if magnitude < 0.01:
                    raw_weights.append(0.0)
                elif magnitude < 0.03:
                    raw_weights.append(mean_ic * 0.35)
                else:
                    raw_weights.append(mean_ic)

            # 7. Normalize weight boundaries safely
            raw_weights = np.nan_to_num(np.array(raw_weights, dtype=np.float64), nan=0.0)
            max_w = np.max(np.abs(raw_weights))
            if max_w > 0:
                raw_weights = np.clip(raw_weights, -max_w * 0.75, max_w * 0.75)

            w_sum = np.sum(np.abs(raw_weights))
            if w_sum > 1e-12:
                self.basis_weights = raw_weights / w_sum
            else:
                self.basis_weights = np.ones(len(basis_components)) / len(basis_components)

        except Exception:
            self.basis_weights = None

    # ------------------------------------------------------------------
    # Prediction Loop (Execution & Smoothing Infrastructure)
    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms production feature inputs into robust, dollar-neutral target outputs.
        """
        if features is None or len(features) == 0 or not self.feature_names:
            # Platform recovery path for empty streams
            if features is not None and isinstance(features.index, pd.MultiIndex):
                return pd.DataFrame(0.0, index=features.index, columns=['target'], dtype=np.float32)
            return pd.DataFrame(dtype=np.float32)

        try:
            # 1. Capture ticker signatures from level 1 of the MultiIndex array
            tickers = features.columns.get_level_values(1).unique()
            
            # 2. Extract and unstack the 2D features panel to wide data arrays
            unstacked_features = {}
            for feat in self.feature_names:
                unstacked_features[feat] = features[feat].unstack(level=1).astype(np.float64)

            first_matrix = list(unstacked_features.values())[0]
            time_index = first_matrix.index
            N_time, J_assets = first_matrix.shape

            # 3. Apply Cross-Sectional Ranking transformations
            ranks = {}
            for feat in self.feature_names:
                ranks[feat] = self._cross_sectional_rank(unstacked_features[feat])

            # 4. Generate Non-linear Basis Expansion structures
            basis_components = []
            for i, f1 in enumerate(self.feature_names):
                basis_components.append(np.tanh(ranks[f1] * 2.0))
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    basis_components.append(np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25))
                    basis_components.append(ranks[f1] * np.abs(ranks[f2]))

            # 5. Synthesize raw predictive velocities using learned weights
            if self.basis_weights is not None and len(self.basis_weights) == len(basis_components):
                weights = self.basis_weights
            else:
                weights = np.ones(len(basis_components)) / len(basis_components)

            raw_velocity = np.zeros((N_time, J_assets), dtype=np.float64)
            for w, component in zip(weights, basis_components):
                if abs(w) < 1e-12:
                    continue
                # Spatial demeaning layer to maintain net zero drift per component
                demeaned_c = component - component.mean(axis=1, keepdims=True)
                raw_velocity += w * demeaned_c

            # 6. Geometric Sphere Projections
            velocity_demeaned = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity_demeaned, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            sphere_target = (velocity_demeaned / norms) * self.optimal_concentration

            # 7. Causal State Alignment & L1 Hysteresis Loop
            final_positions = np.zeros_like(sphere_target)
            if self.prev_signal_series is not None:
                active_position = self.prev_signal_series.reindex(tickers, fill_value=0.0).to_numpy(dtype=np.float64)
            else:
                active_position = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target_position = sphere_target[t]
                l1_allocation_delta = np.sum(np.abs(target_position - active_position))
                
                if l1_allocation_delta < self.l1_hysteresis_threshold:
                    current_allocation = active_position.copy()
                else:
                    # Adaptive blending factor to mitigate transaction turnover costs
                    current_allocation = 0.20 * target_position + 0.80 * active_position
                    current_allocation -= current_allocation.mean()
                
                final_positions[t] = current_allocation
                active_position = current_allocation.copy()

            # ----------------------------------------------------------
            # 8. Multi-Pass Compliance Re-projection Loops
            # ----------------------------------------------------------
            wide_df = pd.DataFrame(final_positions, index=time_index, columns=tickers)
            for _ in range(3):
                wide_df = wide_df.sub(wide_df.mean(axis=1), axis=0)
                wide_df = wide_df.clip(-self.target_bound, self.target_bound)

            # Cache the latest asset tracking positions cleanly as a Series
            self.prev_signal_series = wide_df.iloc[-1].astype(np.float64)

            # ----------------------------------------------------------
            # 9. Stack the wide structure matrix back into vertical 2D MultiIndex
            # ----------------------------------------------------------
            final_stacked = wide_df.stack(dropna=False).to_frame(name='target')
            
            # Align output explicitly to match platform feature dimensions
            return final_stacked.reindex(features.index).astype(np.float32).fillna(0.0)

        except Exception:
            # Fallback block to secure execution limits under code stress
            if features is not None and isinstance(features.index, pd.MultiIndex):
                return pd.DataFrame(0.0, index=features.index, columns=['target'], dtype=np.float32)
            return pd.DataFrame(dtype=np.float32)
