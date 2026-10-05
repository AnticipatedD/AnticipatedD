# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn"
# ]
# ///

import numpy as np
import pandas as pd

from predictor import Predictor


class MyPredictor(Predictor):
    """
    Target-Aware Nonlinear Cross-Sectional Basis Strategy.
    Engineered for AlphaNova Cycle 4 MultiIndex Data structures.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # Portfolio controls
        self.target_bound = 0.20
        self.target_l2 = 0.35

        # Execution controls
        self.hysteresis_threshold = 0.18
        self.execution_alpha = 0.40

        # Learned basis parameters
        self.basis_weights = None
        self.basis_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_sectional_rank(values):
        """
        Map every row cross-sectionally into approximately [-1, 1].
        """
        ranked = values.rank(
            axis=1,
            pct=True,
            method="average"
        )

        return ((ranked - 0.5) * 2.0).fillna(0.0).to_numpy(
            dtype=np.float64
        )

    @staticmethod
    def _safe_ic(signal, target):
        """
        Cross-sectional IC averaged over valid observations.
        """
        signal = np.asarray(signal, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        valid = np.isfinite(signal) & np.isfinite(target)

        if valid.sum() < 3:
            return 0.0

        x = signal[valid]
        y = target[valid]

        x = x - x.mean()
        y = y - y.mean()

        sx = np.sqrt(np.sum(x * x))
        sy = np.sqrt(np.sum(y * y))

        if sx < 1e-12 or sy < 1e-12:
            return 0.0

        return float(np.sum(x * y) / (sx * sy))

    @staticmethod
    def _build_basis_from_ranks(ranks, feature_names):
        """
        Build a compact nonlinear basis with shape [time, assets].
        """
        blocks = []

        # First-order components
        for feat in feature_names:
            x = ranks[feat]
            blocks.append(np.tanh(1.25 * x))
            blocks.append(x * np.abs(x))
            blocks.append(np.sin(np.pi * 0.5 * x))

        # Pairwise nonlinear interactions
        for i in range(len(feature_names)):
            x = ranks[feature_names[i]]

            for j in range(i + 1, len(feature_names)):
                y = ranks[feature_names[j]]

                blocks.append(x * y)
                blocks.append(np.abs(x - y) * np.sign(x + y))
                blocks.append(np.sin(np.pi * x * y))
                blocks.append(np.tanh(2.0 * x * y))

        return blocks

    @staticmethod
    def _demean_basis(block):
        """
        Cross-sectional demeaning of one basis component.
        """
        block = np.asarray(block, dtype=np.float64)
        return block - np.nanmean(
            block,
            axis=1,
            keepdims=True
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:

        self.basis_weights = None
        self.basis_count = 0

        if features is None or len(features) == 0:
            self.feature_names = None
            return

        try:
            self.feature_names = list(
                features.columns
                .get_level_values(0)
                .unique()
            )
        except Exception:
            self.feature_names = None
            return

        if len(self.feature_names) < 2:
            return

        # Construct rank space [time, assets]
        ranks = {}
        for feat in self.feature_names:
            try:
                block = features[feat].astype(np.float64)
                ranks[feat] = self._cross_sectional_rank(block)
            except Exception:
                return

        basis = self._build_basis_from_ranks(ranks, self.feature_names)
        if not basis:
            return

        # --------------------------------------------------------------
        # Robust Target Reshaping Matrix Core
        # Corrects the unstacking vectorization error
        # --------------------------------------------------------------
        try:
            if isinstance(target, pd.DataFrame):
                # If target is MultiIndexed (time, ticker) structure, unstack to shape [time, assets]
                if isinstance(target.index, pd.MultiIndex):
                    target_unstacked = target.iloc[:, 0].unstack(level=1)
                elif isinstance(target.columns, pd.MultiIndex):
                    target_unstacked = target.unstack(level=1)
                else:
                    target_unstacked = target
                target_values = target_unstacked.astype(np.float64).to_numpy()
            elif isinstance(target, pd.Series):
                if isinstance(target.index, pd.MultiIndex):
                    target_values = target.unstack(level=1).astype(np.float64).to_numpy()
                else:
                    target_values = target.to_numpy(dtype=np.float64).reshape(-1, 1)
            else:
                target_values = np.asarray(target, dtype=np.float64)
        except Exception:
            # Safe analytical fallback
            self.basis_weights = np.ones(len(basis), dtype=np.float64) / len(basis)
            self.basis_count = len(basis)
            return

        if target_values.ndim == 1:
            target_values = target_values.reshape(-1, 1)

        n_time = min(basis[0].shape[0], target_values.shape[0])
        if n_time < 2:
            return

        # Target-aware IC estimation loop
        raw_weights = []

        for component in basis:
            component = component[:n_time]

            # Dynamically match width boundaries
            if target_values.shape[1] == 1:
                target_block = np.repeat(target_values[:n_time], component.shape[1], axis=1)
            else:
                width = min(component.shape[1], target_values.shape[1])
                component = component[:, :width]
                target_block = target_values[:n_time, :width]

            ics = []
            for t in range(n_time):
                ic = self._safe_ic(component[t], target_block[t])
                if np.isfinite(ic):
                    ics.append(ic)

            if not ics:
                raw_weights.append(0.0)
                continue

            ic_value = float(np.mean(ics))

            # Non-linear signal shrinkage bounds
            magnitude = abs(ic_value)
            if magnitude < 0.01:
                weight = 0.0
            elif magnitude < 0.03:
                weight = ic_value * 0.35
            elif magnitude < 0.06:
                weight = ic_value * 0.70
            else:
                weight = ic_value

            raw_weights.append(weight)

        raw_weights = np.nan_to_num(np.asarray(raw_weights, dtype=np.float64), nan=0.0)

        # Scale and damp extreme basis dominance limits
        max_weight = np.max(np.abs(raw_weights))
        if max_weight > 0:
            raw_weights = np.clip(raw_weights, -max_weight * 0.75, max_weight * 0.75)

        weight_norm = np.sum(np.abs(raw_weights))
        if weight_norm < 1e-12:
            raw_weights = np.ones(len(basis), dtype=np.float64) / len(basis)
        else:
            raw_weights /= weight_norm

        self.basis_weights = raw_weights
        self.basis_count = len(basis)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:

        try:
            # Handle both MultiIndex levels cleanly to grab ticker handles
            if isinstance(features.columns, pd.MultiIndex):
                tickers = features.columns.get_level_values(1).unique()
            else:
                tickers = features.columns
        except Exception:
            return pd.DataFrame(index=features.index, dtype=np.float32)

        # Build zero baseline shell matching time index and cross-sectional tickers
        zero_signal = pd.DataFrame(
            0.0,
            index=features.index,
            columns=tickers,
            dtype=np.float32
        )

        if (
            features is None
            or len(features) == 0
            or self.feature_names is None
            or len(self.feature_names) < 2
        ):
            return zero_signal

        try:
            # 1. Cross-sectional rank matrix mapping
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                ranks[feat] = self._cross_sectional_rank(block)

            # 2. Build non-linear interactions
            basis = self._build_basis_from_ranks(ranks, self.feature_names)
            if not basis:
                return zero_signal

            # 3. Process signal alignment vectors
            if self.basis_weights is None or len(self.basis_weights) != len(basis):
                weights = np.ones(len(basis), dtype=np.float64) / len(basis)
            else:
                weights = self.basis_weights

            raw_signal = np.zeros_like(basis[0], dtype=np.float64)

            for weight, component in zip(weights, basis):
                if abs(weight) < 1e-12:
                    continue
                component = self._demean_basis(component)
                raw_signal += (weight * component)

            Enforce strict cross-sectional neutrality
            raw_signal -= np.nanmean(raw_signal, axis=1, keepdims=True)
          
            # ----------------------------------------------------------
            # 4. (Continued) Handle missing values safely post-demeaning
            # ----------------------------------------------------------
            raw_signal = np.nan_to_num(raw_signal, nan=0.0)
          
            # ----------------------------------------------------------
            # 5. Coordinate sphere L2 normalization pass
            # ----------------------------------------------------------
            norms = np.linalg.norm(raw_signal, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            target_signal = (raw_signal / norms) * self.target_l2

            # ----------------------------------------------------------
            # 6. Causal execution loop via hysteresis tracking bounds
            # ----------------------------------------------------------
            n_time, n_assets = target_signal.shape
            final_positions = np.zeros_like(target_signal)

            state_valid = (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (n_assets,)
                and self.prev_tickers.equals(tickers)
            )

            if state_valid:
                active_position = self.prev_signal.copy()
            else:
                active_position = np.zeros(n_assets, dtype=np.float64)

            for t in range(n_time):
                desired = target_signal[t]
                delta = desired - active_position
                distance = np.linalg.norm(delta)

                if distance <= self.hysteresis_threshold:
                    # No meaningful movement: retain existing portfolio positions
                    current = active_position.copy()
                else:
                    # Controlled movement toward target to limit turnover costs
                    current = (
                        (1.0 - self.execution_alpha) * active_position
                        + self.execution_alpha * desired
                    )

                # Maintain strict structural neutrality cross-sectionally
                current -= np.mean(current)
                final_positions[t] = current
                active_position = current.copy()

            # ----------------------------------------------------------
            # 7. Apply strict risk bounds
            # ----------------------------------------------------------
            final_positions = np.clip(final_positions, -self.target_bound, self.target_bound)

            # ----------------------------------------------------------
            # 8. Final cross-sectional zero net exposure pass
            # ----------------------------------------------------------
            final_positions -= np.mean(final_positions, axis=1, keepdims=True)

            final_df = pd.DataFrame(
                final_positions,
                index=features.index,
                columns=tickers
            )

            # ----------------------------------------------------------
            # 9. Cache running streaming state for next period rebalance checks
            # ----------------------------------------------------------
            if len(final_df) > 0:
                self.prev_signal = final_df.iloc[-1].to_numpy(dtype=np.float64)
                self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception:
            # AlphaNova competition-safe structural fallback
            return zero_signal
