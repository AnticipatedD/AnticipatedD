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

    Architecture:
        1. Cross-sectional rank normalization
        2. Nonlinear basis expansion
        3. Target-aware IC estimation during train()
        4. Shrunk predictive weighting of basis components
        5. Cross-sectional neutralization
        6. Robust portfolio normalization
        7. Causal turnover-controlled execution

    The important difference from the previous architecture is that
    nonlinear components are no longer averaged with equal weights.
    Their historical cross-sectional relationship with the target
    determines their contribution to the final signal.
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

        Each time slice is demeaned before correlation. This makes the
        statistic insensitive to common cross-sectional shifts.
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
        Build a compact nonlinear basis.

        Each component has shape:
            [time, assets]
        """
        blocks = []

        # --------------------------------------------------------------
        # First-order nonlinear components
        # --------------------------------------------------------------
        for feat in feature_names:
            x = ranks[feat]

            blocks.append(
                np.tanh(1.25 * x)
            )

            blocks.append(
                x * np.abs(x)
            )

            blocks.append(
                np.sin(np.pi * 0.5 * x)
            )

        # --------------------------------------------------------------
        # Pairwise nonlinear interactions
        # --------------------------------------------------------------
        for i in range(len(feature_names)):
            x = ranks[feature_names[i]]

            for j in range(i + 1, len(feature_names)):
                y = ranks[feature_names[j]]

                # Multiplicative interaction
                blocks.append(x * y)

                # Absolute-disparity interaction
                blocks.append(
                    np.abs(x - y) * np.sign(x + y)
                )

                # Phase interaction
                blocks.append(
                    np.sin(np.pi * x * y)
                )

                # Agreement/disagreement geometry
                blocks.append(
                    np.tanh(2.0 * x * y)
                )

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

        # --------------------------------------------------------------
        # Recover anonymized feature names
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # Construct rank space
        # --------------------------------------------------------------
        ranks = {}

        for feat in self.feature_names:
            try:
                block = features[feat].astype(np.float64)
                ranks[feat] = self._cross_sectional_rank(block)
            except Exception:
                return

        # --------------------------------------------------------------
        # Construct nonlinear candidate basis
        # --------------------------------------------------------------
        basis = self._build_basis_from_ranks(
            ranks,
            self.feature_names
        )

        if not basis:
            return

        # --------------------------------------------------------------
        # Convert target into an aligned numerical matrix
        # --------------------------------------------------------------
        try:
            if isinstance(target, pd.DataFrame):

                if isinstance(target.columns, pd.MultiIndex):
                    target_values = target.iloc[:, 0].to_numpy(
                        dtype=np.float64
                    )

                    # If target has one column per ticker, preserve it.
                    if target.shape[1] > 1:
                        target_values = target.to_numpy(
                            dtype=np.float64
                        )

                else:
                    target_values = target.to_numpy(
                        dtype=np.float64
                    )

            else:
                target_values = np.asarray(
                    target,
                    dtype=np.float64
                )

        except Exception:
            self.basis_weights = np.ones(
                len(basis),
                dtype=np.float64
            )
            self.basis_weights /= np.sum(
                np.abs(self.basis_weights)
            )
            self.basis_count = len(basis)
            return

        # --------------------------------------------------------------
        # Normalize target shape
        # --------------------------------------------------------------
        if target_values.ndim == 1:
            target_values = target_values.reshape(-1, 1)

        n_time = min(
            basis[0].shape[0],
            target_values.shape[0]
        )

        if n_time < 2:
            return

        # --------------------------------------------------------------
        # Target-aware IC estimation
        # --------------------------------------------------------------
        raw_weights = []

        for component in basis:

            component = component[:n_time]

            # Align target dimensions.
            if target_values.shape[1] == 1:
                target_block = np.repeat(
                    target_values[:n_time],
                    component.shape[1],
                    axis=1
                )
            else:
                width = min(
                    component.shape[1],
                    target_values.shape[1]
                )

                component = component[:, :width]
                target_block = target_values[
                    :n_time,
                    :width
                ]

            # Calculate mean cross-sectional IC.
            ics = []

            for t in range(n_time):
                ic = self._safe_ic(
                    component[t],
                    target_block[t]
                )

                if np.isfinite(ic):
                    ics.append(ic)

            if not ics:
                raw_weights.append(0.0)
                continue

            ic_value = float(np.mean(ics))

            # ----------------------------------------------------------
            # Shrink weak signals toward zero.
            #
            # This prevents tiny random IC values from becoming
            # disproportionately important.
            # ----------------------------------------------------------
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

        raw_weights = np.asarray(
            raw_weights,
            dtype=np.float64
        )

        # --------------------------------------------------------------
        # Robust weighting
        # --------------------------------------------------------------
        if not np.any(np.isfinite(raw_weights)):
            return

        raw_weights = np.nan_to_num(
            raw_weights,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Clip extreme basis dominance.
        max_weight = np.max(
            np.abs(raw_weights)
        )

        if max_weight > 0:
            raw_weights = np.clip(
                raw_weights,
                -max_weight * 0.75,
                max_weight * 0.75
            )

        weight_norm = np.sum(
            np.abs(raw_weights)
        )

        if weight_norm < 1e-12:
            # No useful learned relationship.
            # Keep a small diversified fallback.
            raw_weights = np.ones(
                len(basis),
                dtype=np.float64
            )

            raw_weights /= np.sum(
                np.abs(raw_weights)
            )

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
            tickers = features.columns.get_level_values(
                1
            ).unique()

        except Exception:
            return pd.DataFrame(
                index=features.index,
                dtype=np.float32
            )

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

            # ----------------------------------------------------------
            # 1. Cross-sectional rank transformation
            # ----------------------------------------------------------

            ranks = {}

            for feat in self.feature_names:
                block = features[feat].astype(np.float64)

                ranks[feat] = self._cross_sectional_rank(
                    block
                )

            # ----------------------------------------------------------
            # 2. Build nonlinear basis
            # ----------------------------------------------------------

            basis = self._build_basis_from_ranks(
                ranks,
                self.feature_names
            )

            if not basis:
                return zero_signal

            # ----------------------------------------------------------
            # 3. Apply learned predictive weights
            # ----------------------------------------------------------

            if (
                self.basis_weights is None
                or len(self.basis_weights) != len(basis)
            ):
                weights = np.ones(
                    len(basis),
                    dtype=np.float64
                )

                weights /= np.sum(
                    np.abs(weights)
                )

            else:
                weights = self.basis_weights

            raw_signal = np.zeros_like(
                basis[0],
                dtype=np.float64
            )

            for weight, component in zip(
                weights,
                basis
            ):
                if abs(weight) < 1e-12:
                    continue

                component = self._demean_basis(
                    component
                )

                raw_signal += (
                    weight * component
                )

            # ----------------------------------------------------------
            # 4. Robust cross-sectional demeaning
            # ----------------------------------------------------------

            raw_signal -= np.nanmean(
                raw_signal,
                axis=1,
                keepdims=True
            )

            raw_signal = np.nan_to_num(
                raw_signal,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # ----------------------------------------------------------
            # 5. L2 portfolio normalization
            # ----------------------------------------------------------

            norms = np.linalg.norm(
                raw_signal,
                axis=1,
                keepdims=True
            )

            norms = np.where(
                norms < 1e-10,
                1.0,
                norms
            )

            target_signal = (
                raw_signal / norms
            ) * self.target_l2

            # ----------------------------------------------------------
            # 6. Causal execution / hysteresis
            # ----------------------------------------------------------

            n_time, n_assets = target_signal.shape

            final_positions = np.zeros_like(
                target_signal
            )

            state_valid = (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (n_assets,)
                and self.prev_tickers.equals(tickers)
            )

            if state_valid:
                active_position = (
                    self.prev_signal.copy()
                )
            else:
                active_position = np.zeros(
                    n_assets,
                    dtype=np.float64
                )

            for t in range(n_time):

                desired = target_signal[t]

                delta = desired - active_position

                distance = np.linalg.norm(
                    delta
                )

                if distance <= self.hysteresis_threshold:

                    # No meaningful movement:
                    # retain existing portfolio.
                    current = active_position.copy()

                else:

                    # Controlled movement toward target.
                    current = (
                        (1.0 - self.execution_alpha)
                        * active_position
                        + self.execution_alpha
                        * desired
                    )

                # Structural neutrality.
                current -= np.mean(
                    current
                )

                final_positions[t] = current

                active_position = current.copy()

            # ----------------------------------------------------------
            # 7. Position bounds
            # ----------------------------------------------------------

            final_positions = np.clip(
                final_positions,
                -self.target_bound,
                self.target_bound
            )

            # ----------------------------------------------------------
            # 8. Final neutrality correction
            # ----------------------------------------------------------

            final_positions -= np.mean(
                final_positions,
                axis=1,
                keepdims=True
            )

            final_df = pd.DataFrame(
                final_positions,
                index=features.index,
                columns=tickers
            )

            # ----------------------------------------------------------
            # 9. Persist streaming state
            # ----------------------------------------------------------

            if len(final_df) > 0:
                self.prev_signal = (
                    final_df.iloc[-1]
                    .to_numpy(dtype=np.float64)
                )

                self.prev_tickers = (
                    final_df.columns.copy()
                )

            return final_df.astype(
                np.float32
            )

        except Exception:
            # Competition-safe fallback.
            return zero_signal
