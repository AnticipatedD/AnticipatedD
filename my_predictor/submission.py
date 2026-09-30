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
    Sharpe-oriented cross-sectional predictor.

    Design:
        1. Rank-normalize features cross-sectionally.
        2. Construct several economically different signal families.
        3. Estimate each family's historical relationship with target.
        4. Use risk-adjusted IC weighting instead of equal weighting.
        5. Combine only stable components.
        6. Volatility-scale the resulting signal.
        7. Enforce cross-sectional neutrality.
        8. Apply moderate turnover control.

    Primary objective:
        maximize the stability of the realized risk-adjusted signal.

    No claim is made that Sharpe is guaranteed.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.signal_weights = None
        self.signal_count = 0

        # Portfolio controls
        self.target_bound = 0.20
        self.target_risk = 0.30

        # Execution controls
        self.hysteresis_threshold = 0.12
        self.execution_alpha = 0.55

    # ================================================================
    # BASIC OPERATIONS
    # ================================================================

    @staticmethod
    def _rank(x):
        """
        Cross-sectional percentile rank mapped to [-1, 1].
        """
        x = x.astype(np.float64)

        r = x.rank(
            axis=1,
            pct=True,
            method="average"
        )

        return (
            (r - 0.5) * 2.0
        ).fillna(0.0).to_numpy(
            dtype=np.float64
        )

    @staticmethod
    def _demean(x):
        return x - np.nanmean(
            x,
            axis=1,
            keepdims=True
        )

    @staticmethod
    def _safe_corr(x, y):
        """
        Pearson correlation between two cross-sectional vectors.
        """
        mask = (
            np.isfinite(x)
            & np.isfinite(y)
        )

        if np.sum(mask) < 4:
            return 0.0

        x = x[mask]
        y = y[mask]

        x = x - np.mean(x)
        y = y - np.mean(y)

        denom = (
            np.sqrt(np.sum(x * x))
            * np.sqrt(np.sum(y * y))
        )

        if denom < 1e-12:
            return 0.0

        return float(
            np.sum(x * y) / denom
        )

    @classmethod
    def _mean_ic(cls, signal, target):
        """
        Mean cross-sectional IC.
        """
        n = min(
            signal.shape[0],
            target.shape[0]
        )

        values = []

        for t in range(n):
            values.append(
                cls._safe_corr(
                    signal[t],
                    target[t]
                )
            )

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    @classmethod
    def _ic_information_ratio(cls, signal, target):
        """
        IC information ratio:

            mean(IC) / std(IC)

        This favors signals whose predictive relationship is
        reasonably stable rather than merely occasionally strong.
        """
        n = min(
            signal.shape[0],
            target.shape[0]
        )

        ics = []

        for t in range(n):
            ics.append(
                cls._safe_corr(
                    signal[t],
                    target[t]
                )
            )

        if len(ics) < 5:
            return 0.0

        ics = np.asarray(
            ics,
            dtype=np.float64
        )

        mean_ic = np.mean(ics)
        std_ic = np.std(ics)

        if std_ic < 1e-8:
            return (
                float(mean_ic) * 10.0
            )

        return float(
            mean_ic / std_ic
        )

    # ================================================================
    # SIGNAL GENERATION
    # ================================================================

    @staticmethod
    def _build_signals(ranks, names):
        """
        Build a deliberately compact collection of structurally
        different candidate signals.

        Returns:
            list[np.ndarray]
        """

        signals = []

        # ------------------------------------------------------------
        # 1. Robust linear consensus
        # ------------------------------------------------------------

        matrix = np.stack(
            [ranks[n] for n in names],
            axis=0
        )

        linear_consensus = np.nanmedian(
            matrix,
            axis=0
        )

        signals.append(
            linear_consensus
        )

        # ------------------------------------------------------------
        # 2. Dispersion / disagreement
        # ------------------------------------------------------------

        feature_mean = np.nanmean(
            matrix,
            axis=0
        )

        feature_dispersion = np.nanmean(
            np.abs(
                matrix - feature_mean[None, :, :]
            ),
            axis=0
        )

        signals.append(
            -feature_dispersion
            * np.sign(feature_mean)
        )

        # ------------------------------------------------------------
        # 3. Feature agreement
        # ------------------------------------------------------------

        agreement = np.prod(
            np.sign(
                matrix + 1e-12
            ),
            axis=0
        )

        magnitude = np.nanmean(
            np.abs(matrix),
            axis=0
        )

        signals.append(
            agreement * magnitude
        )

        # ------------------------------------------------------------
        # 4. Energy imbalance
        # ------------------------------------------------------------

        positive_energy = np.mean(
            np.maximum(matrix, 0.0) ** 2,
            axis=0
        )

        negative_energy = np.mean(
            np.minimum(matrix, 0.0) ** 2,
            axis=0
        )

        signals.append(
            positive_energy
            - negative_energy
        )

        # ------------------------------------------------------------
        # 5. Nonlinear consensus
        # ------------------------------------------------------------

        nonlinear = np.mean(
            np.tanh(
                1.5 * matrix
            ),
            axis=0
        )

        signals.append(
            nonlinear
        )

        # ------------------------------------------------------------
        # 6. Quadratic asymmetry
        # ------------------------------------------------------------

        quadratic = np.mean(
            matrix * np.abs(matrix),
            axis=0
        )

        signals.append(
            quadratic
        )

        # ------------------------------------------------------------
        # 7. Cross-feature covariance direction
        # ------------------------------------------------------------

        if len(names) >= 2:

            covariance_signal = np.zeros_like(
                matrix[0]
            )

            pair_count = 0

            for i in range(len(names)):
                for j in range(i + 1, len(names)):

                    x = matrix[i]
                    y = matrix[j]

                    covariance_signal += (
                        x * y
                    )

                    pair_count += 1

            if pair_count > 0:
                covariance_signal /= pair_count

            signals.append(
                covariance_signal
            )

        return [
            np.nan_to_num(
                s,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
            for s in signals
        ]

    # ================================================================
    # TRAINING
    # ================================================================

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:

        self.signal_weights = None
        self.signal_count = 0

        if (
            features is None
            or len(features) == 0
        ):
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

        # ------------------------------------------------------------
        # Rank features
        # ------------------------------------------------------------

        ranks = {}

        for name in self.feature_names:

            try:
                ranks[name] = self._rank(
                    features[name]
                )
            except Exception:
                return

        # ------------------------------------------------------------
        # Candidate signals
        # ------------------------------------------------------------

        candidates = self._build_signals(
            ranks,
            self.feature_names
        )

        if not candidates:
            return

        # ------------------------------------------------------------
        # Target conversion
        # ------------------------------------------------------------

        try:

            if isinstance(target, pd.DataFrame):
                target_values = target.to_numpy(
                    dtype=np.float64
                )
            else:
                target_values = np.asarray(
                    target,
                    dtype=np.float64
                )

        except Exception:
            return

        if target_values.ndim == 1:
            target_values = (
                target_values[:, None]
            )

        # ------------------------------------------------------------
        # Align target width
        # ------------------------------------------------------------

        scores = []

        for signal in candidates:

            n = min(
                signal.shape[0],
                target_values.shape[0]
            )

            signal_part = signal[:n]

            if target_values.shape[1] == 1:

                target_part = np.repeat(
                    target_values[:n],
                    signal_part.shape[1],
                    axis=1
                )

            else:

                width = min(
                    signal_part.shape[1],
                    target_values.shape[1]
                )

                signal_part = (
                    signal_part[:, :width]
                )

                target_part = (
                    target_values[
                        :n,
                        :width
                    ]
                )

            # --------------------------------------------------------
            # Predictive strength
            # --------------------------------------------------------

            mean_ic = self._mean_ic(
                signal_part,
                target_part
            )

            # --------------------------------------------------------
            # Stability of predictive relationship
            # --------------------------------------------------------

            ic_ir = self._ic_information_ratio(
                signal_part,
                target_part
            )

            # --------------------------------------------------------
            # Combined score
            #
            # Mean IC remains dominant, while IC stability prevents
            # a single unstable component from dominating.
            # --------------------------------------------------------

            score = (
                0.70 * mean_ic
                + 0.30 * (
                    ic_ir
                    * 0.02
                )
            )

            scores.append(
                score
            )

        scores = np.asarray(
            scores,
            dtype=np.float64
        )

        scores = np.nan_to_num(
            scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ------------------------------------------------------------
        # Remove statistically weak directions
        # ------------------------------------------------------------

        threshold = 0.002

        scores[
            np.abs(scores) < threshold
        ] = 0.0

        # ------------------------------------------------------------
        # Robust clipping
        # ------------------------------------------------------------

        if np.any(scores != 0):

            scale = np.percentile(
                np.abs(scores[scores != 0]),
                75
            )

            if scale > 1e-12:

                scores = np.clip(
                    scores,
                    -2.5 * scale,
                    2.5 * scale
                )

        # ------------------------------------------------------------
        # Normalize weights
        # ------------------------------------------------------------

        norm = np.sum(
            np.abs(scores)
        )

        if norm < 1e-12:

            # No measurable predictive direction.
            #
            # A diversified fallback is preferable to producing
            # pathological concentrations.
            weights = np.ones(
                len(candidates),
                dtype=np.float64
            )

            weights /= np.sum(
                np.abs(weights)
            )

        else:

            weights = (
                scores / norm
            )

        self.signal_weights = weights
        self.signal_count = len(candidates)

    # ================================================================
    # PREDICT
    # ================================================================

    def predict(
        self,
        features: pd.DataFrame
    ) -> pd.DataFrame:

        try:

            tickers = (
                features.columns
                .get_level_values(1)
                .unique()
            )

        except Exception:

            return pd.DataFrame(
                index=features.index,
                dtype=np.float32
            )

        zero = pd.DataFrame(
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
            return zero

        try:

            # --------------------------------------------------------
            # Rank features
            # --------------------------------------------------------

            ranks = {}

            for name in self.feature_names:

                ranks[name] = self._rank(
                    features[name]
                )

            # --------------------------------------------------------
            # Generate candidate signals
            # --------------------------------------------------------

            candidates = self._build_signals(
                ranks,
                self.feature_names
            )

            if not candidates:
                return zero

            # --------------------------------------------------------
            # Learned weights
            # --------------------------------------------------------

            if (
                self.signal_weights is None
                or len(self.signal_weights)
                != len(candidates)
            ):

                weights = np.ones(
                    len(candidates),
                    dtype=np.float64
                )

                weights /= np.sum(
                    np.abs(weights)
                )

            else:

                weights = self.signal_weights

            # --------------------------------------------------------
            # Weighted signal
            # --------------------------------------------------------

            signal = np.zeros_like(
                candidates[0],
                dtype=np.float64
            )

            for weight, candidate in zip(
                weights,
                candidates
            ):

                candidate = self._demean(
                    candidate
                )

                signal += (
                    weight * candidate
                )

            signal = np.nan_to_num(
                signal,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # --------------------------------------------------------
            # Cross-sectional neutralization
            # --------------------------------------------------------

            signal = self._demean(
                signal
            )

            # --------------------------------------------------------
            # Cross-sectional volatility normalization
            #
            # Instead of simply fixing the L2 norm, scale each
            # cross-section according to its dispersion.
            # --------------------------------------------------------

            row_std = np.std(
                signal,
                axis=1,
                keepdims=True
            )

            row_std = np.where(
                row_std < 1e-8,
                1.0,
                row_std
            )

            signal = (
                signal / row_std
            )

            # Robustly bound extreme rows.
            signal = np.clip(
                signal,
                -4.0,
                4.0
            )

            # Convert to portfolio scale.
            row_norm = np.linalg.norm(
                signal,
                axis=1,
                keepdims=True
            )

            row_norm = np.where(
                row_norm < 1e-10,
                1.0,
                row_norm
            )

            target = (
                signal
                / row_norm
                * self.target_risk
            )

            # --------------------------------------------------------
            # Causal execution
            # --------------------------------------------------------

            n_time, n_assets = (
                target.shape
            )

            output = np.zeros_like(
                target
            )

            state_valid = (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape
                == (n_assets,)
                and self.prev_tickers.equals(
                    tickers
                )
            )

            if state_valid:

                active = (
                    self.prev_signal.copy()
                )

            else:

                active = np.zeros(
                    n_assets,
                    dtype=np.float64
                )

            for t in range(n_time):

                desired = target[t]

                distance = np.linalg.norm(
                    desired - active
                )

                if (
                    distance
                    <= self.hysteresis_threshold
                ):

                    current = active.copy()

                else:

                    current = (
                        (1.0 - self.execution_alpha)
                        * active
                        + self.execution_alpha
                        * desired
                    )

                current -= np.mean(
                    current
                )

                output[t] = current

                active = current.copy()

            # --------------------------------------------------------
            # Final portfolio constraints
            # --------------------------------------------------------

            output = np.clip(
                output,
                -self.target_bound,
                self.target_bound
            )

            output -= np.mean(
                output,
                axis=1,
                keepdims=True
            )

            result = pd.DataFrame(
                output,
                index=features.index,
                columns=tickers
            )

            # --------------------------------------------------------
            # Persist state
            # --------------------------------------------------------

            if len(result) > 0:

                self.prev_signal = (
                    result.iloc[-1]
                    .to_numpy(
                        dtype=np.float64
                    )
                )

                self.prev_tickers = (
                    result.columns.copy()
                )

            return result.astype(
                np.float32
            )

        except Exception:

            return zero
