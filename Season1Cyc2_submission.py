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
    Robust nonlinear cross-sectional ensemble.

    Design:
      - cross-sectional rank normalization
      - training-only feature weighting
      - bounded nonlinear basis expansion
      - selected pairwise interactions
      - cross-sectional neutrality
      - adaptive concentration
      - continuous turnover-aware execution

    No future information is used by the training stage.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.feature_weights = None

        self.prev_signal = None
        self.prev_tickers = None

        # Portfolio constraints
        self.position_bound = 0.20

        # Adaptive concentration range
        self.min_concentration = 0.15
        self.max_concentration = 0.30

        # Turnover controls
        self.turnover_soft = 0.35
        self.turnover_hard = 1.12

        # Interaction controls
        self.max_interactions = 32
        self.interaction_weight = 0.25

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_corr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)

        if mask.sum() < 3:
            return 0.0

        x = x[mask]
        y = y[mask]

        x_std = np.std(x)
        y_std = np.std(y)

        if x_std < 1e-12 or y_std < 1e-12:
            return 0.0

        value = np.corrcoef(x, y)[0, 1]

        if not np.isfinite(value):
            return 0.0

        return float(value)

    @staticmethod
    def _rank_block(block):
        """
        Cross-sectional percentile rank.
        Produces approximately [-1, 1].
        """
        ranked = block.rank(
            axis=1,
            pct=True,
            method="average",
        )

        ranked = (ranked - 0.5) * 2.0

        return ranked.fillna(0.0).to_numpy(dtype=np.float64)

    @staticmethod
    def _neutralize(signal):
        """
        Cross-sectional dollar neutrality.
        """
        signal = np.asarray(signal, dtype=np.float64)

        if signal.ndim != 2:
            return signal

        mean = np.mean(signal, axis=1, keepdims=True)

        return signal - mean

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Learn feature weights from training data only.

        If target structure is incompatible with the expected
        cross-sectional format, safely fall back to equal weights.
        """

        self.feature_names = None
        self.feature_weights = None

        if features is None or len(features) == 0:
            return

        # We expect AlphaNova's usual two-level feature columns.
        try:
            if not isinstance(features.columns, pd.MultiIndex):
                return

            feature_level = features.columns.get_level_values(0)

            self.feature_names = list(
                pd.Index(feature_level).unique()
            )

        except Exception:
            self.feature_names = None
            return

        if not self.feature_names:
            return

        # Default = equal weighting.
        weights = np.ones(len(self.feature_names), dtype=np.float64)

        # --------------------------------------------------------------
        # Attempt training-only target alignment.
        # --------------------------------------------------------------

        try:
            if target is None or len(target) == 0:
                self.feature_weights = weights / np.sum(weights)
                return

            # Align target rows to features.
            common_index = features.index.intersection(target.index)

            if len(common_index) < 3:
                self.feature_weights = weights / np.sum(weights)
                return

            f_train = features.loc[common_index]
            t_train = target.loc[common_index]

            # Convert target into a usable numeric matrix.
            if isinstance(t_train, pd.DataFrame):
                target_numeric = t_train.apply(
                    pd.to_numeric,
                    errors="coerce",
                )
            else:
                target_numeric = pd.DataFrame(t_train)

            # Use the first numeric target representation if available.
            target_values = target_numeric.to_numpy(dtype=np.float64)

            if target_values.ndim == 1:
                target_values = target_values[:, None]

            # Aggregate target cross-sectionally when necessary.
            target_series = np.nanmean(target_values, axis=1)

            scores = []

            for feat in self.feature_names:
                try:
                    block = f_train[feat].apply(
                        pd.to_numeric,
                        errors="coerce",
                    )

                    ranked = self._rank_block(block)

                    # Median cross-sectional association.
                    row_scores = []

                    for t in range(len(ranked)):
                        corr = self._safe_corr(
                            ranked[t],
                            np.repeat(
                                target_series[t],
                                ranked.shape[1],
                            ),
                        )

                        # Constant target rows naturally return 0.
                        row_scores.append(corr)

                    score = np.nanmedian(row_scores)

                    if not np.isfinite(score):
                        score = 0.0

                    scores.append(score)

                except Exception:
                    scores.append(0.0)

            scores = np.asarray(scores, dtype=np.float64)

            # ----------------------------------------------------------
            # Stable soft weighting.
            #
            # sqrt(|score|) prevents one noisy feature from dominating.
            # Sign is retained.
            # ----------------------------------------------------------

            magnitude = np.sqrt(np.abs(scores))

            if np.sum(magnitude) > 1e-12:
                weights = np.sign(scores) * magnitude

                # If all scores are effectively zero, equal weighting.
                if np.sum(np.abs(weights)) < 1e-12:
                    weights = np.ones_like(weights)
            else:
                weights = np.ones_like(weights)

        except Exception:
            # Training must never prevent predictor construction.
            weights = np.ones(len(self.feature_names), dtype=np.float64)

        denom = np.sum(np.abs(weights))

        if denom < 1e-12:
            weights = np.ones(len(self.feature_names), dtype=np.float64)
            denom = float(len(self.feature_names))

        self.feature_weights = weights / denom

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:

        # --------------------------------------------------------------
        # Validate columns first.
        # --------------------------------------------------------------

        if features is None:
            return pd.DataFrame()

        try:
            if not isinstance(features.columns, pd.MultiIndex):
                raise ValueError(
                    "Expected a MultiIndex feature DataFrame."
                )

            tickers = features.columns.get_level_values(1).unique()

        except Exception as exc:
            raise RuntimeError(
                f"Invalid AlphaNova feature structure: {exc}"
            ) from exc

        zero_signal = pd.DataFrame(
            0.0,
            index=features.index,
            columns=tickers,
            dtype=np.float32,
        )

        if len(features) == 0:
            return zero_signal

        if not self.feature_names:
            return zero_signal

        # --------------------------------------------------------------
        # Build rank representation.
        # --------------------------------------------------------------

        ranks = {}

        try:
            for feat in self.feature_names:

                if feat not in features.columns.get_level_values(0):
                    continue

                block = features[feat].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                ranks[feat] = self._rank_block(block)

        except Exception as exc:
            raise RuntimeError(
                f"Feature rank transformation failed: {exc}"
            ) from exc

        if len(ranks) == 0:
            return zero_signal

        N_time, N_assets = next(iter(ranks.values())).shape

        # --------------------------------------------------------------
        # Feature weights.
        # --------------------------------------------------------------

        if (
            self.feature_weights is None
            or len(self.feature_weights) != len(self.feature_names)
        ):
            weights = np.ones(
                len(self.feature_names),
                dtype=np.float64,
            )
            weights /= np.sum(np.abs(weights))
        else:
            weights = self.feature_weights

        weight_map = {
            feat: weights[i]
            for i, feat in enumerate(self.feature_names)
        }

        # --------------------------------------------------------------
        # Nonlinear feature ensemble.
        # --------------------------------------------------------------

        feature_components = []

        for feat in self.feature_names:

            if feat not in ranks:
                continue

            r = ranks[feat]
            w = weight_map.get(feat, 0.0)

            # Odd nonlinear component.
            odd = np.tanh(2.0 * r)

            # Centered even component.
            even = r * r
            even -= np.mean(even, axis=1, keepdims=True)

            # Cubic directional response.
            cubic = r ** 3

            # Smooth periodic component.
            periodic = np.sin(
                0.5 * np.pi * r
            )

            component = (
                0.40 * odd
                + 0.20 * even
                + 0.20 * cubic
                + 0.20 * periodic
            )

            feature_components.append(
                w * component
            )

        if not feature_components:
            return zero_signal

        raw_signal = np.sum(
            feature_components,
            axis=0,
        )

        # --------------------------------------------------------------
        # Controlled pairwise interactions.
        # --------------------------------------------------------------

        interaction_candidates = []

        active_features = [
            f for f in self.feature_names
            if f in ranks
        ]

        for i in range(len(active_features)):

            f1 = active_features[i]
            r1 = ranks[f1]

            for j in range(i + 1, len(active_features)):

                f2 = active_features[j]
                r2 = ranks[f2]

                # Bounded multiplicative interaction.
                interaction = np.tanh(
                    2.0 * r1 * r2
                )

                # Measure whether the interaction actually
                # has cross-sectional dispersion.
                quality = float(
                    np.mean(
                        np.abs(interaction)
                    )
                )

                interaction_candidates.append(
                    (
                        quality,
                        f1,
                        f2,
                        interaction,
                    )
                )

        # Retain only the most active interactions.
        interaction_candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        selected = interaction_candidates[
            : self.max_interactions
        ]

        if selected:

            interaction_sum = np.zeros_like(
                raw_signal,
                dtype=np.float64,
            )

            total_quality = 0.0

            for quality, f1, f2, interaction in selected:

                if quality <= 1e-12:
                    continue

                w1 = abs(weight_map.get(f1, 0.0))
                w2 = abs(weight_map.get(f2, 0.0))

                pair_weight = np.sqrt(
                    max(w1 * w2, 0.0)
                )

                contribution = (
                    interaction
                    * pair_weight
                    * quality
                )

                interaction_sum += contribution
                total_quality += quality

            if total_quality > 1e-12:

                interaction_sum /= total_quality

                raw_signal += (
                    self.interaction_weight
                    * interaction_sum
                )

        # --------------------------------------------------------------
        # Cross-sectional neutrality.
        # --------------------------------------------------------------

        raw_signal = self._neutralize(
            raw_signal
        )

        # --------------------------------------------------------------
        # Robust row normalization.
        # --------------------------------------------------------------

        scale = np.sqrt(
            np.mean(
                raw_signal ** 2,
                axis=1,
                keepdims=True,
            )
        )

        scale = np.maximum(
            scale,
            1e-10,
        )

        normalized = raw_signal / scale

        # --------------------------------------------------------------
        # Adaptive concentration.
        # --------------------------------------------------------------

        abs_mean = np.mean(
            np.abs(raw_signal),
            axis=1,
            keepdims=True,
        )

        abs_median = np.median(
            np.abs(raw_signal),
            axis=1,
            keepdims=True,
        )

        quality = abs_median / (
            abs_mean + 1e-10
        )

        quality = np.clip(
            quality,
            0.0,
            1.0,
        )

        concentration = (
            self.min_concentration
            + (
                self.max_concentration
                - self.min_concentration
            )
            * quality
        )

        target_positions = (
            normalized
            * concentration
        )

        # Hard position protection.
        target_positions = np.clip(
            target_positions,
            -self.position_bound,
            self.position_bound,
        )

        target_positions = self._neutralize(
            target_positions
        )

        # --------------------------------------------------------------
        # Stateful turnover-aware execution.
        # --------------------------------------------------------------

        final_positions = np.zeros_like(
            target_positions,
            dtype=np.float64,
        )

        if (
            self.prev_signal is not None
            and self.prev_tickers is not None
            and len(self.prev_signal) == N_assets
            and self.prev_tickers.equals(tickers)
        ):
            active = self.prev_signal.copy()
        else:
            active = np.zeros(
                N_assets,
                dtype=np.float64,
            )

        for t in range(N_time):

            desired = target_positions[t]

            turnover = np.sum(
                np.abs(
                    desired - active
                )
            )

            if turnover <= self.turnover_soft:

                # Small movement: preserve previous portfolio.
                alpha = 0.0

            elif turnover >= self.turnover_hard:

                # Large movement: move meaningfully,
                # but don't jump directly to the target.
                alpha = 0.35

            else:

                # Continuous transition.
                alpha = 0.35 * (
                    (turnover - self.turnover_soft)
                    / (
                        self.turnover_hard
                        - self.turnover_soft
                    )
                )

            current = (
                (1.0 - alpha) * active
                + alpha * desired
            )

            # Maintain neutrality after execution.
            current -= np.mean(current)

            # Maintain hard position constraint.
            current = np.clip(
                current,
                -self.position_bound,
                self.position_bound,
            )

            # Re-center after clipping.
            current -= np.mean(current)

            final_positions[t] = current
            active = current.copy()

        # --------------------------------------------------------------
        # Final safety validation.
        # --------------------------------------------------------------

        final_df = pd.DataFrame(
            final_positions,
            index=features.index,
            columns=tickers,
        )

        final_df = final_df.replace(
            [np.inf, -np.inf],
            0.0,
        ).fillna(0.0)

        final_df = final_df.sub(
            final_df.mean(axis=1),
            axis=0,
        )

        final_df = final_df.clip(
            -self.position_bound,
            self.position_bound,
        )

        final_df = final_df.sub(
            final_df.mean(axis=1),
            axis=0,
        )

        # --------------------------------------------------------------
        # Preserve causal state.
        # --------------------------------------------------------------

        if len(final_df) > 0:

            self.prev_signal = (
                final_df.iloc[-1]
                .to_numpy(dtype=np.float64)
            )

            self.prev_tickers = (
                final_df.columns.copy()
            )

        return final_df.astype(np.float32)
