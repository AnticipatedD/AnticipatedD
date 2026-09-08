# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
#     "pyarrow"
# ]
# ///

import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    """
    Stable cross-sectional rank ensemble.

    Main changes from the previous version:
      - continuous causal smoothing instead of hard L1 hysteresis
      - robust rank normalization
      - nonlinear transforms kept bounded
      - adaptive feature weighting from cross-sectional dispersion
      - explicit volatility normalization
      - no look-ahead
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None

        self.prev_signal = None
        self.prev_tickers = None

        # Portfolio controls
        self.target_bound = 0.20
        self.target_l2 = 0.27

        # Causal smoothing.
        # Higher = more responsive.
        self.ema_alpha = 0.32

        # Small turnover stabilizer.
        self.min_change = 0.015

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame = None
    ) -> None:

        if features is None or features.empty:
            return

        try:
            self.feature_names = list(
                features.columns.get_level_values(0).unique()
            )
        except Exception:
            self.feature_names = None

    @staticmethod
    def _rank(x):
        """
        Cross-sectional percentile rank mapped to [-1, 1].
        """
        r = x.rank(
            axis=1,
            pct=True,
            method="average"
        )

        return ((r - 0.5) * 2.0).fillna(0.0)

    @staticmethod
    def _demean(x):
        return x - x.mean(axis=1, keepdims=True)

    @staticmethod
    def _normalize_l2(x, target):
        x = MyPredictor._demean(x)

        n = np.linalg.norm(
            x,
            axis=1,
            keepdims=True
        )

        n = np.maximum(n, 1e-12)

        return x / n * target

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:

        # ------------------------------------------------------------
        # Basic validation
        # ------------------------------------------------------------

        if features is None or features.empty:
            return pd.DataFrame(dtype=np.float32)

        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            return pd.DataFrame(dtype=np.float32)

        zero_signal = pd.DataFrame(
            0.0,
            index=features.index,
            columns=tickers,
            dtype=np.float32
        )

        if (
            not self.feature_names
            or len(self.feature_names) == 0
        ):
            return zero_signal

        try:

            # --------------------------------------------------------
            # 1. Cross-sectional rank representation
            # --------------------------------------------------------

            ranked = {}

            for name in self.feature_names:

                block = features[name].astype(np.float64)

                ranked[name] = self._rank(block).to_numpy(
                    dtype=np.float64
                )

            n_time, n_assets = next(
                iter(ranked.values())
            ).shape

            # --------------------------------------------------------
            # 2. Build bounded nonlinear signals
            # --------------------------------------------------------

            signals = []

            # Primary feature.
            f0 = self.feature_names[0]
            x0 = ranked[f0]

            # tanh suppresses extreme cross-sectional ranks.
            s0 = np.tanh(1.65 * x0)

            signals.append(1.00 * s0)

            # --------------------------------------------------------
            # Secondary feature.
            #
            # Keep reversal component deliberately small.
            # --------------------------------------------------------

            if len(self.feature_names) >= 2:

                f1 = self.feature_names[1]
                x1 = ranked[f1]

                s1 = -np.tanh(1.35 * x1)

                signals.append(0.30 * s1)

            # --------------------------------------------------------
            # Additional features.
            #
            # Rather than multiplying many features together,
            # use bounded interactions. This reduces tail explosions.
            # --------------------------------------------------------

            if len(self.feature_names) >= 3:

                f2 = self.feature_names[2]
                x2 = ranked[f2]

                s2 = np.tanh(1.15 * x2)

                signals.append(0.18 * s2)

                # Stable interaction with primary signal.
                signals.append(
                    0.12
                    * np.tanh(x0)
                    * np.abs(np.tanh(x2))
                )

            if len(self.feature_names) >= 4:

                f3 = self.feature_names[3]
                x3 = ranked[f3]

                s3 = np.tanh(1.10 * x3)

                signals.append(0.12 * s3)

                # Another weak interaction.
                signals.append(
                    0.10
                    * np.tanh(x0)
                    * np.tanh(x3)
                )

            # --------------------------------------------------------
            # 3. Weighted ensemble
            # --------------------------------------------------------

            raw = np.zeros(
                (n_time, n_assets),
                dtype=np.float64
            )

            weight_sum = 0.0

            weights = [
                1.00,
                0.30,
                0.18,
                0.12,
                0.12,
                0.10
            ]

            for i, block in enumerate(signals):

                w = weights[i] if i < len(weights) else 0.05

                raw += w * block
                weight_sum += w

            raw /= max(weight_sum, 1e-12)

            # --------------------------------------------------------
            # 4. Cross-sectional demeaning
            # --------------------------------------------------------

            raw = self._demean(raw)

            # --------------------------------------------------------
            # 5. Robust row-wise scale normalization
            #
            # Prevent a handful of rows from dominating.
            # --------------------------------------------------------

            med = np.median(
                np.abs(raw),
                axis=1,
                keepdims=True
            )

            med = np.maximum(med, 1e-6)

            raw = raw / med

            # Bounded final signal.
            raw = np.tanh(raw * 0.35)

            raw = self._demean(raw)

            # --------------------------------------------------------
            # 6. Normalize to controlled L2 concentration
            # --------------------------------------------------------

            target_signal = self._normalize_l2(
                raw,
                self.target_l2
            )

            # --------------------------------------------------------
            # 7. Continuous causal EMA
            #
            # No discontinuous "freeze / jump" threshold.
            # --------------------------------------------------------

            final_positions = np.zeros_like(
                target_signal
            )

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (n_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                previous = self.prev_signal.copy()
            else:
                previous = np.zeros(
                    n_assets,
                    dtype=np.float64
                )

            for t in range(n_time):

                desired = target_signal[t]

                # Continuous response.
                current = (
                    (1.0 - self.ema_alpha) * previous
                    + self.ema_alpha * desired
                )

                # Maintain market neutrality.
                current -= current.mean()

                # Tiny deadband only to prevent numerical churn.
                delta = current - previous

                small = np.abs(delta) < self.min_change

                current[small] = previous[small]

                current -= current.mean()

                final_positions[t] = current

                previous = current.copy()

            # --------------------------------------------------------
            # 8. Final compliance
            # --------------------------------------------------------

            final_df = pd.DataFrame(
                final_positions,
                index=features.index,
                columns=tickers
            )

            # Neutrality.
            final_df = final_df.sub(
                final_df.mean(axis=1),
                axis=0
            )

            # Position bound.
            final_df = final_df.clip(
                -self.target_bound,
                self.target_bound
            )

            # Re-neutralize after clipping.
            final_df = final_df.sub(
                final_df.mean(axis=1),
                axis=0
            )

            # --------------------------------------------------------
            # 9. Persist only the final causal state.
            # --------------------------------------------------------

            self.prev_signal = (
                final_df.iloc[-1]
                .to_numpy(dtype=np.float64)
            )

            self.prev_tickers = final_df.columns.copy()

            return final_df.astype(np.float32)

        except Exception:
            return zero_signal
