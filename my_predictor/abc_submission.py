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
    REGIME01

    Conservative cross-sectional rank signal.

    Design:
    - Uses the native MultiIndex feature layout.
    - Uses Features 1-4 only.
    - Excludes Features 5-6, which were weakest in recent windows.
    - Uses fixed weights rather than noisy target fitting.
    - Detects a broad recent regime from feature cross-sectional
      dispersion and changes sign only when the reversal is strong.
    - Returns a wide DataFrame: timestamps x tickers.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = [
            "Feature.1",
            "Feature.2",
            "Feature.3",
            "Feature.4",
        ]

        self.weights = {
            "Feature.1": 0.30,
            "Feature.2": 0.30,
            "Feature.3": 0.25,
            "Feature.4": 0.15,
        }

        self.output_scale = 0.32
        self.position_cap = 1.35

        self.recent_window = 2500
        self.regime_window = 500
        self.regime_threshold = 0.18

        self.use_state = False
        self.previous_signal = None
        self.previous_tickers = None

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame = None,
    ) -> None:
        # No target fitting.
        # This prevents in-sample feature selection and
        # avoids the previous Ridge alignment problem.
        return None

    def _get_tickers(self, features):
        if isinstance(
            features.columns,
            pd.MultiIndex,
        ):
            return (
                features.columns
                .get_level_values(1)
                .unique()
            )

        return features.columns

    def _get_block(self, features, feature_name):
        if isinstance(
            features.columns,
            pd.MultiIndex,
        ):
            block = features[feature_name]
        else:
            block = features[[feature_name]]

        if isinstance(block, pd.Series):
            block = block.to_frame()

        tickers = self._get_tickers(
            features
        )

        return block.reindex(
            columns=tickers
        ).astype(np.float64)

    def _cross_sectional_rank(self, block):
        ranked = (
            block.rank(
                axis=1,
                pct=True,
                method="average",
            )
            - 0.5
        ) * 2.0

        ranked = ranked.fillna(0.0)

        values = ranked.to_numpy(
            dtype=np.float64
        )

        values -= values.mean(
            axis=1,
            keepdims=True,
        )

        return values

    def _recent_regime(self, features):
        """
        Uses only feature behavior, not target values.

        The regime statistic measures whether recent values are
        moving persistently in one broad direction. It is used
        conservatively: a reversal is activated only when the
        recent mean is materially different from the older mean.
        """
        values = []

        for name in self.feature_names:
            block = self._get_block(
                features,
                name,
            )

            ranked = self._cross_sectional_rank(
                block
            )

            row_average = ranked.mean(
                axis=1
            )

            values.append(row_average)

        if not values:
            return 1.0

        matrix = np.vstack(values)

        if matrix.shape[1] < (
            self.recent_window
            + self.regime_window
        ):
            return 1.0

        recent = matrix[
            :,
            -self.regime_window:,
        ].mean()

        prior = matrix[
            :,
            -(
                self.regime_window
                + self.recent_window
            ):-
            self.regime_window,
        ].mean()

        change = recent - prior

        if change < -self.regime_threshold:
            return -1.0

        return 1.0

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        if features is None or features.empty:
            return pd.DataFrame()

        tickers = self._get_tickers(
            features
        )

        zero = pd.DataFrame(
            0.0,
            index=features.index,
            columns=tickers,
            dtype=np.float32,
        )

        try:
            signal = np.zeros(
                (
                    len(features),
                    len(tickers),
                ),
                dtype=np.float64,
            )

            used_weight = 0.0

            for name in self.feature_names:
                if isinstance(
                    features.columns,
                    pd.MultiIndex,
                ):
                    available = set(
                        features.columns
                        .get_level_values(0)
                    )
                else:
                    available = set(
                        features.columns
                    )

                if name not in available:
                    continue

                block = self._get_block(
                    features,
                    name,
                )

                ranked = self._cross_sectional_rank(
                    block
                )

                weight = self.weights[name]

                signal += weight * ranked
                used_weight += abs(weight)

            if used_weight <= 1e-12:
                return zero

            signal /= used_weight

            regime = self._recent_regime(
                features
            )

            signal *= regime

            signal -= signal.mean(
                axis=1,
                keepdims=True,
            )

            # Robust scale across each timestamp.
            row_scale = np.nanpercentile(
                np.abs(signal),
                75,
                axis=1,
                keepdims=True,
            )

            signal /= np.maximum(
                row_scale,
                1e-8,
            )

            # Smooth bounded exposure.
            signal = np.tanh(
                signal
            ) * self.output_scale

            signal -= signal.mean(
                axis=1,
                keepdims=True,
            )

            if (
                self.use_state
                and self.previous_signal is not None
                and self.previous_tickers is not None
                and self.previous_tickers.equals(
                    tickers
                )
                and len(self.previous_signal)
                == len(tickers)
            ):
                active = (
                    self.previous_signal.copy()
                )

                for t in range(len(signal)):
                    movement = np.sum(
                        np.abs(
                            signal[t] - active
                        )
                    )

                    if movement < 0.55:
                        signal[t] = active
                    else:
                        signal[t] = (
                            0.70 * active
                            + 0.30 * signal[t]
                        )

                    signal[t] -= signal[t].mean()
                    active = signal[t].copy()

            result = pd.DataFrame(
                signal,
                index=features.index,
                columns=tickers,
            )

            result = result.clip(
                -self.position_cap,
                self.position_cap,
            )

            result = result.sub(
                result.mean(axis=1),
                axis=0,
            )

            self.previous_signal = (
                result.iloc[-1]
                .to_numpy(dtype=np.float64)
            )

            self.previous_tickers = tickers.copy()

            return result.astype(
                np.float32
            )

        except Exception:
            return zero


predictor = MyPredictor()
