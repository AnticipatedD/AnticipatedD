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
    ENGINEERED02

    Fixed, target-independent cross-sectional ensemble.

    Design goals:
    - no target-based feature selection
    - no pairwise feature explosion
    - no train/validation contamination
    - no spherical normalization
    - no stateful hysteresis
    - robust row-wise demeaning
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None

        self.output_scale = 0.20
        self.target_bound = 1.35

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame = None,
    ) -> None:
        self.feature_names = None

        if features is None or features.empty:
            return

        try:
            self.feature_names = list(
                features.columns
                .get_level_values(0)
                .unique()
            )
        except Exception:
            self.feature_names = None

    def _rank(self, block):
        values = block.astype(np.float64)

        ranked = (
            values.rank(
                axis=1,
                pct=True,
                method="average",
            )
            - 0.5
        ) * 2.0

        return ranked.fillna(0.0).to_numpy()

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        try:
            tickers = (
                features.columns
                .get_level_values(1)
                .unique()
            )
        except Exception:
            tickers = pd.Index([])

        zero_signal = pd.DataFrame(
            0.0,
            index=getattr(features, "index", []),
            columns=tickers,
            dtype=np.float32,
        )

        if (
            features is None
            or features.empty
            or not self.feature_names
        ):
            return zero_signal

        try:
            available = set(
                features.columns.get_level_values(0)
            )

            blocks = []

            for name in self.feature_names:
                if name not in available:
                    continue

                ranked = self._rank(
                    features[name]
                )

                ranked -= ranked.mean(
                    axis=1,
                    keepdims=True,
                )

                blocks.append(ranked)

            if len(blocks) == 0:
                return zero_signal

            cube = np.stack(
                blocks,
                axis=0,
            )

            # Equal-weight robust consensus.
            # No data-dependent feature selection.
            signal = cube.mean(axis=0)

            # Add a bounded nonlinear consensus term.
            # The fixed coefficient avoids fitting noise.
            nonlinear = np.tanh(
                1.25 * signal
            )

            signal = (
                0.75 * signal
                + 0.25 * nonlinear
            )

            signal -= signal.mean(
                axis=1,
                keepdims=True,
            )

            # Robust row scaling without forcing identical
            # spherical exposure on every row.
            scale = np.nanpercentile(
                np.abs(signal),
                75,
                axis=1,
                keepdims=True,
            )

            signal = signal / np.maximum(
                scale,
                1e-8,
            )

            signal = np.tanh(
                signal
            ) * self.output_scale

            signal -= signal.mean(
                axis=1,
                keepdims=True,
            )

            result = pd.DataFrame(
                signal,
                index=features.index,
                columns=tickers,
            )

            result = result.clip(
                -self.target_bound,
                self.target_bound,
            )

            result = result.sub(
                result.mean(axis=1),
                axis=0,
            )

            return result.astype(np.float32)

        except Exception:
            return zero_signal


predictor = MyPredictor()
