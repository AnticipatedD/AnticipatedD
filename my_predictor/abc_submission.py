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
    """Low-complexity, stateless cross-sectional rank signal."""

    def __init__(self):
        super().__init__()

        self.weights = {
            "Feature.1": 0.40,
            "Feature.2": 0.30,
            "Feature.3": 0.20,
            "Feature.4": 0.10,
        }

        self.scale = 0.10
        self.clip_value = 0.30

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame,
    ) -> None:
        # Deliberately fixed and target-free.
        return None

    def _tickers(
        self,
        features: pd.DataFrame,
    ) -> pd.Index:
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

    def _available(
        self,
        features: pd.DataFrame,
    ) -> set:
        if isinstance(
            features.columns,
            pd.MultiIndex,
        ):
            return set(
                features.columns
                .get_level_values(0)
            )

        return set(features.columns)

    def _block(
        self,
        features: pd.DataFrame,
        name: str,
        tickers: pd.Index,
    ) -> pd.DataFrame:
        if isinstance(
            features.columns,
            pd.MultiIndex,
        ):
            block = features[name]
        else:
            block = features[[name]]

        if isinstance(block, pd.Series):
            block = block.to_frame()

        return block.reindex(
            columns=tickers
        ).astype("float64")

    def _rank(
        self,
        block: pd.DataFrame,
    ) -> np.ndarray:
        ranked = block.rank(
            axis=1,
            method="average",
            pct=True,
        )

        ranked = (
            ranked
            .sub(0.5)
            .mul(2.0)
            .fillna(0.0)
        )

        values = ranked.to_numpy(
            dtype="float64"
        )

        values -= values.mean(
            axis=1,
            keepdims=True,
        )

        return values

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        if features is None:
            return pd.DataFrame()

        tickers = self._tickers(features)

        result = np.zeros(
            (
                len(features),
                len(tickers),
            ),
            dtype="float64",
        )

        available = self._available(
            features
        )

        total_weight = 0.0

        for name, weight in (
            self.weights.items()
        ):
            if name not in available:
                continue

            block = self._block(
                features,
                name,
                tickers,
            )

            result += (
                weight
                * self._rank(block)
            )

            total_weight += abs(weight)

        if total_weight == 0.0:
            return pd.DataFrame(
                0.0,
                index=features.index,
                columns=tickers,
                dtype="float32",
            )

        result /= total_weight

        result -= result.mean(
            axis=1,
            keepdims=True,
        )

        # Mild bounded exposure.
        # No interactions, state, adaptive sign,
        # spherical normalization, or target fitting.
        result = np.tanh(result)
        result *= self.scale

        result = np.clip(
            result,
            -self.clip_value,
            self.clip_value,
        )

        result -= result.mean(
            axis=1,
            keepdims=True,
        )

        return pd.DataFrame(
            result.astype("float32"),
            index=features.index,
            columns=tickers,
        )


predictor = MyPredictor()
