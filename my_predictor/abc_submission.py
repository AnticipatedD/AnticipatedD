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
    Target-aligned cross-sectional ridge-style rank ensemble.

    Uses the training target to estimate feature directions and weights.
    The final prediction remains cross-sectionally demeaned and bounded.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.feature_weights = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 1.35
        self.output_scale = 0.22
        self.turnover_threshold = 0.70
        self.update_fraction = 0.35
        self.ridge = 0.08
        self.minimum_weight = 0.05

    def _rank_matrix(self, block):
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

    def _target_array(self, target, index):
        if target is None:
            return None

        if isinstance(target, pd.DataFrame):
            if target.shape[1] == 1:
                result = target.iloc[:, 0]
            else:
                result = target.mean(axis=1)
        elif isinstance(target, pd.Series):
            result = target
        else:
            result = pd.Series(
                np.asarray(target).reshape(-1),
                index=index,
            )

        result = result.reindex(index)
        return result.astype(np.float64).to_numpy()

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame = None,
    ) -> None:
        self.feature_names = None
        self.feature_weights = None

        if features is None or features.empty:
            return

        try:
            names = list(
                features.columns
                .get_level_values(0)
                .unique()
            )

            y = self._target_array(
                target,
                features.index,
            )

            if y is None or len(y) != len(features):
                self.feature_names = names
                return

            y = np.nan_to_num(
                y,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            y = y - np.mean(y)
            y_scale = np.std(y)

            if y_scale < 1e-12:
                self.feature_names = names
                return

            y = y / y_scale

            weights = {}
            usable_names = []

            available = set(
                features.columns.get_level_values(0)
            )

            for name in names:
                if name not in available:
                    continue

                x = self._rank_matrix(
                    features[name]
                )

                x = np.nan_to_num(
                    x,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                x_mean = x.mean(axis=1)
                x_mean = x_mean - x_mean.mean()

                x_scale = np.std(x_mean)

                if x_scale < 1e-10:
                    continue

                x_mean = x_mean / x_scale

                covariance = np.mean(
                    x_mean * y
                )

                weight = covariance / (
                    1.0 + self.ridge
                )

                if abs(weight) >= self.minimum_weight:
                    weights[name] = weight
                    usable_names.append(name)

            self.feature_names = usable_names

            if not weights:
                self.feature_weights = None
                return

            total = sum(
                abs(value)
                for value in weights.values()
            )

            if total > 1e-12:
                weights = {
                    key: value / total
                    for key, value in weights.items()
                }

            self.feature_weights = weights

        except Exception:
            self.feature_names = None
            self.feature_weights = None

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
            or not self.feature_weights
        ):
            return zero_signal

        try:
            available = set(
                features.columns.get_level_values(0)
            )

            raw = np.zeros(
                (len(features), len(tickers)),
                dtype=np.float64,
            )

            total_abs_weight = 0.0

            for name in self.feature_names:
                if name not in available:
                    continue

                weight = self.feature_weights.get(
                    name,
                    0.0,
                )

                if abs(weight) < 1e-12:
                    continue

                block = self._rank_matrix(
                    features[name]
                )

                raw += weight * block
                total_abs_weight += abs(weight)

            if total_abs_weight < 1e-12:
                return zero_signal

            raw -= raw.mean(
                axis=1,
                keepdims=True,
            )

            # Robust row scaling without forcing every row
            # to have identical spherical magnitude.
            row_scale = np.nanpercentile(
                np.abs(raw),
                75,
                axis=1,
                keepdims=True,
            )

            raw = raw / np.maximum(
                row_scale,
                1e-8,
            )

            raw = np.tanh(raw)
            raw -= raw.mean(
                axis=1,
                keepdims=True,
            )

            desired = raw * self.output_scale

            n_time, n_assets = desired.shape
            output = np.zeros_like(desired)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (n_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(
                    n_assets,
                    dtype=np.float64,
                )

            for t in range(n_time):
                target_position = desired[t]

                movement = np.sum(
                    np.abs(
                        target_position
                        - active
                    )
                )

                if movement <= self.turnover_threshold:
                    current = active.copy()
                else:
                    current = (
                        (1.0 - self.update_fraction)
                        * active
                        + self.update_fraction
                        * target_position
                    )

                current -= current.mean()
                output[t] = current
                active = current.copy()

            result = pd.DataFrame(
                output,
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

            self.prev_signal = (
                result.iloc[-1]
                .to_numpy(dtype=np.float64)
            )

            self.prev_tickers = result.columns.copy()

            return result.astype(np.float32)

        except Exception:
            return zero_signal


predictor = MyPredictor()
