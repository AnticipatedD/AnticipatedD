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
    Dependency-free regularized cross-sectional ridge-style predictor.

    Input:
        DataFrame with columns (feature, ticker)
        and rows representing timestamps.

    Output:
        DataFrame with rows representing timestamps
        and columns representing tickers.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.feature_weights = None
        self.prev_signal = None
        self.prev_tickers = None

        self.ridge = 2.0
        self.output_scale = 0.20
        self.target_bound = 1.35

        self.turnover_threshold = 0.60
        self.update_fraction = 0.30

    def _feature_names(self, features):
        if isinstance(features.columns, pd.MultiIndex):
            return list(
                features.columns
                .get_level_values(0)
                .unique()
            )

        return list(features.columns)

    def _feature_block(self, features, name):
        if isinstance(features.columns, pd.MultiIndex):
            block = features[name]
        else:
            block = features[[name]]

        if isinstance(block, pd.Series):
            block = block.to_frame()

        return block.astype(np.float64)

    def _rank_block(self, block):
        ranked = (
            block.rank(
                axis=1,
                pct=True,
                method="average",
            )
            - 0.5
        ) * 2.0

        ranked = ranked.fillna(0.0)

        return ranked.to_numpy(
            dtype=np.float64
        )

    def _target_matrix(self, target, features):
        if target is None:
            return None

        if isinstance(target, pd.DataFrame):
            target_df = target.copy()

            if (
                isinstance(target_df.columns, pd.MultiIndex)
                and isinstance(features.columns, pd.MultiIndex)
            ):
                feature_tickers = (
                    features.columns
                    .get_level_values(1)
                    .unique()
                )

                target_tickers = (
                    target_df.columns
                    .get_level_values(-1)
                    .unique()
                )

                if set(feature_tickers).issubset(
                    set(target_tickers)
                ):
                    target_df = target_df.reindex(
                        index=features.index,
                        columns=feature_tickers,
                    )
                else:
                    target_df = target_df.reindex(
                        index=features.index
                    )

            else:
                target_df = target_df.reindex(
                    index=features.index
                )

            if isinstance(
                target_df.columns,
                pd.MultiIndex,
            ):
                target_df = target_df.groupby(
                    level=-1,
                    axis=1,
                ).mean()

            target_df = target_df.reindex(
                index=features.index
            )

            if isinstance(features.columns, pd.MultiIndex):
                tickers = (
                    features.columns
                    .get_level_values(1)
                    .unique()
                )

                target_df = target_df.reindex(
                    columns=tickers
                )

            return target_df.astype(
                np.float64
            ).fillna(0.0).to_numpy()

        if isinstance(target, pd.Series):
            values = target.reindex(
                features.index
            ).astype(np.float64)

            values = values.fillna(0.0).to_numpy()

            if isinstance(features.columns, pd.MultiIndex):
                n_assets = len(
                    features.columns
                    .get_level_values(1)
                    .unique()
                )
            else:
                n_assets = len(features.columns)

            return np.repeat(
                values[:, None],
                n_assets,
                axis=1,
            )

        array = np.asarray(
            target,
            dtype=np.float64,
        )

        if array.ndim == 1:
            if len(array) != len(features):
                raise ValueError(
                    "One-dimensional target length does not "
                    "match the number of feature rows."
                )

            if isinstance(features.columns, pd.MultiIndex):
                n_assets = len(
                    features.columns
                    .get_level_values(1)
                    .unique()
                )
            else:
                n_assets = len(features.columns)

            return np.repeat(
                array[:, None],
                n_assets,
                axis=1,
            )

        if array.ndim == 2:
            return array

        raise ValueError(
            "Unsupported target shape."
        )

    def _fit_weights(self, features, target):
        names = self._feature_names(features)
        target_matrix = self._target_matrix(
            target,
            features,
        )

        if target_matrix is None:
            return names, None

        weights = {}
        scores = []

        for name in names:
            block = self._feature_block(
                features,
                name,
            )

            x = self._rank_block(block)

            if x.shape != target_matrix.shape:
                raise ValueError(
                    "Feature and target shapes do not match."
                )

            y = np.nan_to_num(
                target_matrix,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            x = np.nan_to_num(
                x,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            x = x - x.mean(
                axis=1,
                keepdims=True,
            )

            y = y - y.mean(
                axis=1,
                keepdims=True,
            )

            x_flat = x.reshape(-1)
            y_flat = y.reshape(-1)

            x_scale = np.std(x_flat)
            y_scale = np.std(y_flat)

            if (
                x_scale < 1e-10
                or y_scale < 1e-10
            ):
                continue

            x_flat = x_flat / x_scale
            y_flat = y_flat / y_scale

            score = np.mean(
                x_flat * y_flat
            )

            if np.isfinite(score):
                scores.append(
                    (name, float(score))
                )

        if not scores:
            return names, None

        # Strong shrinkage; no top-feature selection.
        # This avoids choosing random winners among many features.
        for name, score in scores:
            weights[name] = score / (
                self.ridge + abs(score)
            )

        normalizer = sum(
            abs(value)
            for value in weights.values()
        )

        if normalizer < 1e-12:
            return names, None

        weights = {
            name: value / normalizer
            for name, value in weights.items()
        }

        return names, weights

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame = None,
    ) -> None:
        if features is None or features.empty:
            self.feature_names = None
            self.feature_weights = None
            return

        try:
            (
                self.feature_names,
                self.feature_weights,
            ) = self._fit_weights(
                features,
                target,
            )
        except Exception:
            self.feature_names = None
            self.feature_weights = None

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        if features is None or features.empty:
            return pd.DataFrame(
                index=getattr(features, "index", [])
            )

        try:
            if isinstance(
                features.columns,
                pd.MultiIndex,
            ):
                tickers = (
                    features.columns
                    .get_level_values(1)
                    .unique()
                )
            else:
                tickers = features.columns

            zero_signal = pd.DataFrame(
                0.0,
                index=features.index,
                columns=tickers,
                dtype=np.float32,
            )

            if (
                self.feature_names is None
                or self.feature_weights is None
            ):
                return zero_signal

            signal = np.zeros(
                (len(features), len(tickers)),
                dtype=np.float64,
            )

            used_weight = 0.0

            for name in self.feature_names:
                if name not in self.feature_weights:
                    continue

                block = self._feature_block(
                    features,
                    name,
                )

                ranked = self._rank_block(
                    block
                )

                ranked -= ranked.mean(
                    axis=1,
                    keepdims=True,
                )

                weight = self.feature_weights[name]

                signal += weight * ranked
                used_weight += abs(weight)

            if used_weight < 1e-12:
                return zero_signal

            signal -= signal.mean(
                axis=1,
                keepdims=True,
            )

            # Preserve direction while limiting extremes.
            row_scale = np.nanpercentile(
                np.abs(signal),
                75,
                axis=1,
                keepdims=True,
            )

            signal = signal / np.maximum(
                row_scale,
                1e-8,
            )

            desired = (
                np.tanh(signal)
                * self.output_scale
            )

            result_array = np.zeros_like(
                desired
            )

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and len(self.prev_signal)
                == len(tickers)
                and self.prev_tickers.equals(
                    tickers
                )
            ):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(
                    len(tickers),
                    dtype=np.float64,
                )

            for t in range(len(features)):
                desired_t = desired[t]

                movement = np.sum(
                    np.abs(
                        desired_t - active
                    )
                )

                if movement <= self.turnover_threshold:
                    current = active.copy()
                else:
                    current = (
                        (1.0 - self.update_fraction)
                        * active
                        + self.update_fraction
                        * desired_t
                    )

                current -= current.mean()
                result_array[t] = current
                active = current.copy()

            result = pd.DataFrame(
                result_array,
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

            self.prev_tickers = tickers.copy()

            return result.astype(
                np.float32
            )

        except Exception:
            try:
                return pd.DataFrame(
                    0.0,
                    index=features.index,
                    columns=tickers,
                    dtype=np.float32,
                )
            except Exception:
                return pd.DataFrame()


predictor = MyPredictor()
