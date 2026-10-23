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
    REGULARIZED01

    Conservative target-aligned rank ensemble intended to reduce
    false discoveries and improve robustness in AlphaNova's
    overfitting test.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.feature_weights = None

        self.prev_signal = None
        self.prev_tickers = None

        self.output_scale = 0.18
        self.target_bound = 1.35

        self.ridge = 1.50
        self.keep_fraction = 0.25
        self.min_abs_score = 0.04

        self.turnover_threshold = 0.55
        self.update_fraction = 0.25

    def _rank(self, values):
        ranked = (
            values.astype(np.float64)
            .rank(
                axis=1,
                pct=True,
                method="average",
            )
            - 0.5
        ) * 2.0

        return ranked.fillna(0.0).to_numpy()

    def _target_vector(self, target, index):
        if target is None:
            return None

        if isinstance(target, pd.DataFrame):
            if target.shape[1] == 1:
                y = target.iloc[:, 0]
            else:
                y = target.mean(axis=1)
        elif isinstance(target, pd.Series):
            y = target
        else:
            y = pd.Series(
                np.asarray(target).reshape(-1),
                index=index,
            )

        y = y.reindex(index)
        y = y.astype(np.float64).to_numpy()

        y = np.nan_to_num(
            y,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        y -= y.mean()

        scale = np.std(y)

        if scale < 1e-10:
            return None

        return y / scale

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
            y = self._target_vector(
                target,
                features.index,
            )

            if y is None:
                return

            available = set(
                features.columns.get_level_values(0)
            )

            candidates = []

            for name in features.columns.get_level_values(0).unique():
                if name not in available:
                    continue

                x = self._rank(features[name])

                # One robust cross-sectional summary per row.
                x = x.mean(axis=1)
                x -= x.mean()

                scale = np.std(x)

                if scale < 1e-10:
                    continue

                x /= scale

                score = np.mean(x * y)

                if not np.isfinite(score):
                    continue

                candidates.append(
                    (name, float(score))
                )

            if not candidates:
                return

            # Strong shrinkage and conservative selection.
            scores = np.array(
                [score for _, score in candidates],
                dtype=np.float64,
            )

            abs_scores = np.abs(scores)

            cutoff = np.quantile(
                abs_scores,
                1.0 - self.keep_fraction,
            )

            selected = []

            for name, score in candidates:
                if (
                    abs(score) >= cutoff
                    and abs(score) >= self.min_abs_score
                ):
                    shrunk = score / (
                        self.ridge + abs(score)
                    )
                    selected.append(
                        (name, shrunk)
                    )

            if not selected:
                return

            total = sum(
                abs(weight)
                for _, weight in selected
            )

            if total <= 1e-12:
                return

            self.feature_names = [
                name for name, _ in selected
            ]

            self.feature_weights = {
                name: weight / total
                for name, weight in selected
            }

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

            n_time = len(features)
            n_assets = len(tickers)

            raw = np.zeros(
                (n_time, n_assets),
                dtype=np.float64,
            )

            total = 0.0

            for name in self.feature_names:
                if name not in available:
                    continue

                weight = self.feature_weights.get(
                    name,
                    0.0,
                )

                if abs(weight) < 1e-12:
                    continue

                ranked = self._rank(
                    features[name]
                )

                # Preserve the full cross-sectional ranking,
                # rather than collapsing assets to one row statistic.
                ranked -= ranked.mean(
                    axis=1,
                    keepdims=True,
                )

                raw += weight * ranked
                total += abs(weight)

            if total <= 1e-12:
                return zero_signal

            raw -= raw.mean(
                axis=1,
                keepdims=True,
            )

            # Mild robust compression; no forced sphere normalization.
            scale = np.nanpercentile(
                np.abs(raw),
                75,
                axis=1,
                keepdims=True,
            )

            raw = raw / np.maximum(
                scale,
                1e-8,
            )

            desired = (
                np.tanh(raw)
                * self.output_scale
            )

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
                        target_position - active
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
