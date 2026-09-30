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

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.weights = None
        self.prev_signal = None
        self.prev_tickers = None

        self.bound = 0.20
        self.portfolio_norm = 0.30
        self.alpha = 0.60
        self.hysteresis = 0.10

    @staticmethod
    def robust_normalize(x):
        med = np.nanmedian(x, axis=1, keepdims=True)
        mad = np.nanmedian(
            np.abs(x - med),
            axis=1,
            keepdims=True
        )

        scale = 1.4826 * mad + 1e-8

        z = (x - med) / scale

        return np.tanh(z)

    @staticmethod
    def corr(x, y):
        mask = np.isfinite(x) & np.isfinite(y)

        if mask.sum() < 4:
            return 0.0

        x = x[mask]
        y = y[mask]

        x -= x.mean()
        y -= y.mean()

        d = np.sqrt(
            np.sum(x * x) *
            np.sum(y * y)
        )

        if d < 1e-12:
            return 0.0

        return float(np.sum(x * y) / d)

    def train(self, features, target):

        if features is None or len(features) == 0:
            return

        self.feature_names = list(
            features.columns
            .get_level_values(0)
            .unique()
        )

        if len(self.feature_names) < 1:
            return

        target = np.asarray(
            target,
            dtype=np.float64
        )

        if target.ndim == 1:
            target = target[:, None]

        scores = []

        for name in self.feature_names:

            x = features[name].to_numpy(
                dtype=np.float64
            )

            x = self.robust_normalize(x)

            n = min(
                len(x),
                len(target)
            )

            total = 0.0
            count = 0

            for t in range(n):

                y = target[t]

                if y.size == 1:
                    y = np.repeat(
                        y,
                        x.shape[1]
                    )

                width = min(
                    x.shape[1],
                    y.size
                )

                total += self.corr(
                    x[t, :width],
                    y[:width]
                )

                count += 1

            scores.append(
                total / max(count, 1)
            )

        scores = np.asarray(scores)

        scores[
            np.abs(scores) < 0.003
        ] = 0.0

        norm = np.sum(
            np.abs(scores)
        )

        if norm < 1e-12:
            self.weights = np.ones(
                len(scores)
            ) / len(scores)
        else:
            self.weights = (
                scores / norm
            )

    def predict(self, features):

        tickers = features.columns.get_level_values(1).unique()

        zero = pd.DataFrame(
            0.0,
            index=features.index,
            columns=tickers
        )

        if not self.feature_names:
            return zero.astype(np.float32)

        try:

            signal = None

            for i, name in enumerate(
                self.feature_names
            ):

                x = features[name].to_numpy(
                    dtype=np.float64
                )

                x = self.robust_normalize(x)

                w = (
                    self.weights[i]
                    if self.weights is not None
                    else 1.0 / len(self.feature_names)
                )

                if signal is None:
                    signal = w * x
                else:
                    signal += w * x

            signal -= signal.mean(
                axis=1,
                keepdims=True
            )

            norm = np.linalg.norm(
                signal,
                axis=1,
                keepdims=True
            )

            norm[norm < 1e-9] = 1.0

            target = (
                signal / norm
            ) * self.portfolio_norm

            output = np.zeros_like(target)

            valid_state = (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape
                == (target.shape[1],)
                and self.prev_tickers.equals(tickers)
            )

            active = (
                self.prev_signal.copy()
                if valid_state
                else np.zeros(target.shape[1])
            )

            for t in range(len(target)):

                d = np.linalg.norm(
                    target[t] - active
                )

                if d < self.hysteresis:
                    current = active.copy()
                else:
                    current = (
                        (1 - self.alpha) * active
                        + self.alpha * target[t]
                    )

                current -= current.mean()

                output[t] = current
                active = current.copy()

            output = np.clip(
                output,
                -self.bound,
                self.bound
            )

            output -= output.mean(
                axis=1,
                keepdims=True
            )

            result = pd.DataFrame(
                output,
                index=features.index,
                columns=tickers
            )

            if len(result):
                self.prev_signal = result.iloc[-1].to_numpy()
                self.prev_tickers = result.columns.copy()

            return result.astype(np.float32)

        except Exception:
            return zero.astype(np.float32)
