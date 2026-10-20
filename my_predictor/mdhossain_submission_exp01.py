# /// script
# dependencies = [
#   "numpy",
#   "pandas"
# ]
# ///

import numpy as np
import pandas as pd

from predictor import Predictor


class MyPredictor(Predictor):
    """
    Experiment 01:
    Lower concentration, lower hysteresis, faster adaptation.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 0.12
        self.optimal_concentration = 0.18
        self.l1_hysteresis_threshold = 0.55
        self.smoothing = 0.65

    def train(self, features, target=None):
        if features is None or features.empty:
            self.feature_names = None
            return

        try:
            self.feature_names = list(
                features.columns
                .get_level_values(0)
                .unique()
            )
        except Exception:
            self.feature_names = None

    def predict(self, features):
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
            or self.feature_names is None
            or len(self.feature_names) < 2
        ):
            return zero_signal

        try:
            ranks = {}

            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue

                values = features[feat].astype(np.float64)

                ranks[feat] = (
                    (
                        values.rank(
                            axis=1,
                            pct=True,
                            method="average",
                        )
                        - 0.5
                    )
                    * 2.0
                ).fillna(0.0).to_numpy()

            if len(ranks) < 2:
                return zero_signal

            n_time, n_assets = next(
                iter(ranks.values())
            ).shape

            blocks = []
            names = list(ranks.keys())

            for i, f1 in enumerate(names):
                r1 = ranks[f1]

                blocks.append(np.tanh(1.5 * r1))

                for j in range(i + 1, len(names)):
                    r2 = ranks[names[j]]

                    blocks.append(
                        np.sin(r1 * np.pi * 0.25)
                        * np.cos(r2 * np.pi * 0.25)
                    )

                    blocks.append(
                        r1 * np.abs(r2)
                    )

            raw = np.mean(blocks, axis=0)

            raw -= raw.mean(axis=1, keepdims=True)

            norms = np.linalg.norm(
                raw,
                axis=1,
                keepdims=True,
            )

            target_positions = (
                raw / np.maximum(norms, 1e-8)
            ) * self.optimal_concentration

            output = np.zeros_like(target_positions)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and len(self.prev_signal) == n_assets
                and self.prev_tickers.equals(tickers)
            ):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(
                    n_assets,
                    dtype=np.float64,
                )

            for t in range(n_time):
                desired = target_positions[t]
                distance = np.sum(np.abs(desired - active))

                if distance <= self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    current = (
                        self.smoothing * active
                        + (1.0 - self.smoothing) * desired
                    )
                    current -= current.mean()

                output[t] = current
                active = current.copy()

            result = pd.DataFrame(
                output,
                index=features.index,
                columns=tickers,
            )

            result -= result.mean(axis=1).to_numpy()[:, None]
            result = result.clip(
                -self.target_bound,
                self.target_bound,
            )
            result -= result.mean(axis=1).to_numpy()[:, None]

            self.prev_signal = (
                result.iloc[-1].to_numpy(dtype=np.float64)
            )
            self.prev_tickers = result.columns.copy()

            return result.astype(np.float32)

        except Exception:
            return zero_signal


predictor = MyPredictor()
