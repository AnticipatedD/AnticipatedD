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
    EXP04:
    Same signal construction as the stronger baseline, with reduced
    hysteresis and faster execution adjustment to reduce compression.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 1.35
        self.optimal_concentration = 0.28
        self.l1_hysteresis_threshold = 0.85
        self.update_fraction = 0.35

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame = None,
    ) -> None:
        if features is not None and not features.empty:
            try:
                self.feature_names = list(
                    features.columns
                    .get_level_values(0)
                    .unique()
                )
            except Exception:
                self.feature_names = None

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
            or self.feature_names is None
            or len(self.feature_names) < 2
        ):
            return zero_signal

        try:
            ranks = {}

            available_features = set(
                features.columns.get_level_values(0)
            )

            for feat in self.feature_names:
                if feat not in available_features:
                    continue

                values = features[feat].astype(np.float64)

                ranks[feat] = (
                    (
                        values.rank(
                            axis=1,
                            pct=True,
                            method="max",
                        )
                        - 0.5
                    )
                    * 2.0
                ).fillna(0.0).to_numpy()

            if len(ranks) < 2:
                return zero_signal

            feature_list = list(ranks.keys())
            interaction_blocks = []

            for i, f1 in enumerate(feature_list):
                r1 = ranks[f1]

                interaction_blocks.append(
                    np.tanh(r1 * 2.0)
                )

                for j in range(i + 1, len(feature_list)):
                    f2 = feature_list[j]
                    r2 = ranks[f2]

                    interaction_blocks.append(
                        np.sin(r1 * np.pi * 0.25)
                        * np.cos(r2 * np.pi * 0.25)
                    )

                    interaction_blocks.append(
                        r1 * np.abs(r2)
                    )

            raw_velocity = np.mean(
                interaction_blocks,
                axis=0,
            )

            raw_velocity -= raw_velocity.mean(
                axis=1,
                keepdims=True,
            )

            norms = np.linalg.norm(
                raw_velocity,
                axis=1,
                keepdims=True,
            )

            sphere_target = (
                raw_velocity
                / np.maximum(norms, 1e-8)
            ) * self.optimal_concentration

            n_time, n_assets = sphere_target.shape
            final_positions = np.zeros_like(
                sphere_target
            )

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (n_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                active_position = self.prev_signal.copy()
            else:
                active_position = np.zeros(
                    n_assets,
                    dtype=np.float64,
                )

            for t in range(n_time):
                target_position = sphere_target[t]

                turnover = np.sum(
                    np.abs(
                        target_position
                        - active_position
                    )
                )

                if turnover < self.l1_hysteresis_threshold:
                    current = active_position.copy()
                else:
                    current = (
                        (1.0 - self.update_fraction)
                        * active_position
                        + self.update_fraction
                        * target_position
                    )

                    current -= current.mean()

                final_positions[t] = current
                active_position = current.copy()

            result = pd.DataFrame(
                final_positions,
                index=features.index,
                columns=tickers,
            )

            result = result.sub(
                result.mean(axis=1),
                axis=0,
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
