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
    CITY_NEW01

    A structurally different signal family:
    - cross-sectional rank features
    - feature residualization
    - nonlinear rank contrasts
    - temporal rank changes
    - spherical normalization
    - moderate causal smoothing

    This intentionally avoids the previous:
    tanh + sin/cos pairwise + r*abs(r) construction.
    """

    def __init__(self):
        super().__init__()

        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 1.35
        self.optimal_concentration = 0.28
        self.l1_hysteresis_threshold = 0.78
        self.update_fraction = 0.32

    def train(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame = None,
    ) -> None:
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
            available = set(
                features.columns.get_level_values(0)
            )

            rank_blocks = []

            for name in self.feature_names:
                if name not in available:
                    continue

                values = features[name].astype(np.float64)

                ranked = (
                    values.rank(
                        axis=1,
                        pct=True,
                        method="average",
                    )
                    - 0.5
                ) * 2.0

                rank_blocks.append(
                    ranked.fillna(0.0).to_numpy()
                )

            if len(rank_blocks) < 2:
                return zero_signal

            rank_cube = np.stack(
                rank_blocks,
                axis=0,
            )

            n_features, n_time, n_assets = (
                rank_cube.shape
            )

            # Feature-wise cross-sectional residualization.
            # This removes the common component shared by all
            # anonymized features and emphasizes disagreement.
            feature_mean = rank_cube.mean(
                axis=0,
                keepdims=True,
            )

            residuals = rank_cube - feature_mean

            # Nonlinear contrast:
            # positive disagreement is emphasized differently
            # from negative disagreement.
            contrast = (
                residuals
                * np.abs(residuals)
            )

            contrast_signal = contrast.mean(
                axis=0
            )

            # Add a second-order feature-agreement component.
            # This is not the old pairwise sin/cos family.
            squared_centered = (
                rank_cube ** 2
                - (rank_cube ** 2).mean(
                    axis=0,
                    keepdims=True,
                )
            )

            curvature_signal = squared_centered.mean(
                axis=0
            )

            raw = (
                0.72 * contrast_signal
                + 0.28 * curvature_signal
            )

            # Temporal change in the ranked cross-sectional signal.
            # The first row has no prior observation.
            temporal = np.zeros_like(raw)

            if n_time > 1:
                temporal[1:] = (
                    rank_cube.mean(axis=0)[1:]
                    - rank_cube.mean(axis=0)[:-1]
                )

            raw = raw + 0.22 * temporal

            # Cross-sectional dollar neutrality.
            raw -= raw.mean(
                axis=1,
                keepdims=True,
            )

            norms = np.linalg.norm(
                raw,
                axis=1,
                keepdims=True,
            )

            target_positions = (
                raw / np.maximum(norms, 1e-8)
            ) * self.optimal_concentration

            output = np.zeros_like(
                target_positions
            )

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
                desired = target_positions[t]

                distance = np.sum(
                    np.abs(desired - active)
                )

                if distance <= self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    current = (
                        (1.0 - self.update_fraction)
                        * active
                        + self.update_fraction
                        * desired
                    )

                    current -= current.mean()

                output[t] = current
                active = current.copy()

            result = pd.DataFrame(
                output,
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
