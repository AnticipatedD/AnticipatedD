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
    AlphaNova Cycle 3 - Cross-Sectional Ridge + Interaction Predictor

    Design:
      1. Cross-sectional robust z-scores.
      2. Rank features.
      3. First-order + pairwise interaction terms.
      4. Ridge regression with unpenalized intercept.
      5. Cross-sectional de-meaning.
      6. Mild temporal smoothing.
      7. Tanh signal compression.
      8. L1 market-neutral portfolio normalization.

    Dependencies: numpy, pandas only.
    """

    def __init__(
        self,
        ridge=12.0,
        smooth_alpha=0.20,
        clip_value=3.0,
        epsilon=1e-8,
    ):
        self.ridge = float(ridge)
        self.smooth_alpha = float(smooth_alpha)
        self.clip_value = float(clip_value)
        self.epsilon = float(epsilon)

        self.weights = None
        self.feature_names = None
        self.prev_signal = None

    # ------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------

    def _get_feature_names(self, features):
        if isinstance(features.columns, pd.MultiIndex):
            return list(features.columns.get_level_values(0).unique())

        return list(features.columns)

    def _get_feature_frame(self, features, name):
        if isinstance(features.columns, pd.MultiIndex):
            return features[name]

        return features[[name]]

    def _cross_sectional_zscore(self, df):
        mean = df.mean(axis=1)
        std = df.std(axis=1).replace(0.0, self.epsilon)

        z = df.sub(mean, axis=0).div(std, axis=0)

        return z.clip(
            -self.clip_value,
            self.clip_value
        ).fillna(0.0)

    def _cross_sectional_rank(self, df):
        return (
            df.rank(axis=1, pct=True)
            .sub(0.5)
            .fillna(0.0)
        )

    def _extract_features(self, features):
        names = self._get_feature_names(features)

        matrices = []
        used_names = []

        z_features = {}
        r_features = {}

        # --------------------------------------------------------
        # Base feature transformations
        # --------------------------------------------------------

        for name in names:
            df = self._get_feature_frame(features, name)

            # If a MultiIndex feature expands to ticker columns,
            # use it directly. Otherwise retain normal DataFrame.
            z = self._cross_sectional_zscore(df)
            r = self._cross_sectional_rank(df)

            z_features[name] = z
            r_features[name] = r

            matrices.append(z.to_numpy(dtype=np.float64).ravel())
            used_names.append("z_" + str(name))

            matrices.append(r.to_numpy(dtype=np.float64).ravel())
            used_names.append("r_" + str(name))

        # --------------------------------------------------------
        # Pairwise interactions
        # --------------------------------------------------------

        n = len(names)

        for a in range(n):
            for b in range(a + 1, n):
                na = names[a]
                nb = names[b]

                za = z_features[na]
                zb = z_features[nb]

                interaction = (za * zb).clip(
                    -self.clip_value,
                    self.clip_value
                )

                matrices.append(
                    interaction.to_numpy(dtype=np.float64).ravel()
                )

                used_names.append(
                    "int_" + str(na) + "_" + str(nb)
                )

        if not matrices:
            rows = len(features.index)
            return (
                np.zeros((rows, 1), dtype=np.float64),
                names,
            )

        X = np.column_stack(matrices)

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=self.clip_value,
            neginf=-self.clip_value,
        )

        return X, used_names

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    def train(self, features, target=None):
        X, feature_names = self._extract_features(features)

        self.feature_names = feature_names

        if target is None:
            self.weights = np.zeros(X.shape[1] + 1)
            return

        y = target.to_numpy(dtype=np.float64).ravel()

        valid = np.isfinite(y)

        if not np.any(valid):
            self.weights = np.zeros(X.shape[1] + 1)
            return

        X = X[valid]
        y = y[valid]

        # --------------------------------------------------------
        # Standardize model matrix globally over training samples.
        # This prevents large interaction columns dominating the
        # ridge solution.
        # --------------------------------------------------------

        x_mean = X.mean(axis=0)
        x_std = X.std(axis=0)

        x_std = np.where(
            x_std < self.epsilon,
            1.0,
            x_std
        )

        Xn = (X - x_mean) / x_std

        # Save normalization parameters.
        self.x_mean = x_mean
        self.x_std = x_std

        # --------------------------------------------------------
        # Add intercept.
        # --------------------------------------------------------

        Xb = np.column_stack(
            [
                np.ones(len(Xn)),
                Xn,
            ]
        )

        p = Xb.shape[1]

        # Ridge penalty.
        penalty = np.eye(p, dtype=np.float64)

        # Do not penalize intercept.
        penalty[0, 0] = 0.0

        A = Xb.T @ Xb + self.ridge * penalty
        b = Xb.T @ y

        try:
            self.weights = np.linalg.solve(A, b)

        except np.linalg.LinAlgError:
            # Stable fallback.
            self.weights = np.linalg.pinv(A) @ b

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    def predict(self, features):
        X, _ = self._extract_features(features)

        rows = len(features.index)

        if isinstance(features.columns, pd.MultiIndex):
            tickers = features.columns.get_level_values(1).unique()
        else:
            tickers = features.columns

        tickers = list(tickers)

        cols = len(tickers)

        # --------------------------------------------------------
        # Defensive shape handling.
        # --------------------------------------------------------

        if (
            self.weights is None
            or not hasattr(self, "x_mean")
            or X.shape[1] != len(self.x_mean)
            or len(self.weights) != X.shape[1] + 1
        ):
            raw = np.zeros(len(X), dtype=np.float64)

        else:
            Xn = (
                X - self.x_mean
            ) / np.where(
                self.x_std < self.epsilon,
                1.0,
                self.x_std,
            )

            Xn = np.nan_to_num(
                Xn,
                nan=0.0,
                posinf=self.clip_value,
                neginf=-self.clip_value,
            )

            Xb = np.column_stack(
                [
                    np.ones(len(Xn)),
                    Xn,
                ]
            )

            raw = Xb @ self.weights

        # --------------------------------------------------------
        # Restore panel.
        # --------------------------------------------------------

        expected = rows * cols

        if len(raw) != expected:
            raw = np.resize(raw, expected)

        signal = pd.DataFrame(
            raw.reshape(rows, cols),
            index=features.index,
            columns=tickers,
        )

        signal = signal.replace(
            [np.inf, -np.inf],
            0.0
        ).fillna(0.0)

        # --------------------------------------------------------
        # Cross-sectional neutralization.
        # --------------------------------------------------------

        signal = signal.sub(
            signal.mean(axis=1),
            axis=0
        )

        # --------------------------------------------------------
        # Mild temporal smoothing.
        #
        # alpha=0.20 keeps substantially more current information
        # than the original 0.08.
        # --------------------------------------------------------

        if self.prev_signal is not None:
            previous = self.prev_signal.reindex(
                index=signal.index,
                columns=signal.columns,
                fill_value=0.0,
            )

            signal = (
                self.smooth_alpha * signal
                + (1.0 - self.smooth_alpha) * previous
            )

        self.prev_signal = signal.copy()

        # --------------------------------------------------------
        # Non-linear compression.
        # --------------------------------------------------------

        signal = np.tanh(
            signal / 1.5
        )

        # --------------------------------------------------------
        # Final cross-sectional de-meaning.
        # --------------------------------------------------------

        signal = signal.sub(
            signal.mean(axis=1),
            axis=0
        )

        # --------------------------------------------------------
        # L1 portfolio normalization.
        #
        # Sum(abs(weights)) ~= 1
        # Sum(weights) ~= 0
        # --------------------------------------------------------

        l1 = signal.abs().sum(axis=1)

        l1 = l1.replace(
            0.0,
            self.epsilon
        )

        output = signal.div(
            l1,
            axis=0
        )

        # Absolute final neutrality enforcement.
        output = output.sub(
            output.mean(axis=1),
            axis=0
        )

        # Re-normalize after neutrality adjustment.
        l1_final = output.abs().sum(axis=1)

        l1_final = l1_final.replace(
            0.0,
            self.epsilon
        )

        output = output.div(
            l1_final,
            axis=0
        )

        return output.fillna(0.0)
