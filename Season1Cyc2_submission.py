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
    Target-trained cross-sectional ridge predictor.

    Main safeguards:
    - Explicit MultiIndex ticker alignment.
    - Cross-sectional target/sign diagnostics.
    - No ticker-specific parameters.
    - Row-order-independent feature construction.
    - Cross-sectional de-meaning.
    - Conservative causal smoothing.
    - Robust NaN and singular-matrix handling.
    """

    def __init__(self):
        try:
            super().__init__()
        except Exception:
            pass

        self.trained = False
        self.feature_names = None
        self.training_tickers = None
        self.n_assets = None
        self.n_features = None

        self.center = None
        self.scale = None
        self.coef = None
        self.orientation = 1.0

        # Less persistent than the previous candidate.
        self.alpha = 0.65

        # Lower position scale reduces turnover drag.
        self.signal_rms = 0.08

        # Stronger regularization than an unregularized expansion,
        # but less aggressive than the previous candidate.
        self.ridge_strength = 4.0

        self.previous_signal = None
        self.previous_tickers = None

    def train(self, features, target):
        if features is None or target is None:
            raise ValueError("features and target are required")

        x_raw, feature_names, tickers = self._features_to_tensor(features)

        if x_raw.ndim != 3:
            raise ValueError("invalid feature tensor")

        t_count, asset_count, feature_count = x_raw.shape

        if t_count < 10 or asset_count < 2 or feature_count < 1:
            raise ValueError("insufficient training dimensions")

        y = self._target_to_matrix(
            target,
            t_count,
            asset_count,
            pd.Index(tickers),
        )

        if y.shape != (t_count, asset_count):
            raise ValueError("target alignment failed")

        x_engineered = self._engineer(x_raw)

        x_flat = x_engineered.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        finite = np.isfinite(y_flat)
        finite &= np.all(np.isfinite(x_flat), axis=1)

        if int(np.sum(finite)) < max(20, asset_count * 5):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[finite].astype(np.float64, copy=False)
        y_fit = y_flat[finite].astype(np.float64, copy=False)

        self.center = np.nanmedian(x_fit, axis=0)
        q25 = np.nanpercentile(x_fit, 25.0, axis=0)
        q75 = np.nanpercentile(x_fit, 75.0, axis=0)

        self.scale = q75 - q25

        self.center = np.nan_to_num(
            self.center,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        self.scale = np.nan_to_num(
            self.scale,
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

        self.scale = np.where(self.scale < 1.0e-8, 1.0, self.scale)

        x_fit = (x_fit - self.center) / self.scale
        x_fit = np.clip(x_fit, -6.0, 6.0)

        y_center = float(np.nanmean(y_fit))
        y_scale = float(np.nanstd(y_fit))

        if not np.isfinite(y_scale) or y_scale < 1.0e-8:
            y_scale = 1.0

        y_fit = (y_fit - y_center) / y_scale

        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit

        diagonal = np.diag(gram).copy()
        diagonal = np.maximum(diagonal, 1.0)

        penalty = self.ridge_strength * float(np.mean(diagonal))
        gram = gram + np.eye(gram.shape[0]) * penalty

        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(
                gram,
                rhs,
                rcond=1.0e-8,
            )[0]

        coef = np.nan_to_num(
            coef,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        self.coef = coef
        self.orientation = self._choose_orientation(
            x_fit,
            coef,
            y_fit,
            t_count,
            asset_count,
            finite,
        )

        self.feature_names = list(feature_names)
        self.training_tickers = pd.Index(tickers)
        self.n_assets = asset_count
        self.n_features = feature_count
        self.trained = True

        # Do not carry state from a previous training/evaluation cycle.
        self.previous_signal = None
        self.previous_tickers = None

    def predict(self, features):
        if not self.trained:
            raise RuntimeError("predictor has not been trained")

        x_raw, _, tickers = self._features_to_tensor(features)

        t_count, asset_count, _ = x_raw.shape

        if asset_count != self.n_assets:
            raise ValueError("asset count differs from training")

        x_engineered = self._engineer(x_raw)
        x_flat = x_engineered.reshape(t_count * asset_count, -1)

        x_flat = np.nan_to_num(
            x_flat,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        x_flat = (x_flat - self.center) / self.scale
        x_flat = np.clip(x_flat, -6.0, 6.0)

        raw = x_flat @ self.coef
        raw = raw.reshape(t_count, asset_count)
        raw = raw * self.orientation

        raw = np.nan_to_num(
            raw,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        raw = raw - np.mean(raw, axis=1, keepdims=True)

        row_rms = np.sqrt(
            np.mean(raw * raw, axis=1, keepdims=True)
        )
        row_rms = np.maximum(row_rms, 1.0e-8)

        target_signal = raw / row_rms
        target_signal = target_signal * self.signal_rms

        same_state = (
            self.previous_signal is not None
            and self.previous_tickers is not None
            and len(self.previous_signal) == asset_count
            and pd.Index(tickers).equals(self.previous_tickers)
        )

        output = np.zeros_like(target_signal, dtype=np.float64)

        if same_state:
            active = self.previous_signal.astype(
                np.float64,
                copy=True,
            )
        else:
            active = target_signal[0].copy()

        for i in range(t_count):
            if i == 0 and not same_state:
                active = target_signal[i].copy()
            else:
                active = (
                    self.alpha * target_signal[i]
                    + (1.0 - self.alpha) * active
                )

            active = active - np.mean(active)

            current_rms = np.sqrt(
                np.mean(active * active)
            )

            if current_rms > 1.0e-8:
                active = active / current_rms
                active = active * self.signal_rms
            else:
                active[:] = 0.0

            active = active - np.mean(active)
            output[i] = active

        output = np.nan_to_num(
            output,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        output = output - np.mean(
            output,
            axis=1,
            keepdims=True,
        )

        self.previous_signal = output[-1].copy()
        self.previous_tickers = pd.Index(tickers)

        return pd.DataFrame(
            output.astype(np.float32),
            index=features.index,
            columns=pd.Index(tickers),
        )

    def _features_to_tensor(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            level_zero = list(
                dict.fromkeys(
                    features.columns.get_level_values(0)
                )
            )

            ticker_order = list(
                dict.fromkeys(
                    features.columns.get_level_values(1)
                )
            )

            if self.feature_names is None:
                selected_names = level_zero
            else:
                selected_names = [
                    name
                    for name in self.feature_names
                    if name in level_zero
                ]

            if len(selected_names) == 0:
                raise ValueError("no recognized feature columns")

            blocks = []

            for name in selected_names:
                block = features[name]

                if isinstance(block, pd.Series):
                    block = block.to_frame()

                block = block.reindex(columns=ticker_order)

                blocks.append(
                    block.to_numpy(dtype=np.float64)
                )

            tensor = np.stack(blocks, axis=2)

            return (
                np.nan_to_num(
                    tensor,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                selected_names,
                ticker_order,
            )

        values = features.to_numpy(dtype=np.float64)

        if self.n_features is None:
            feature_count = 6
        else:
            feature_count = self.n_features

        if values.ndim != 2:
            raise ValueError("flat features must be two-dimensional")

        if values.shape[1] % feature_count != 0:
            raise ValueError(
                "flat feature count is not divisible by feature count"
            )

        asset_count = values.shape[1] // feature_count

        tensor = values.reshape(
            values.shape[0],
            asset_count,
            feature_count,
        )

        return (
            np.nan_to_num(
                tensor,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            [f"feature_{i}" for i in range(feature_count)],
            list(range(asset_count)),
        )

    def _target_to_matrix(
        self,
        target,
        t_count,
        asset_count,
        tickers,
    ):
        if isinstance(target, pd.DataFrame):
            target_frame = target.copy()

            if len(target_frame) != t_count:
                raise ValueError("target row count differs from features")

            if isinstance(target_frame.columns, pd.MultiIndex):
                target_labels = list(
                    target_frame.columns.get_level_values(-1)
                )

                target_frame.columns = target_labels

            else:
                target_labels = list(target_frame.columns)

            # Prefer explicit ticker matching.
            if set(tickers).issubset(set(target_frame.columns)):
                aligned = target_frame.reindex(columns=list(tickers))

                if aligned.shape == (t_count, asset_count):
                    return aligned.to_numpy(dtype=np.float64)

            values = target_frame.to_numpy(dtype=np.float64)

            if values.shape == (t_count, asset_count):
                return values

            if values.shape == (t_count, 1):
                return np.repeat(values, asset_count, axis=1)

        values = np.asarray(target, dtype=np.float64)

        if values.ndim == 1:
            if values.size == t_count * asset_count:
                return values.reshape(t_count, asset_count)

            if values.size == t_count:
                return np.repeat(
                    values.reshape(t_count, 1),
                    asset_count,
                    axis=1,
                )

        if values.ndim == 2:
            if values.shape == (t_count, asset_count):
                return values

            if values.shape == (t_count, 1):
                return np.repeat(values, asset_count, axis=1)

        raise ValueError("unsupported target shape")

    def _engineer(self, x_raw):
        x_raw = np.nan_to_num(
            x_raw.astype(np.float64, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        _, asset_count, feature_count = x_raw.shape
        blocks = []

        for f in range(feature_count):
            value = x_raw[:, :, f]

            cross_mean = np.mean(
                value,
                axis=1,
                keepdims=True,
            )

            deviation = value - cross_mean

            order = np.argsort(
                np.argsort(value, axis=1),
                axis=1,
            )

            denominator = max(asset_count - 1, 1)
            rank = order.astype(np.float64) / denominator
            rank = rank - 0.5

            bounded = np.tanh(
                np.clip(value, -6.0, 6.0)
            )

            bounded_deviation = np.tanh(
                np.clip(deviation, -6.0, 6.0)
            )

            blocks.append(bounded)
            blocks.append(bounded_deviation)
            blocks.append(rank)

        # A restrained interaction set, rather than a large
        # high-frequency expansion.
        for f1 in range(feature_count):
            for f2 in range(f1 + 1, feature_count):
                a = np.tanh(
                    np.clip(x_raw[:, :, f1], -5.0, 5.0)
                )
                b = np.tanh(
                    np.clip(x_raw[:, :, f2], -5.0, 5.0)
                )

                blocks.append(a * b)

        engineered = np.stack(blocks, axis=2)

        return np.nan_to_num(
            engineered,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _choose_orientation(
        self,
        x_fit,
        coef,
        y_fit,
        t_count,
        asset_count,
        finite,
    ):
        predictions = x_fit @ coef
        predictions = np.nan_to_num(
            predictions,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        y_values = np.asarray(y_fit, dtype=np.float64)
        p_values = np.asarray(predictions, dtype=np.float64)

        if y_values.size < asset_count * 2:
            return 1.0

        usable_rows = min(
            y_values.size // asset_count,
            t_count,
        )

        usable_size = usable_rows * asset_count

        y_matrix = y_values[:usable_size].reshape(
            usable_rows,
            asset_count,
        )

        p_matrix = p_values[:usable_size].reshape(
            usable_rows,
            asset_count,
        )

        y_matrix = y_matrix - np.mean(
            y_matrix,
            axis=1,
            keepdims=True,
        )

        p_matrix = p_matrix - np.mean(
            p_matrix,
            axis=1,
            keepdims=True,
        )

        numerator = np.sum(
            p_matrix * y_matrix,
            axis=1,
        )

        denominator = np.sqrt(
            np.sum(p_matrix * p_matrix, axis=1)
            * np.sum(y_matrix * y_matrix, axis=1)
        )

        valid_rows = denominator > 1.0e-10

        if not np.any(valid_rows):
            return 1.0

        cross_sectional_ic = (
            numerator[valid_rows]
            / denominator[valid_rows]
        )

        mean_ic = float(np.nanmean(cross_sectional_ic))

        if not np.isfinite(mean_ic):
            return 1.0

        return 1.0 if mean_ic >= 0.0 else -1.0
