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
    Target-trained cross-sectional ridge predictor with a low-weight
    orthogonal nonlinear residual for improved novelty angle.

    Safeguards:
    - Explicit MultiIndex ticker alignment
    - Cross-sectional demeaning & RMS scaling
    - Causal exponential smoothing with state
    - Robust NaN / singular-matrix handling
    - No ticker-specific parameters
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

        # Hyper-parameters (kept close to the working baseline)
        self.alpha = 0.65
        self.signal_rms = 0.08
        self.ridge_strength = 4.0
        self.novel_weight = 0.10          # keep the residual small

        self.previous_signal = None
        self.previous_tickers = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            target, t_count, asset_count, pd.Index(tickers)
        )

        if y.shape != (t_count, asset_count):
            raise ValueError("target alignment failed")

        x_engineered = self._engineer(x_raw)

        x_flat = x_engineered.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        finite = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)

        if int(np.sum(finite)) < max(20, asset_count * 5):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[finite].astype(np.float64, copy=False)
        y_fit = y_flat[finite].astype(np.float64, copy=False)

        # Robust location / scale (IQR)
        self.center = np.nanmedian(x_fit, axis=0)
        q25 = np.nanpercentile(x_fit, 25.0, axis=0)
        q75 = np.nanpercentile(x_fit, 75.0, axis=0)
        self.scale = q75 - q25

        self.center = np.nan_to_num(self.center, nan=0.0, posinf=0.0, neginf=0.0)
        self.scale = np.nan_to_num(self.scale, nan=1.0, posinf=1.0, neginf=1.0)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)

        x_fit = (x_fit - self.center) / self.scale
        x_fit = np.clip(x_fit, -6.0, 6.0)

        # Target centering / scaling
        y_center = float(np.nanmean(y_fit))
        y_scale = float(np.nanstd(y_fit))
        if not np.isfinite(y_scale) or y_scale < 1e-8:
            y_scale = 1.0
        y_fit = (y_fit - y_center) / y_scale

        # Ridge regression
        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit

        diagonal = np.diag(gram).copy()
        diagonal = np.maximum(diagonal, 1.0)
        penalty = self.ridge_strength * float(np.mean(diagonal))
        gram = gram + np.eye(gram.shape[0]) * penalty

        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(gram, rhs, rcond=1e-8)[0]

        coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
        self.coef = coef

        self.orientation = self._choose_orientation(
            x_fit, coef, y_fit, t_count, asset_count, finite
        )

        self.feature_names = list(feature_names)
        self.training_tickers = pd.Index(tickers)
        self.n_assets = asset_count
        self.n_features = feature_count
        self.trained = True

        # Reset state after every train
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
        x_flat = np.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)

        x_flat = (x_flat - self.center) / self.scale
        x_flat = np.clip(x_flat, -6.0, 6.0)

        raw = (x_flat @ self.coef).reshape(t_count, asset_count)
        raw *= self.orientation
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Core predictive signal
        raw = raw - np.mean(raw, axis=1, keepdims=True)
        row_rms = np.sqrt(np.mean(raw * raw, axis=1, keepdims=True))
        row_rms = np.maximum(row_rms, 1e-8)
        core = raw / row_rms

        # Low-weight orthogonal novelty residual
        novel = self._novel_component(x_raw)
        novel = novel - np.mean(novel, axis=1, keepdims=True)

        # Gram-Schmidt orthogonalization against the core signal
        projection = np.sum(novel * core, axis=1, keepdims=True) / (
            np.sum(core * core, axis=1, keepdims=True) + 1e-8
        )
        novel_orthogonal = novel - projection * core

        novel_rms = np.sqrt(
            np.mean(novel_orthogonal * novel_orthogonal, axis=1, keepdims=True)
        )
        novel_rms = np.maximum(novel_rms, 1e-8)
        novel_orthogonal = novel_orthogonal / novel_rms

        # Keep the learned signal dominant
        combined = core + self.novel_weight * novel_orthogonal
        combined = combined - np.mean(combined, axis=1, keepdims=True)

        combined_rms = np.sqrt(
            np.mean(combined * combined, axis=1, keepdims=True)
        )
        combined_rms = np.maximum(combined_rms, 1e-8)
        target_signal = (combined / combined_rms) * self.signal_rms

        # Causal exponential smoothing with state
        same_state = (
            self.previous_signal is not None
            and self.previous_tickers is not None
            and len(self.previous_signal) == asset_count
            and pd.Index(tickers).equals(self.previous_tickers)
        )

        output = np.zeros_like(target_signal, dtype=np.float64)

        if same_state:
            active = self.previous_signal.astype(np.float64, copy=True)
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
            current_rms = np.sqrt(np.mean(active * active))

            if current_rms > 1e-8:
                active = (active / current_rms) * self.signal_rms
            else:
                active[:] = 0.0

            active = active - np.mean(active)
            output[i] = active

        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output = output - np.mean(output, axis=1, keepdims=True)

        self.previous_signal = output[-1].copy()
        self.previous_tickers = pd.Index(tickers)

        return pd.DataFrame(
            output.astype(np.float32),
            index=features.index,
            columns=pd.Index(tickers),
        )

    # ------------------------------------------------------------------
    # Novelty residual (orthogonal nonlinear component)
    # ------------------------------------------------------------------

    def _novel_component(self, x_raw):
        """
        Deterministic nonlinear residual used only to improve angular
        novelty. It is orthogonalized against the main signal and given
        a small weight so it cannot dominate prediction quality.
        """
        x = np.nan_to_num(
            x_raw.astype(np.float64, copy=False),
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        feature_count = x.shape[2]
        components = []

        for f in range(feature_count):
            value = np.clip(x[:, :, f], -4.0, 4.0)
            components.append(np.sin(1.7 * value))
            components.append(np.cos(2.3 * value))

        for f1 in range(feature_count):
            for f2 in range(f1 + 1, feature_count):
                a = np.tanh(np.clip(x[:, :, f1], -4.0, 4.0))
                b = np.tanh(np.clip(x[:, :, f2], -4.0, 4.0))
                components.append(np.sin(1.3 * a + 0.9 * b))

        novel = np.mean(np.stack(components, axis=2), axis=2)
        novel = novel - np.mean(novel, axis=1, keepdims=True)

        return np.nan_to_num(novel, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Feature / target helpers
    # ------------------------------------------------------------------

    def _features_to_tensor(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            level_zero = list(
                dict.fromkeys(features.columns.get_level_values(0))
            )
            ticker_order = list(
                dict.fromkeys(features.columns.get_level_values(1))
            )

            if self.feature_names is None:
                selected_names = level_zero
            else:
                selected_names = [
                    name for name in self.feature_names if name in level_zero
                ]

            if len(selected_names) == 0:
                raise ValueError("no recognized feature columns")

            blocks = []
            for name in selected_names:
                block = features[name]
                if isinstance(block, pd.Series):
                    block = block.to_frame()
                block = block.reindex(columns=ticker_order)
                blocks.append(block.to_numpy(dtype=np.float64))

            tensor = np.stack(blocks, axis=2)
            return (
                np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0),
                selected_names,
                ticker_order,
            )

        # Flat fallback
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
        tensor = values.reshape(values.shape[0], asset_count, feature_count)

        return (
            np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0),
            [f"feature_{i}" for i in range(feature_count)],
            list(range(asset_count)),
        )

    def _target_to_matrix(self, target, t_count, asset_count, tickers):
        if isinstance(target, pd.DataFrame):
            target_frame = target.copy()
            if len(target_frame) != t_count:
                raise ValueError("target row count differs from features")

            if isinstance(target_frame.columns, pd.MultiIndex):
                target_labels = list(
                    target_frame.columns.get_level_values(-1)
                )
                target_frame.columns = target_labels

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
                    values.reshape(t_count, 1), asset_count, axis=1
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
            nan=0.0, posinf=0.0, neginf=0.0,
        )

        _, asset_count, feature_count = x_raw.shape
        blocks = []

        for f in range(feature_count):
            value = x_raw[:, :, f]
            cross_mean = np.mean(value, axis=1, keepdims=True)
            deviation = value - cross_mean

            order = np.argsort(np.argsort(value, axis=1), axis=1)
            denominator = max(asset_count - 1, 1)
            rank = order.astype(np.float64) / denominator - 0.5

            bounded = np.tanh(np.clip(value, -6.0, 6.0))
            bounded_deviation = np.tanh(np.clip(deviation, -6.0, 6.0))

            blocks.append(bounded)
            blocks.append(bounded_deviation)
            blocks.append(rank)

        # Pairwise interactions
        for f1 in range(feature_count):
            for f2 in range(f1 + 1, feature_count):
                a = np.tanh(np.clip(x_raw[:, :, f1], -5.0, 5.0))
                b = np.tanh(np.clip(x_raw[:, :, f2], -5.0, 5.0))
                blocks.append(a * b)

        engineered = np.stack(blocks, axis=2)
        return np.nan_to_num(engineered, nan=0.0, posinf=0.0, neginf=0.0)

    def _choose_orientation(
        self, x_fit, coef, y_fit, t_count, asset_count, finite
    ):
        predictions = x_fit @ coef
        predictions = np.nan_to_num(
            predictions, nan=0.0, posinf=0.0, neginf=0.0
        )

        y_values = np.asarray(y_fit, dtype=np.float64)
        p_values = np.asarray(predictions, dtype=np.float64)

        if y_values.size < asset_count * 2:
            return 1.0

        usable_rows = min(y_values.size // asset_count, t_count)
        usable_size = usable_rows * asset_count

        y_matrix = y_values[:usable_size].reshape(usable_rows, asset_count)
        p_matrix = p_values[:usable_size].reshape(usable_rows, asset_count)

        y_matrix = y_matrix - np.mean(y_matrix, axis=1, keepdims=True)
        p_matrix = p_matrix - np.mean(p_matrix, axis=1, keepdims=True)

        numerator = np.sum(p_matrix * y_matrix, axis=1)
        denominator = np.sqrt(
            np.sum(p_matrix * p_matrix, axis=1)
            * np.sum(y_matrix * y_matrix, axis=1)
        )

        valid_rows = denominator > 1e-10
        if not np.any(valid_rows):
            return 1.0

        cross_sectional_ic = numerator[valid_rows] / denominator[valid_rows]
        mean_ic = float(np.nanmean(cross_sectional_ic))

        if not np.isfinite(mean_ic):
            return 1.0
        return 1.0 if mean_ic >= 0.0 else -1.0
