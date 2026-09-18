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
    Cross-sectional ridge signal with nonlinear feature expansion.

    Design goals:
    - learns from the supplied target;
    - no ticker-specific learned parameters;
    - row-order independent feature construction;
    - cross-sectional de-meaning;
    - turnover-aware causal smoothing;
    - robust handling of MultiIndex and flat DataFrames;
    - no external files or city-coordinate features.
    """

    def __init__(self):
        try:
            super().__init__()
        except Exception:
            pass

        self.trained = False
        self.feature_names = None
        self.tickers = None
        self.n_assets = None
        self.n_base_features = None
        self.n_model_features = None

        self.center = None
        self.scale = None
        self.coef = None
        self.intercept = 0.0

        self.alpha = 0.20
        self.signal_rms = 0.12
        self.ridge_strength = 12.0
        self.previous_signal = None
        self.previous_tickers = None

    def train(self, features, target):
        if features is None or target is None:
            raise ValueError("features and target are required")

        x_raw, names, tickers = self._features_to_tensor(features)

        if x_raw.ndim != 3:
            raise ValueError("features must represent a T x J x F tensor")

        t_count, asset_count, feature_count = x_raw.shape

        if t_count < 20 or asset_count < 2 or feature_count < 1:
            raise ValueError("insufficient training dimensions")

        y = self._target_to_matrix(target, t_count, asset_count, tickers)

        if y.shape != (t_count, asset_count):
            raise ValueError("target cannot be aligned to feature dimensions")

        x_engineered = self._engineer(x_raw)
        x_flat = x_engineered.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        valid = np.isfinite(y_flat)
        valid &= np.all(np.isfinite(x_flat), axis=1)

        if np.sum(valid) < max(20, asset_count * 5):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[valid].astype(np.float64, copy=False)
        y_fit = y_flat[valid].astype(np.float64, copy=False)

        self.center = np.nanmedian(x_fit, axis=0)
        q25 = np.nanpercentile(x_fit, 25.0, axis=0)
        q75 = np.nanpercentile(x_fit, 75.0, axis=0)
        self.scale = q75 - q25

        self.center = np.nan_to_num(self.center, nan=0.0, posinf=0.0, neginf=0.0)
        self.scale = np.nan_to_num(self.scale, nan=1.0, posinf=1.0, neginf=1.0)
        self.scale = np.where(self.scale < 1.0e-8, 1.0, self.scale)

        x_fit = (x_fit - self.center) / self.scale
        x_fit = np.clip(x_fit, -8.0, 8.0)

        y_center = float(np.mean(y_fit))
        y_scale = float(np.std(y_fit))
        if not np.isfinite(y_scale) or y_scale < 1.0e-8:
            y_scale = 1.0

        y_fit = (y_fit - y_center) / y_scale

        # Add an intercept column implicitly through centering.
        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit

        diagonal = np.diag(gram).copy()
        diagonal = np.maximum(diagonal, 1.0)
        regularization = self.ridge_strength * np.mean(diagonal)

        gram = gram + np.eye(gram.shape[0], dtype=np.float64) * regularization

        try:
            self.coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.lstsq(gram, rhs, rcond=1.0e-8)[0]

        self.intercept = 0.0
        self.feature_names = list(names)
        self.tickers = pd.Index(tickers)
        self.n_assets = asset_count
        self.n_base_features = feature_count
        self.n_model_features = x_engineered.shape[2]
        self.trained = True

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
        x_flat = np.clip(x_flat, -8.0, 8.0)

        raw = x_flat @ self.coef + self.intercept
        raw = raw.reshape(t_count, asset_count)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Mandatory cross-sectional de-meaning.
        raw = raw - np.mean(raw, axis=1, keepdims=True)

        # Normalize each row to a stable RMS scale.
        row_rms = np.sqrt(np.mean(raw * raw, axis=1, keepdims=True))
        row_rms = np.maximum(row_rms, 1.0e-8)
        target_signal = raw / row_rms * self.signal_rms

        # Use prior state only when ticker identity and dimensions agree.
        use_previous = (
            self.previous_signal is not None
            and self.previous_tickers is not None
            and len(self.previous_signal) == asset_count
            and pd.Index(tickers).equals(self.previous_tickers)
        )

        output = np.zeros_like(target_signal, dtype=np.float64)

        if use_previous:
            active = self.previous_signal.astype(np.float64, copy=True)
        else:
            active = np.zeros(asset_count, dtype=np.float64)

        for i in range(t_count):
            if i == 0 and not use_previous:
                active = target_signal[i].copy()
            else:
                active = (
                    self.alpha * target_signal[i]
                    + (1.0 - self.alpha) * active
                )

            active = active - np.mean(active)

            current_rms = np.sqrt(np.mean(active * active))
            if current_rms > 1.0e-8:
                active = active / current_rms * self.signal_rms
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

    def _features_to_tensor(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            level_zero = list(dict.fromkeys(features.columns.get_level_values(0)))
            level_one = list(dict.fromkeys(features.columns.get_level_values(1)))

            if self.feature_names is not None:
                selected_names = [
                    name for name in self.feature_names
                    if name in level_zero
                ]
            else:
                selected_names = level_zero

            if len(selected_names) == 0:
                raise ValueError("no recognized feature columns")

            blocks = []

            for name in selected_names:
                block = features[name]

                if isinstance(block, pd.Series):
                    block = block.to_frame()

                block = block.reindex(columns=level_one)
                blocks.append(block.to_numpy(dtype=np.float64))

            tensor = np.stack(blocks, axis=2)

            return (
                np.nan_to_num(
                    tensor,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                selected_names,
                level_one,
            )

        values = features.to_numpy(dtype=np.float64)

        if self.n_base_features is not None:
            feature_count = self.n_base_features
        else:
            feature_count = 6

        if values.shape[1] % feature_count != 0:
            raise ValueError(
                "flat feature count is not divisible by the feature count"
            )

        asset_count = values.shape[1] // feature_count
        tensor = values.reshape(values.shape[0], asset_count, feature_count)

        tickers = list(range(asset_count))
        names = [f"feature_{i}" for i in range(feature_count)]

        return (
            np.nan_to_num(
                tensor,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            names,
            tickers,
        )

    def _target_to_matrix(self, target, t_count, asset_count, tickers):
        if isinstance(target, pd.DataFrame):
            if isinstance(target.columns, pd.MultiIndex):
                blocks = []

                for level in range(target.columns.nlevels):
                    pass

                target_values = target.to_numpy(dtype=np.float64)

                if target_values.shape == (t_count, asset_count):
                    return target_values

                if target_values.shape[1] == asset_count:
                    return target_values

                if target_values.shape[1] == 1:
                    return np.repeat(
                        target_values,
                        asset_count,
                        axis=1,
                    )

            target_values = target.to_numpy(dtype=np.float64)

            if target_values.ndim == 2:
                if target_values.shape == (t_count, asset_count):
                    return target_values

                if target_values.shape[1] == 1:
                    return np.repeat(
                        target_values,
                        asset_count,
                        axis=1,
                    )

        target_values = np.asarray(target, dtype=np.float64)

        if target_values.ndim == 1:
            if target_values.size == t_count * asset_count:
                return target_values.reshape(t_count, asset_count)

            if target_values.size == t_count:
                return np.repeat(
                    target_values.reshape(t_count, 1),
                    asset_count,
                    axis=1,
                )

        if target_values.ndim == 2:
            if target_values.shape == (t_count, asset_count):
                return target_values

            if target_values.shape == (t_count, 1):
                return np.repeat(target_values, asset_count, axis=1)

        raise ValueError("unsupported target shape")

    def _engineer(self, x_raw):
        x_raw = np.nan_to_num(
            x_raw.astype(np.float64, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        t_count, asset_count, feature_count = x_raw.shape
        blocks = []

        # Original values, cross-sectional deviations, and rank-like values.
        for f in range(feature_count):
            value = x_raw[:, :, f]
            cross_mean = np.mean(value, axis=1, keepdims=True)
            deviation = value - cross_mean

            order = np.argsort(np.argsort(value, axis=1), axis=1)
            rank = order.astype(np.float64) / max(asset_count - 1, 1)
            rank = rank - 0.5

            blocks.append(np.tanh(np.clip(value, -8.0, 8.0)))
            blocks.append(np.tanh(np.clip(deviation, -8.0, 8.0)))
            blocks.append(rank)

        # Pairwise nonlinear interactions.
        for f1 in range(feature_count):
            for f2 in range(f1 + 1, feature_count):
                a = np.tanh(np.clip(x_raw[:, :, f1], -6.0, 6.0))
                b = np.tanh(np.clip(x_raw[:, :, f2], -6.0, 6.0))

                blocks.append(a * b)
                blocks.append(np.sign(a - b) * np.sqrt(np.abs(a * b) + 1.0e-8))

        engineered = np.stack(blocks, axis=2)

        return np.nan_to_num(
            engineered,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
