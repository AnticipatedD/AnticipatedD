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
    Balanced Cross-Sectional Ridge
    - Adaptive sign correction for positive IC
    - Stronger non-linear / trigonometric features for City & Global Novelty > 60°
    - Robust scaling + double demeaning
    - Mild continuous rank blend (does not destroy novelty)
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

        self.alpha = 0.28
        self.signal_rms = 0.10
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
        if t_count < 15 or asset_count < 2 or feature_count < 1:
            raise ValueError("insufficient training dimensions")

        y = self._target_to_matrix(target, t_count, asset_count, tickers)

        x_engineered = self._engineer(x_raw)
        x_flat = x_engineered.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        valid = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)
        if np.sum(valid) < max(20, asset_count * 4):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[valid].astype(np.float64, copy=False)
        y_fit = y_flat[valid].astype(np.float64, copy=False)

        # Robust scaling
        self.center = np.nanmedian(x_fit, axis=0)
        q20 = np.nanpercentile(x_fit, 20.0, axis=0)
        q80 = np.nanpercentile(x_fit, 80.0, axis=0)
        self.scale = q80 - q20

        self.center = np.nan_to_num(self.center, nan=0.0, posinf=0.0, neginf=0.0)
        self.scale = np.nan_to_num(self.scale, nan=1.0, posinf=1.0, neginf=1.0)
        self.scale = np.where(self.scale < 1e-10, 1.0, self.scale)

        x_fit = np.clip((x_fit - self.center) / self.scale, -6.0, 6.0)

        y_center = float(np.mean(y_fit))
        y_scale = float(np.std(y_fit))
        if not np.isfinite(y_scale) or y_scale < 1e-10:
            y_scale = 1.0
        y_fit = (y_fit - y_center) / y_scale

        # Ridge
        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit
        diag = np.maximum(np.diag(gram), 1.0)
        reg = self.ridge_strength * np.mean(diag)
        gram = gram + np.eye(gram.shape[0]) * reg

        try:
            self.coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.lstsq(gram, rhs, rcond=1e-8)[0]

        # Adaptive sign correction → positive IC
        train_pred = x_fit @ self.coef
        if np.std(train_pred) > 1e-10 and np.std(y_fit) > 1e-10:
            ic = np.corrcoef(train_pred, y_fit)[0, 1]
            if np.isfinite(ic) and ic < 0.0:
                self.coef *= -1.0

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

        x_engineered = self._engineer(x_raw)
        x_flat = x_engineered.reshape(t_count * asset_count, -1)
        x_flat = np.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
        x_flat = np.clip((x_flat - self.center) / self.scale, -6.0, 6.0)

        raw = (x_flat @ self.coef + self.intercept).reshape(t_count, asset_count)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Primary demean
        raw -= np.mean(raw, axis=1, keepdims=True)

        # Mild continuous rank blend (preserves novelty better than pure rank)
        rank_component = np.zeros_like(raw)
        for i in range(t_count):
            order = np.argsort(np.argsort(raw[i]))
            rank_component[i] = order.astype(np.float64) / max(asset_count - 1, 1) - 0.5

        # 80% continuous signal + 20% rank → balance of IC stability & novelty
        mixed = 0.80 * raw + 0.20 * rank_component
        mixed -= np.mean(mixed, axis=1, keepdims=True)

        # Scale to target RMS
        row_rms = np.sqrt(np.mean(mixed ** 2, axis=1, keepdims=True))
        row_rms = np.maximum(row_rms, 1e-10)
        target_signal = mixed / row_rms * self.signal_rms

        # EMA smoothing
        use_previous = (
            self.previous_signal is not None
            and self.previous_tickers is not None
            and len(self.previous_signal) == asset_count
            and pd.Index(tickers).equals(self.previous_tickers)
        )

        output = np.zeros_like(target_signal, dtype=np.float64)
        active = (
            self.previous_signal.astype(np.float64).copy()
            if use_previous
            else np.zeros(asset_count, dtype=np.float64)
        )

        for i in range(t_count):
            if i == 0 and not use_previous:
                active = target_signal[i].copy()
            else:
                active = self.alpha * target_signal[i] + (1.0 - self.alpha) * active

            active -= np.mean(active)
            rms = np.sqrt(np.mean(active ** 2))
            if rms > 1e-10:
                active = active / rms * self.signal_rms
            else:
                active[:] = 0.0
            active -= np.mean(active)
            output[i] = active

        output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
        output -= np.mean(output, axis=1, keepdims=True)

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

            selected_names = (
                [n for n in self.feature_names if n in level_zero]
                if self.feature_names is not None
                else level_zero
            )
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
                np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0),
                selected_names,
                level_one,
            )

        values = features.to_numpy(dtype=np.float64)
        feature_count = self.n_base_features if self.n_base_features is not None else 6
        if values.shape[1] % feature_count != 0:
            feature_count = 1

        asset_count = values.shape[1] // feature_count
        tensor = values.reshape(values.shape[0], asset_count, feature_count)
        return (
            np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0),
            [f"feature_{i}" for i in range(feature_count)],
            list(range(asset_count)),
        )

    def _target_to_matrix(self, target, t_count, asset_count, tickers):
        if isinstance(target, pd.DataFrame):
            vals = target.to_numpy(dtype=np.float64)
            if vals.ndim == 2:
                if vals.shape == (t_count, asset_count):
                    return vals
                if vals.shape[1] == 1:
                    return np.repeat(vals, asset_count, axis=1)

        vals = np.asarray(target, dtype=np.float64)
        if vals.ndim == 1:
            if vals.size == t_count * asset_count:
                return vals.reshape(t_count, asset_count)
            if vals.size == t_count:
                return np.repeat(vals.reshape(t_count, 1), asset_count, axis=1)
        if vals.ndim == 2:
            if vals.shape == (t_count, asset_count):
                return vals
            if vals.shape[1] == 1:
                return np.repeat(vals, asset_count, axis=1)

        return np.zeros((t_count, asset_count), dtype=np.float64)

    def _engineer(self, x_raw):
        """
        Richer feature set for novelty while keeping core predictive power.
        """
        x_raw = np.nan_to_num(x_raw.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        _, asset_count, feature_count = x_raw.shape
        blocks = []

        for f in range(feature_count):
            value = x_raw[:, :, f]

            mu = np.mean(value, axis=1, keepdims=True)
            deviation = value - mu

            med = np.median(value, axis=1, keepdims=True)
            mad = np.median(np.abs(value - med), axis=1, keepdims=True)
            mad = np.maximum(mad, 1e-8)
            robust_z = (value - med) / (1.4826 * mad)

            order = np.argsort(np.argsort(value, axis=1), axis=1)
            rank = order.astype(np.float64) / max(asset_count - 1, 1) - 0.5

            # Core predictive features
            blocks.append(rank)
            blocks.append(np.tanh(np.clip(robust_z, -5.0, 5.0)))
            blocks.append(np.tanh(np.clip(deviation, -5.0, 5.0)))
            blocks.append(np.tanh(np.clip(value, -5.0, 5.0)))

            # Stronger novelty / spatial projections
            blocks.append(np.sin(rank * np.pi))
            blocks.append(np.cos(rank * np.pi * 0.5))
            blocks.append(np.sin(np.clip(robust_z, -4, 4) * 0.8))
            blocks.append(np.cos(np.clip(deviation, -4, 4) * 0.6))

        # Pairwise interactions (important for novelty)
        max_f = min(feature_count, 5)
        for f1 in range(max_f):
            for f2 in range(f1 + 1, max_f):
                a = np.tanh(np.clip(x_raw[:, :, f1], -4.0, 4.0))
                b = np.tanh(np.clip(x_raw[:, :, f2], -4.0, 4.0))
                blocks.append(a * b)
                blocks.append(np.sign(a - b) * np.sqrt(np.abs(a * b) + 1e-8))

        engineered = np.stack(blocks, axis=2)
        return np.nan_to_num(engineered, nan=0.0, posinf=0.0, neginf=0.0)
