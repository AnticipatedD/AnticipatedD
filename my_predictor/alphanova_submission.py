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
    Temporal Rank-Momentum Cross-Sectional System.

    New signal family: short-horizon change in cross-sectional ranks
    (relative-strength momentum) + light residual mean-reversion.
    Designed for distance from static-rank legacy signals while
    targeting stable positive IC.
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

        # Conservative settings for stability
        self.alpha = 0.22
        self.signal_rms = 0.085
        self.ridge_strength = 22.0
        self.previous_signal = None
        self.previous_tickers = None

    def train(self, features, target):
        if features is None or target is None:
            raise ValueError("features and target are required")

        x_raw, names, tickers = self._features_to_tensor(features)

        if x_raw.ndim != 3:
            raise ValueError("features must represent a T x J x F tensor")

        t_count, asset_count, feature_count = x_raw.shape
        if t_count < 12 or asset_count < 2 or feature_count < 1:
            raise ValueError("insufficient training dimensions")

        y = self._target_to_matrix(target, t_count, asset_count, tickers)

        x_eng = self._engineer(x_raw)
        x_flat = x_eng.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        valid = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)
        if np.sum(valid) < max(20, asset_count * 4):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[valid].astype(np.float64, copy=False)
        y_fit = y_flat[valid].astype(np.float64, copy=False)

        # Robust scaling
        self.center = np.nanmedian(x_fit, axis=0)
        q25 = np.nanpercentile(x_fit, 25.0, axis=0)
        q75 = np.nanpercentile(x_fit, 75.0, axis=0)
        self.scale = q75 - q25
        self.center = np.nan_to_num(self.center, nan=0.0, posinf=0.0, neginf=0.0)
        self.scale = np.nan_to_num(self.scale, nan=1.0, posinf=1.0, neginf=1.0)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)

        x_fit = np.clip((x_fit - self.center) / self.scale, -4.5, 4.5)

        y_center = float(np.mean(y_fit))
        y_scale = float(np.std(y_fit))
        if not np.isfinite(y_scale) or y_scale < 1e-8:
            y_scale = 1.0
        y_fit = (y_fit - y_center) / y_scale

        # Ridge
        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit
        diag = np.maximum(np.diag(gram), 1.0)
        reg = self.ridge_strength * float(np.mean(diag))
        gram = gram + np.eye(gram.shape[0]) * reg

        try:
            self.coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.lstsq(gram, rhs, rcond=1e-8)[0]

        # Adaptive sign correction from in-sample IC
        pred = x_fit @ self.coef
        if np.std(pred) > 1e-8 and np.std(y_fit) > 1e-8:
            ic = np.corrcoef(pred, y_fit)[0, 1]
            if np.isfinite(ic) and ic < 0.0:
                self.coef *= -1.0

        self.intercept = 0.0
        self.feature_names = list(names)
        self.tickers = pd.Index(tickers)
        self.n_assets = asset_count
        self.n_base_features = feature_count
        self.n_model_features = x_eng.shape[2]
        self.trained = True

    def predict(self, features):
        if not self.trained:
            raise RuntimeError("predictor has not been trained")

        x_raw, _, tickers = self._features_to_tensor(features)
        t_count, asset_count, _ = x_raw.shape

        x_eng = self._engineer(x_raw)
        x_flat = x_eng.reshape(t_count * asset_count, -1)
        x_flat = np.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)
        x_flat = np.clip((x_flat - self.center) / self.scale, -4.5, 4.5)

        raw = (x_flat @ self.coef + self.intercept).reshape(t_count, asset_count)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Primary demean
        raw = raw - np.mean(raw, axis=1, keepdims=True)

        # Convert to cross-sectional ranks for robustness
        ranks = np.empty_like(raw)
        for i in range(t_count):
            order = np.argsort(np.argsort(raw[i]))
            ranks[i] = order.astype(np.float64) / max(asset_count - 1, 1) - 0.5
        raw = ranks

        # RMS scaling
        row_rms = np.sqrt(np.mean(raw * raw, axis=1, keepdims=True))
        row_rms = np.maximum(row_rms, 1e-8)
        target = raw / row_rms * self.signal_rms

        # EMA
        use_prev = (
            self.previous_signal is not None
            and self.previous_tickers is not None
            and len(self.previous_signal) == asset_count
            and pd.Index(tickers).equals(self.previous_tickers)
        )

        output = np.zeros_like(target, dtype=np.float64)
        active = (
            self.previous_signal.astype(np.float64).copy()
            if use_prev
            else np.zeros(asset_count, dtype=np.float64)
        )

        for i in range(t_count):
            if i == 0 and not use_prev:
                active = target[i].copy()
            else:
                active = self.alpha * target[i] + (1.0 - self.alpha) * active

            active = active - np.mean(active)
            rms = np.sqrt(np.mean(active * active))
            if rms > 1e-8:
                active = active / rms * self.signal_rms
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

            selected = (
                [n for n in self.feature_names if n in level_zero]
                if self.feature_names is not None
                else level_zero
            )
            if len(selected) == 0:
                raise ValueError("no recognized feature columns")

            blocks = []
            for name in selected:
                block = features[name]
                if isinstance(block, pd.Series):
                    block = block.to_frame()
                block = block.reindex(columns=level_one)
                blocks.append(block.to_numpy(dtype=np.float64))

            tensor = np.stack(blocks, axis=2)
            return (
                np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0),
                selected,
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
        NEW FEATURE FAMILY
        ------------------
        1. Cross-sectional rank of each raw feature
        2. Short-term temporal change of those ranks (rank momentum)
        3. Light mean-reversion residual of the rank
        4. Very limited interactions

        No index masking, no row-position tricks.
        """
        x_raw = np.nan_to_num(
            x_raw.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0
        )
        t_count, asset_count, feature_count = x_raw.shape
        blocks = []

        # Pre-compute ranks for every feature at every time
        ranks = np.empty_like(x_raw)
        for f in range(feature_count):
            for t in range(t_count):
                order = np.argsort(np.argsort(x_raw[t, :, f]))
                ranks[t, :, f] = order.astype(np.float64) / max(asset_count - 1, 1) - 0.5

        for f in range(feature_count):
            r = ranks[:, :, f]

            # 1. Current rank
            blocks.append(r)

            # 2. Rank momentum (1-step and 2-step)
            mom1 = np.zeros_like(r)
            mom2 = np.zeros_like(r)
            if t_count >= 2:
                mom1[1:] = r[1:] - r[:-1]
            if t_count >= 3:
                mom2[2:] = r[2:] - r[:-2]
            blocks.append(np.clip(mom1, -1.0, 1.0))
            blocks.append(np.clip(mom2, -1.0, 1.0))

            # 3. Short mean-reversion residual (rank - trailing mean of rank)
            trail = np.zeros_like(r)
            if t_count >= 4:
                # simple 3-period trailing mean
                for t in range(3, t_count):
                    trail[t] = r[t] - np.mean(r[t-3:t], axis=0)
            blocks.append(np.clip(trail, -1.0, 1.0))

            # 4. Mild non-linear transform of current rank (keeps some curvature)
            blocks.append(np.tanh(2.5 * r))

        # Limited interactions between rank-momentum of first few features
        max_f = min(feature_count, 3)
        for f1 in range(max_f):
            for f2 in range(f1 + 1, max_f):
                # interaction of 1-step momentums
                m1 = blocks[f1 * 5 + 1]
                m2 = blocks[f2 * 5 + 1]
                blocks.append(m1 * m2)

        engineered = np.stack(blocks, axis=2)
        return np.nan_to_num(engineered, nan=0.0, posinf=0.0, neginf=0.0)
