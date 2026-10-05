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
    Cross-Sectional Ridge System with Orthogonal Spatial Novelty Projections.
    IC-aware sign correction, robust scaling, double demean, light EMA.
    Designed for positive Sharpe and City Novelty > 60° on AlphaNova residuals.
    """

    def __init__(self):
        super().__init__()
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

        # Tuned hyper-parameters
        self.alpha = 0.25               # EMA responsiveness
        self.signal_rms = 0.12
        self.ridge_strength = 12.0      # slightly lower for better recovery
        self.previous_signal = None
        self.previous_tickers = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, features, target):
        if features is None or target is None:
            raise ValueError("features and target are required")

        x_raw, names, tickers = self._features_to_tensor(features)
        if x_raw.ndim != 3:
            raise ValueError("features must be convertible to T × J × F")

        t_count, asset_count, feature_count = x_raw.shape
        if t_count < 20 or asset_count < 2 or feature_count < 1:
            raise ValueError("insufficient training dimensions")

        y = self._target_to_matrix(target, t_count, asset_count)

        x_eng = self._engineer(x_raw)
        x_flat = x_eng.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        valid = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)
        if np.sum(valid) < max(30, asset_count * 4):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[valid].astype(np.float64)
        y_fit = y_flat[valid].astype(np.float64)

        # Robust location / scale (median + IQR)
        self.center = np.nanmedian(x_fit, axis=0)
        q25 = np.nanpercentile(x_fit, 25.0, axis=0)
        q75 = np.nanpercentile(x_fit, 75.0, axis=0)
        self.scale = q75 - q25

        self.center = np.nan_to_num(self.center, nan=0.0)
        self.scale = np.nan_to_num(self.scale, nan=1.0)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)

        x_fit = np.clip((x_fit - self.center) / self.scale, -8.0, 8.0)

        # Standardise target
        y_mean = float(np.mean(y_fit))
        y_std = float(np.std(y_fit))
        if not np.isfinite(y_std) or y_std < 1e-8:
            y_std = 1.0
        y_fit = (y_fit - y_mean) / y_std

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

        # ---------- IC-aware sign correction (mathematical fix) ----------
        raw_pred = x_fit @ self.coef
        ic = self._safe_ic(raw_pred, y_fit)
        if ic < 0.0:
            self.coef = -self.coef          # flip only when training IC is negative
        # -----------------------------------------------------------------

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
        x_flat = np.clip((x_flat - self.center) / self.scale, -8.0, 8.0)

        raw = (x_flat @ self.coef + self.intercept).reshape(t_count, asset_count)
        raw = np.nan_to_num(raw, nan=0.0)

        # Primary demean
        raw = raw - np.mean(raw, axis=1, keepdims=True)

        # RMS normalisation to target risk level
        row_rms = np.sqrt(np.mean(raw * raw, axis=1, keepdims=True))
        row_rms = np.maximum(row_rms, 1e-8)
        target = raw / row_rms * self.signal_rms

        # Causal EMA with demean after every step
        use_prev = (
            self.previous_signal is not None
            and self.previous_tickers is not None
            and len(self.previous_signal) == asset_count
            and pd.Index(tickers).equals(self.previous_tickers)
        )

        output = np.zeros_like(target)
        active = (
            self.previous_signal.astype(np.float64)
            if use_prev
            else np.zeros(asset_count, dtype=np.float64)
        )

        for i in range(t_count):
            if i == 0 and not use_prev:
                active = target[i].copy()
            else:
                active = self.alpha * target[i] + (1.0 - self.alpha) * active

            active -= np.mean(active)
            rms = np.sqrt(np.mean(active * active))
            if rms > 1e-8:
                active = active / rms * self.signal_rms
            else:
                active[:] = 0.0
            active -= np.mean(active)
            output[i] = active

        # Mandatory final demean (platform safety)
        output = output - np.mean(output, axis=1, keepdims=True)
        output = np.nan_to_num(output, nan=0.0)

        self.previous_signal = output[-1].copy()
        self.previous_tickers = pd.Index(tickers)

        return pd.DataFrame(
            output.astype(np.float32),
            index=features.index,
            columns=pd.Index(tickers),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_ic(x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 5:
            return 0.0
        x = x[mask] - np.mean(x[mask])
        y = y[mask] - np.mean(y[mask])
        sx = np.sqrt(np.sum(x * x))
        sy = np.sqrt(np.sum(y * y))
        if sx < 1e-12 or sy < 1e-12:
            return 0.0
        return float(np.sum(x * y) / (sx * sy))

    def _features_to_tensor(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            level0 = list(dict.fromkeys(features.columns.get_level_values(0)))
            level1 = list(dict.fromkeys(features.columns.get_level_values(1)))

            selected = (
                [n for n in self.feature_names if n in level0]
                if self.feature_names is not None
                else level0
            )
            if not selected:
                raise ValueError("no recognized feature columns")

            blocks = []
            for name in selected:
                block = features[name]
                if isinstance(block, pd.Series):
                    block = block.to_frame()
                block = block.reindex(columns=level1)
                blocks.append(block.to_numpy(dtype=np.float64))

            tensor = np.stack(blocks, axis=2)
            return np.nan_to_num(tensor), selected, level1

        # Flat fallback
        values = features.to_numpy(dtype=np.float64)
        f_count = self.n_base_features or 6
        if values.shape[1] % f_count != 0:
            f_count = 1
        j = values.shape[1] // f_count
        tensor = values.reshape(values.shape[0], j, f_count)
        return (
            np.nan_to_num(tensor),
            [f"f{i}" for i in range(f_count)],
            list(range(j)),
        )

    def _target_to_matrix(self, target, t_count, asset_count):
        """Expand target to (T, J) only when it is truly per-asset."""
        arr = np.asarray(target, dtype=np.float64)

        if arr.ndim == 2 and arr.shape == (t_count, asset_count):
            return np.nan_to_num(arr)

        if arr.ndim == 1:
            if arr.size == t_count * asset_count:
                return np.nan_to_num(arr.reshape(t_count, asset_count))
            if arr.size == t_count:
                # Official scalar-per-timestamp case → expand by repeat
                return np.nan_to_num(np.repeat(arr.reshape(t_count, 1), asset_count, axis=1))

        if arr.ndim == 2 and arr.shape[1] == 1 and arr.shape[0] == t_count:
            return np.nan_to_num(np.repeat(arr, asset_count, axis=1))

        # Safe zero fallback
        return np.zeros((t_count, asset_count), dtype=np.float64)

    def _engineer(self, x_raw):
        x = np.nan_to_num(x_raw.astype(np.float64), nan=0.0)
        t, j, f = x.shape
        blocks = []

        for i in range(f):
            v = x[:, :, i]
            dev = v - np.mean(v, axis=1, keepdims=True)
            # Rank in [-0.5, 0.5]
            order = np.argsort(np.argsort(v, axis=1), axis=1)
            rank = order.astype(np.float64) / max(j - 1, 1) - 0.5

            blocks.append(np.tanh(np.clip(v, -8.0, 8.0)))
            blocks.append(np.tanh(np.clip(dev, -8.0, 8.0)))
            blocks.append(rank)

            # Orthogonal trigonometric novelty projections
            blocks.append(np.sin(v * np.pi / 4.0))
            blocks.append(np.cos(dev * np.pi / 4.0))

        # Selected pairwise interactions (keep sparsity)
        for i in range(f):
            for k in range(i + 1, f):
                a = np.tanh(np.clip(x[:, :, i], -6.0, 6.0))
                b = np.tanh(np.clip(x[:, :, k], -6.0, 6.0))
                blocks.append(a * b)
                blocks.append(np.sign(a - b) * np.sqrt(np.abs(a * b) + 1e-8))

        eng = np.stack(blocks, axis=2)
        return np.nan_to_num(eng, nan=0.0)
