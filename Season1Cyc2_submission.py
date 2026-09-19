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
    Cross-sectional target-trained ridge with MAD scaling,
    rank-augmented features, and soft-clip exponential smoothing.
    Designed for higher novelty distance while remaining causal
    and ticker-agnostic.
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

        # Smoothing and position scale – deliberately different from prior
        self.alpha = 0.42          # more reactive than 0.65
        self.signal_rms = 0.11     # slightly higher gross exposure target
        self.ridge_rel = 0.35      # relative ridge strength (new formulation)
        self.clip_level = 4.5

        self.previous_signal = None
        self.previous_tickers = None

    def train(self, features, target):
        if features is None or target is None:
            raise ValueError("features and target are required")

        x_raw, feature_names, tickers = self._features_to_tensor(features)

        if x_raw.ndim != 3:
            raise ValueError("invalid feature tensor")

        t_count, asset_count, feature_count = x_raw.shape

        if t_count < 12 or asset_count < 2 or feature_count < 1:
            raise ValueError("insufficient training dimensions")

        y = self._target_to_matrix(
            target, t_count, asset_count, pd.Index(tickers)
        )

        if y.shape != (t_count, asset_count):
            raise ValueError("target alignment failed")

        x_eng = self._engineer(x_raw)

        x_flat = x_eng.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        finite = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)

        if int(np.sum(finite)) < max(30, asset_count * 6):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[finite].astype(np.float64, copy=False)
        y_fit = y_flat[finite].astype(np.float64, copy=False)

        # Robust location / scale (MAD)
        self.center = np.nanmedian(x_fit, axis=0)
        mad = np.nanmedian(np.abs(x_fit - self.center), axis=0)
        self.scale = 1.4826 * mad          # consistent with normal std

        self.center = np.nan_to_num(self.center, nan=0.0, posinf=0.0, neginf=0.0)
        self.scale = np.nan_to_num(self.scale, nan=1.0, posinf=1.0, neginf=1.0)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)

        x_fit = (x_fit - self.center) / self.scale
        x_fit = np.clip(x_fit, -self.clip_level, self.clip_level)

        # Target centering / scaling
        y_c = float(np.nanmean(y_fit))
        y_s = float(np.nanstd(y_fit))
        if not np.isfinite(y_s) or y_s < 1e-8:
            y_s = 1.0
        y_fit = (y_fit - y_c) / y_s

        # Ridge via relative penalty on Gram
        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit

        fro = np.linalg.norm(gram, ord="fro")
        if not np.isfinite(fro) or fro < 1e-12:
            fro = 1.0
        penalty = self.ridge_rel * fro / max(gram.shape[0], 1)
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

        x_eng = self._engineer(x_raw)
        x_flat = x_eng.reshape(t_count * asset_count, -1)
        x_flat = np.nan_to_num(x_flat, nan=0.0, posinf=0.0, neginf=0.0)

        x_flat = (x_flat - self.center) / self.scale
        x_flat = np.clip(x_flat, -self.clip_level, self.clip_level)

        raw = (x_flat @ self.coef).reshape(t_count, asset_count)
        raw *= self.orientation
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

        # Cross-sectional demean + soft clip
        raw = raw - np.mean(raw, axis=1, keepdims=True)
        raw = np.tanh(raw / 2.0) * 2.0          # soft non-linearity

        # Per-row RMS normalization to target
        row_rms = np.sqrt(np.mean(raw * raw, axis=1, keepdims=True))
        row_rms = np.maximum(row_rms, 1e-8)
        target = (raw / row_rms) * self.signal_rms

        # Causal exponential smoothing with state
        same_state = (
            self.previous_signal is not None
            and self.previous_tickers is not None
            and len(self.previous_signal) == asset_count
            and pd.Index(tickers).equals(self.previous_tickers)
        )

        output = np.zeros_like(target, dtype=np.float64)

        if same_state:
            active = self.previous_signal.astype(np.float64, copy=True)
        else:
            active = target[0].copy()

        for i in range(t_count):
            if i == 0 and not same_state:
                active = target[i].copy()
            else:
                active = self.alpha * target[i] + (1.0 - self.alpha) * active

            # Re-center and re-scale every step
            active = active - np.mean(active)
            rms = np.sqrt(np.mean(active * active))
            if rms > 1e-8:
                active = (active / rms) * self.signal_rms
            else:
                active[:] = 0.0

            active = active - np.mean(active)   # final demean
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
    # Helpers
    # ------------------------------------------------------------------

    def _features_to_tensor(self, features):
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if isinstance(features.columns, pd.MultiIndex):
            level_zero = list(dict.fromkeys(features.columns.get_level_values(0)))
            ticker_order = list(dict.fromkeys(features.columns.get_level_values(1)))

            if self.feature_names is None:
                selected = level_zero
            else:
                selected = [n for n in self.feature_names if n in level_zero]

            if len(selected) == 0:
                raise ValueError("no recognized feature columns")

            blocks = []
            for name in selected:
                block = features[name]
                if isinstance(block, pd.Series):
                    block = block.to_frame()
                block = block.reindex(columns=ticker_order)
                blocks.append(block.to_numpy(dtype=np.float64))

            tensor = np.stack(blocks, axis=2)
            return (
                np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0),
                selected,
                ticker_order,
            )

        # Flat fallback
        values = features.to_numpy(dtype=np.float64)
        if self.n_features is None:
            feature_count = 6
        else:
            feature_count = self.n_features

        if values.ndim != 2 or values.shape[1] % feature_count != 0:
            raise ValueError("flat features shape invalid")

        asset_count = values.shape[1] // feature_count
        tensor = values.reshape(values.shape[0], asset_count, feature_count)

        return (
            np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0),
            [f"feature_{i}" for i in range(feature_count)],
            list(range(asset_count)),
        )

    def _target_to_matrix(self, target, t_count, asset_count, tickers):
        if isinstance(target, pd.DataFrame):
            tf = target.copy()
            if len(tf) != t_count:
                raise ValueError("target row count differs from features")

            if isinstance(tf.columns, pd.MultiIndex):
                tf.columns = list(tf.columns.get_level_values(-1))

            if set(tickers).issubset(set(tf.columns)):
                aligned = tf.reindex(columns=list(tickers))
                if aligned.shape == (t_count, asset_count):
                    return aligned.to_numpy(dtype=np.float64)

            values = tf.to_numpy(dtype=np.float64)
            if values.shape == (t_count, asset_count):
                return values
            if values.shape == (t_count, 1):
                return np.repeat(values, asset_count, axis=1)

        values = np.asarray(target, dtype=np.float64)

        if values.ndim == 1:
            if values.size == t_count * asset_count:
                return values.reshape(t_count, asset_count)
            if values.size == t_count:
                return np.repeat(values.reshape(t_count, 1), asset_count, axis=1)

        if values.ndim == 2:
            if values.shape == (t_count, asset_count):
                return values
            if values.shape == (t_count, 1):
                return np.repeat(values, asset_count, axis=1)

        raise ValueError("unsupported target shape")

    def _engineer(self, x_raw):
        """
        New feature set for novelty:
        - cross-sectional z-score
        - signed rank (centered)
        - absolute deviation from cross-median (dispersion)
        - limited pairwise products of z-scores
        """
        x = np.nan_to_num(x_raw.astype(np.float64, copy=False),
                          nan=0.0, posinf=0.0, neginf=0.0)
        t, n, f = x.shape
        blocks = []

        for i in range(f):
            v = x[:, :, i]

            # Cross-sectional mean & std
            mu = np.mean(v, axis=1, keepdims=True)
            sd = np.std(v, axis=1, keepdims=True)
            sd = np.maximum(sd, 1e-8)
            z = (v - mu) / sd
            z = np.clip(z, -5.0, 5.0)

            # Centered rank
            order = np.argsort(np.argsort(v, axis=1), axis=1)
            rank = order.astype(np.float64) / max(n - 1, 1) - 0.5

            # Dispersion relative to cross-median
            med = np.median(v, axis=1, keepdims=True)
            disp = np.abs(v - med)
            disp = disp / (np.mean(disp, axis=1, keepdims=True) + 1e-8)
            disp = np.clip(disp, 0.0, 5.0)

            blocks.append(z)
            blocks.append(rank)
            blocks.append(disp)

        # Sparse pairwise interactions on z-scores only
        z_list = []
        for i in range(f):
            v = x[:, :, i]
            mu = np.mean(v, axis=1, keepdims=True)
            sd = np.maximum(np.std(v, axis=1, keepdims=True), 1e-8)
            z_list.append(np.clip((v - mu) / sd, -4.0, 4.0))

        for i in range(len(z_list)):
            for j in range(i + 1, len(z_list)):
                blocks.append(z_list[i] * z_list[j])

        eng = np.stack(blocks, axis=2)
        return np.nan_to_num(eng, nan=0.0, posinf=0.0, neginf=0.0)

    def _choose_orientation(self, x_fit, coef, y_fit, t_count, asset_count, finite):
        """Robust sign choice via average cross-sectional Spearman-like IC."""
        pred = x_fit @ coef
        pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

        y_vals = np.asarray(y_fit, dtype=np.float64)
        p_vals = np.asarray(pred, dtype=np.float64)

        usable_rows = min(y_vals.size // asset_count, t_count)
        if usable_rows < 2:
            return 1.0

        size = usable_rows * asset_count
        y_mat = y_vals[:size].reshape(usable_rows, asset_count)
        p_mat = p_vals[:size].reshape(usable_rows, asset_count)

        # Rank both sides
        def rank_rows(m):
            order = np.argsort(np.argsort(m, axis=1), axis=1)
            return order.astype(np.float64)

        ry = rank_rows(y_mat)
        rp = rank_rows(p_mat)

        ry = ry - np.mean(ry, axis=1, keepdims=True)
        rp = rp - np.mean(rp, axis=1, keepdims=True)

        num = np.sum(rp * ry, axis=1)
        den = np.sqrt(np.sum(rp * rp, axis=1) * np.sum(ry * ry, axis=1))
        valid = den > 1e-10

        if not np.any(valid):
            return 1.0

        mean_ic = float(np.nanmean(num[valid] / den[valid]))
        if not np.isfinite(mean_ic):
            return 1.0
        return 1.0 if mean_ic >= 0.0 else -1.0
