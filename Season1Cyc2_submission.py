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
    Target-trained cross-sectional ridge + controlled non-linear
    spatial residual + L1-style hysteresis filter.
    Designed for positive Sharpe while preserving novelty.
    """

    def __init__(self):
        try:
            super().__init__()
        except Exception:
            pass

        self.trained = False
        self.feature_names = None
        self.n_assets = None
        self.n_features = None

        self.center = None
        self.scale = None
        self.coef = None
        self.orientation = 1.0

        # Core signal parameters
        self.ridge_strength = 3.5
        self.signal_rms = 0.09
        self.alpha = 0.55                 # base EMA weight

        # Novelty / execution parameters (inspired by your code)
        self.novel_weight = 0.12
        self.hysteresis_threshold = 0.55  # milder than 1.12
        self.hysteresis_blend = 0.35      # how much new signal when threshold is crossed
        self.target_bound = 0.18

        self.prev_signal = None
        self.prev_tickers = None

    # ------------------------------------------------------------------
    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is None or target is None or features.empty:
            return

        x_raw, feature_names, tickers = self._to_tensor(features)
        t_count, asset_count, feature_count = x_raw.shape

        if t_count < 15 or asset_count < 3:
            return

        y = self._align_target(target, t_count, asset_count, tickers)
        if y is None:
            return

        x_eng = self._engineer(x_raw)
        x_flat = x_eng.reshape(-1, x_eng.shape[-1])
        y_flat = y.reshape(-1)

        finite = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)
        if finite.sum() < max(40, asset_count * 8):
            return

        x_fit = x_flat[finite].astype(np.float64)
        y_fit = y_flat[finite].astype(np.float64)

        # Robust scaling
        self.center = np.nanmedian(x_fit, axis=0)
        q25, q75 = np.nanpercentile(x_fit, [25, 75], axis=0)
        self.scale = np.maximum(q75 - q25, 1e-8)

        x_fit = np.clip((x_fit - self.center) / self.scale, -5.0, 5.0)

        y_c = np.nanmean(y_fit)
        y_s = np.nanstd(y_fit)
        if not np.isfinite(y_s) or y_s < 1e-8:
            y_s = 1.0
        y_fit = (y_fit - y_c) / y_s

        # Ridge
        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit
        pen = self.ridge_strength * np.mean(np.diag(gram).clip(min=1.0))
        gram = gram + np.eye(gram.shape[0]) * pen

        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(gram, rhs, rcond=1e-8)[0]

        self.coef = np.nan_to_num(coef, nan=0.0)
        self.orientation = self._sign_from_ic(x_fit, self.coef, y_fit, asset_count)

        self.feature_names = list(feature_names)
        self.n_assets = asset_count
        self.n_features = feature_count
        self.trained = True
        self.prev_signal = None
        self.prev_tickers = None

    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if not self.trained or features is None or features.empty:
            return zero

        try:
            x_raw, _, tickers = self._to_tensor(features)
            t_count, asset_count, _ = x_raw.shape
            if asset_count != self.n_assets:
                return zero

            x_eng = self._engineer(x_raw)
            x_flat = x_eng.reshape(-1, x_eng.shape[-1])
            x_flat = np.nan_to_num(x_flat, nan=0.0)
            x_flat = np.clip((x_flat - self.center) / self.scale, -5.0, 5.0)

            # Core linear prediction
            raw = (x_flat @ self.coef).reshape(t_count, asset_count)
            raw *= self.orientation
            raw = raw - np.mean(raw, axis=1, keepdims=True)

            # Mild non-linear residual (novelty driver, low weight)
            novel = self._spatial_residual(x_raw)
            novel = novel - np.mean(novel, axis=1, keepdims=True)

            # Orthogonalize residual against core
            proj = np.sum(novel * raw, axis=1, keepdims=True) / (
                np.sum(raw * raw, axis=1, keepdims=True) + 1e-8
            )
            novel = novel - proj * raw
            novel_rms = np.sqrt(np.mean(novel ** 2, axis=1, keepdims=True))
            novel = novel / np.maximum(novel_rms, 1e-8)

            combined = raw + self.novel_weight * novel
            combined = combined - np.mean(combined, axis=1, keepdims=True)

            # RMS normalization
            rms = np.sqrt(np.mean(combined ** 2, axis=1, keepdims=True))
            target = (combined / np.maximum(rms, 1e-8)) * self.signal_rms

            # L1-style hysteresis filter (keeps novelty flavour)
            output = np.zeros_like(target)
            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and len(self.prev_signal) == asset_count
                and self.prev_tickers.equals(pd.Index(tickers))
            ):
                active = self.prev_signal.copy()
            else:
                active = target[0].copy()

            for t in range(t_count):
                delta = np.sum(np.abs(target[t] - active))
                if delta < self.hysteresis_threshold:
                    current = active.copy()
                else:
                    current = (
                        self.hysteresis_blend * target[t]
                        + (1.0 - self.hysteresis_blend) * active
                    )
                    current -= current.mean()

                # final demean + soft bound
                current = current - current.mean()
                current = np.clip(current, -self.target_bound, self.target_bound)
                current = current - current.mean()

                output[t] = current
                active = current.copy()

            self.prev_signal = output[-1].copy()
            self.prev_tickers = pd.Index(tickers)

            return pd.DataFrame(
                output.astype(np.float32),
                index=features.index,
                columns=tickers,
            )

        except Exception:
            return zero

    # ------------------------------------------------------------------
    # Feature engineering (cleaner + still novel)
    # ------------------------------------------------------------------
    def _engineer(self, x_raw: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x_raw.astype(np.float64), nan=0.0)
        t, n, f = x.shape
        blocks = []

        for i in range(f):
            v = x[:, :, i]
            # cross-sectional rank
            order = np.argsort(np.argsort(v, axis=1), axis=1)
            rank = order.astype(np.float64) / max(n - 1, 1) - 0.5
            # z-score
            mu = v.mean(axis=1, keepdims=True)
            sd = np.maximum(v.std(axis=1, keepdims=True), 1e-8)
            z = np.clip((v - mu) / sd, -4.0, 4.0)
            blocks.append(rank)
            blocks.append(z)
            blocks.append(np.tanh(z))

        # limited pairwise products of ranks (novelty without explosion)
        ranks = []
        for i in range(f):
            v = x[:, :, i]
            order = np.argsort(np.argsort(v, axis=1), axis=1)
            ranks.append(order.astype(np.float64) / max(n - 1, 1) - 0.5)

        for i in range(len(ranks)):
            for j in range(i + 1, min(i + 3, len(ranks))):  # limited interactions
                blocks.append(ranks[i] * ranks[j])

        return np.nan_to_num(np.stack(blocks, axis=2), nan=0.0)

    def _spatial_residual(self, x_raw: np.ndarray) -> np.ndarray:
        """Low-weight non-linear residual that keeps the spatial flavour."""
        x = np.nan_to_num(x_raw.astype(np.float64), nan=0.0)
        t, n, f = x.shape
        comps = []

        for i in range(f):
            v = x[:, :, i]
            order = np.argsort(np.argsort(v, axis=1), axis=1)
            r = order.astype(np.float64) / max(n - 1, 1) - 0.5
            comps.append(np.sin(r * np.pi * 0.4))
            comps.append(np.cos(r * np.pi * 0.3))

        residual = np.mean(np.stack(comps, axis=2), axis=2)
        return residual - residual.mean(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, features: pd.DataFrame):
        if not isinstance(features.columns, pd.MultiIndex):
            raise ValueError("expected MultiIndex columns")
        names = list(dict.fromkeys(features.columns.get_level_values(0)))
        tickers = list(dict.fromkeys(features.columns.get_level_values(1)))
        if self.feature_names is not None:
            names = [n for n in self.feature_names if n in names]
        blocks = []
        for n in names:
            b = features[n]
            if isinstance(b, pd.Series):
                b = b.to_frame()
            b = b.reindex(columns=tickers)
            blocks.append(b.to_numpy(dtype=np.float64))
        tensor = np.stack(blocks, axis=2)
        return np.nan_to_num(tensor, nan=0.0), names, tickers

    def _align_target(self, target, t_count, asset_count, tickers):
        if isinstance(target, pd.DataFrame):
            if isinstance(target.columns, pd.MultiIndex):
                target = target.copy()
                target.columns = target.columns.get_level_values(-1)
            if set(tickers).issubset(target.columns):
                return target.reindex(columns=tickers).to_numpy(dtype=np.float64)
            vals = target.to_numpy(dtype=np.float64)
            if vals.shape == (t_count, asset_count):
                return vals
        vals = np.asarray(target, dtype=np.float64)
        if vals.ndim == 2 and vals.shape == (t_count, asset_count):
            return vals
        return None

    def _sign_from_ic(self, x_fit, coef, y_fit, asset_count):
        pred = x_fit @ coef
        n_rows = min(len(y_fit) // asset_count, 500)
        if n_rows < 3:
            return 1.0
        y_mat = y_fit[: n_rows * asset_count].reshape(n_rows, asset_count)
        p_mat = pred[: n_rows * asset_count].reshape(n_rows, asset_count)
        y_mat -= y_mat.mean(axis=1, keepdims=True)
        p_mat -= p_mat.mean(axis=1, keepdims=True)
        num = np.sum(p_mat * y_mat, axis=1)
        den = np.sqrt(np.sum(p_mat**2, axis=1) * np.sum(y_mat**2, axis=1))
        valid = den > 1e-8
        if not np.any(valid):
            return 1.0
        ic = np.nanmean(num[valid] / den[valid])
        return 1.0 if (np.isfinite(ic) and ic >= 0) else -1.0
