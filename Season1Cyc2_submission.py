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
    Clean target-ranked cross-sectional ridge.
    Goal: positive IC / Sharpe while keeping Global Novelty ~50-55°.
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

        # Conservative, high-regularization settings
        self.ridge_strength = 8.0
        self.signal_rms = 0.095
        self.alpha = 0.42
        self.clip = 4.0

        self.prev_signal = None
        self.prev_tickers = None

    # ------------------------------------------------------------------
    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is None or target is None or features.empty:
            return

        x_raw, feature_names, tickers = self._to_tensor(features)
        t_count, asset_count, feature_count = x_raw.shape

        if t_count < 25 or asset_count < 4:
            return

        y = self._align_target(target, t_count, asset_count, tickers)
        if y is None:
            return

        # Rank the target – this is critical for IC
        y_ranked = self._cs_rank(y)

        x_eng = self._engineer(x_raw)
        x_flat = x_eng.reshape(-1, x_eng.shape[-1])
        y_flat = y_ranked.reshape(-1)

        finite = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)
        if finite.sum() < max(60, asset_count * 12):
            return

        x_fit = x_flat[finite].astype(np.float64)
        y_fit = y_flat[finite].astype(np.float64)

        # MAD scaling
        self.center = np.nanmedian(x_fit, axis=0)
        mad = np.nanmedian(np.abs(x_fit - self.center), axis=0)
        self.scale = np.maximum(1.4826 * mad, 1e-8)

        x_fit = np.clip((x_fit - self.center) / self.scale, -self.clip, self.clip)

        # Very strong ridge
        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit
        diag = np.maximum(np.diag(gram), 1.0)
        penalty = self.ridge_strength * float(np.mean(diag))
        gram = gram + np.eye(gram.shape[0]) * penalty

        try:
            coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coef = np.linalg.lstsq(gram, rhs, rcond=1e-8)[0]

        self.coef = np.nan_to_num(coef, nan=0.0)

        # Orientation on the second half only
        self.orientation = self._orientation_second_half(
            x_fit, self.coef, y_fit, asset_count
        )

        self.feature_names = list(feature_names)
        self.n_assets = asset_count
        self.n_features = feature_count
        self.trained = True
        self.prev_signal = None
        self.prev_tickers = None

    # ------------------------------------------------------------------
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = list(features.columns.get_level_values(1).unique())
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
            x_flat = np.clip((x_flat - self.center) / self.scale, -self.clip, self.clip)

            raw = (x_flat @ self.coef).reshape(t_count, asset_count)
            raw *= self.orientation
            raw = np.nan_to_num(raw, nan=0.0)

            # Strict demean + RMS
            raw -= np.mean(raw, axis=1, keepdims=True)
            rms = np.sqrt(np.mean(raw ** 2, axis=1, keepdims=True))
            target = (raw / np.maximum(rms, 1e-8)) * self.signal_rms

            # Simple causal EMA
            output = np.zeros_like(target)
            same = (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and len(self.prev_signal) == asset_count
                and list(self.prev_tickers) == list(tickers)
            )

            active = self.prev_signal.copy() if same else target[0].copy()

            for t in range(t_count):
                if t == 0 and not same:
                    active = target[t].copy()
                else:
                    active = self.alpha * target[t] + (1.0 - self.alpha) * active

                active -= np.mean(active)
                cur_rms = np.sqrt(np.mean(active ** 2))
                if cur_rms > 1e-8:
                    active = (active / cur_rms) * self.signal_rms
                else:
                    active[:] = 0.0
                active -= np.mean(active)
                output[t] = active

            self.prev_signal = output[-1].copy()
            self.prev_tickers = list(tickers)

            return pd.DataFrame(
                output.astype(np.float32),
                index=features.index,
                columns=tickers,
            )
        except Exception:
            return zero

    # ------------------------------------------------------------------
    # Minimal, high-signal feature set
    # ------------------------------------------------------------------
    def _engineer(self, x_raw: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x_raw.astype(np.float64), nan=0.0)
        t, n, f = x.shape
        blocks = []

        for i in range(f):
            v = x[:, :, i]

            # Cross-sectional rank (centered)
            order = np.argsort(np.argsort(v, axis=1), axis=1)
            rank = order.astype(np.float64) / max(n - 1, 1) - 0.5

            # Cross-sectional z-score
            mu = np.mean(v, axis=1, keepdims=True)
            sd = np.maximum(np.std(v, axis=1, keepdims=True), 1e-8)
            z = np.clip((v - mu) / sd, -3.5, 3.5)

            blocks.append(rank)
            blocks.append(z)

        return np.nan_to_num(np.stack(blocks, axis=2), nan=0.0)

    def _cs_rank(self, mat: np.ndarray) -> np.ndarray:
        """Cross-sectional rank, centered at 0."""
        t, n = mat.shape
        out = np.zeros_like(mat)
        for i in range(t):
            row = mat[i]
            finite = np.isfinite(row)
            if finite.sum() < 3:
                continue
            order = np.argsort(np.argsort(row[finite]))
            r = np.zeros(n)
            r[finite] = order.astype(np.float64) / max(finite.sum() - 1, 1) - 0.5
            out[i] = r
        return out

    def _orientation_second_half(self, x_fit, coef, y_fit, asset_count):
        pred = x_fit @ coef
        n_rows = len(y_fit) // asset_count
        if n_rows < 8:
            return 1.0

        start = (n_rows // 2) * asset_count
        y_mat = y_fit[start:].reshape(-1, asset_count)
        p_mat = pred[start:].reshape(-1, asset_count)

        # Spearman IC
        def rank_rows(m):
            return np.argsort(np.argsort(m, axis=1), axis=1).astype(np.float64)

        ry = rank_rows(y_mat)
        rp = rank_rows(p_mat)
        ry -= ry.mean(axis=1, keepdims=True)
        rp -= rp.mean(axis=1, keepdims=True)

        num = np.sum(rp * ry, axis=1)
        den = np.sqrt(np.sum(rp**2, axis=1) * np.sum(ry**2, axis=1))
        valid = den > 1e-8
        if not np.any(valid):
            return 1.0
        ic = float(np.nanmean(num[valid] / den[valid]))
        return 1.0 if (np.isfinite(ic) and ic >= 0.0) else -1.0

    # ------------------------------------------------------------------
    def _to_tensor(self, features: pd.DataFrame):
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
        return np.nan_to_num(np.stack(blocks, axis=2), nan=0.0), names, tickers

    def _align_target(self, target, t_count, asset_count, tickers):
        if isinstance(target, pd.DataFrame):
            tf = target.copy()
            if isinstance(tf.columns, pd.MultiIndex):
                tf.columns = tf.columns.get_level_values(-1)
            if set(tickers).issubset(tf.columns):
                return tf.reindex(columns=tickers).to_numpy(dtype=np.float64)
            vals = tf.to_numpy(dtype=np.float64)
            if vals.shape == (t_count, asset_count):
                return vals
        vals = np.asarray(target, dtype=np.float64)
        if vals.ndim == 2 and vals.shape == (t_count, asset_count):
            return vals
        return None
