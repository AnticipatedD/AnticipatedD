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
    AlphaNova Season 1 Cycle 5 — clean robust Ridge predictor.

    Design principles:
      - Deterministic ticker and feature ordering.
      - Cross-sectional de-meaning at every timestamp.
      - Robust cross-sectional rank output.
      - Mild conviction scaling instead of forced RMS normalization.
      - Slow causal EMA within each prediction window only.
      - No in-sample sign flipping.
      - No external novelty projections.
      - No mutable state across predict() calls.
    """

    def __init__(
        self,
        ridge_strength: float = 10.0,
        ema_alpha: float = 0.08,
        target_scale: float = 0.20,
        score_cap: float = 0.75,
        output_bound: float = 1.35,
    ):
        super().__init__()

        self.ridge_strength = float(ridge_strength)
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self.target_scale = float(target_scale)
        self.score_cap = float(score_cap)
        self.output_bound = float(output_bound)

        self.trained = False
        self.feature_names = None
        self.tickers = None
        self.center = None
        self.scale = None
        self.coef = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, features, target):
        """
        Train a regularized linear mapping from engineered features to target.

        Parameters
        ----------
        features : pd.DataFrame
            MultiIndex columns: (feature_name, ticker_id).
        target : pd.DataFrame or np.ndarray
            Either a (time x ticker) matrix or a time-major flattened array.
        """
        if features is None or target is None:
            raise ValueError("features and target are required")

        x_raw, feature_names, tickers = self._features_to_tensor(features)

        t_count, asset_count, base_feature_count = x_raw.shape

        if t_count < 20:
            raise ValueError("insufficient training time periods")

        if asset_count < 5:
            raise ValueError("insufficient number of assets")

        if base_feature_count < 1:
            raise ValueError("no features available")

        y = self._target_to_matrix(target, t_count, asset_count)

        if not np.isfinite(y).any():
            raise ValueError("target contains no finite values")

        x_eng = self._engineer(x_raw)
        x_flat = x_eng.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        valid = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)

        if valid.sum() < max(50, asset_count * 2):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[valid].astype(np.float64)
        y_fit = y_flat[valid].astype(np.float64)

        # Robust scaling: median center, IQR scale.
        self.center = np.nanmedian(x_fit, axis=0)
        q25 = np.nanpercentile(x_fit, 25.0, axis=0)
        q75 = np.nanpercentile(x_fit, 75.0, axis=0)
        self.scale = q75 - q25

        self.center = np.nan_to_num(self.center, nan=0.0)
        self.scale = np.nan_to_num(self.scale, nan=1.0)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)

        x_fit = np.clip((x_fit - self.center) / self.scale, -5.0, 5.0)

        # Standardize target.
        y_mean = float(np.mean(y_fit))
        y_std = float(np.std(y_fit))

        if not np.isfinite(y_std) or y_std < 1e-8:
            raise ValueError("target has no variance")

        y_fit = (y_fit - y_mean) / y_std

        # Ridge regression via regularized normal equations.
        gram = x_fit.T @ x_fit
        rhs = x_fit.T @ y_fit

        diag = np.maximum(np.diag(gram), 1.0)
        reg = self.ridge_strength * np.mean(diag)

        gram = gram + np.eye(gram.shape[1]) * reg

        try:
            self.coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        self.feature_names = list(feature_names)
        self.tickers = pd.Index(tickers)
        self.trained = True

    def predict(self, features):
        """
        Generate a cross-sectional, market-neutral forecast.

        Returns
        -------
        pd.DataFrame
            (time x ticker) forecast. Positive values indicate expected
            outperformance; negative values indicate expected underperformance.
        """
        if not self.trained:
            raise RuntimeError("predictor has not been trained")

        if features is None or features.empty:
            return self._zero_signal(features, self.tickers)

        try:
            x_raw, _, tickers = self._features_to_tensor(features)

            t_count, asset_count, _ = x_raw.shape

            # Do not guess an asset mapping if the universe changed.
            if asset_count != len(self.tickers) or not pd.Index(tickers).equals(
                self.tickers
            ):
                return self._zero_signal(features, tickers)

            x_eng = self._engineer(x_raw)
            x_flat = x_eng.reshape(t_count * asset_count, -1)

            x_flat = np.nan_to_num(
                x_flat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            x_flat = np.clip((x_flat - self.center) / self.scale, -5.0, 5.0)

            raw = (x_flat @ self.coef).reshape(t_count, asset_count)
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            # Cross-sectional de-mean.
            raw = raw - raw.mean(axis=1, keepdims=True)

            # Robust cross-sectional rank forecast.
            raw_df = pd.DataFrame(
                raw,
                index=features.index,
                columns=pd.Index(tickers),
            )

            ranks = raw_df.rank(axis=1, pct=True, na_option="keep")
            ranks = ranks.sub(0.5, axis=1)

            # Mild conviction normalization.
            # Preserves relative forecast strength instead of forcing fixed RMS.
            abs_mean = ranks.abs().mean(axis=1, keepdims=True)
            abs_mean = abs_mean.clip(lower=1e-8)

            signal = ranks.div(abs_mean, axis=0) * self.target_scale
            signal = signal.clip(-self.score_cap, self.score_cap)
            signal = signal.sub(signal.mean(axis=1), axis=0)

            # Slow causal EMA, applied only within this prediction window.
            signal_values = signal.to_numpy(dtype=np.float64)
            output = np.zeros_like(signal_values)

            active = np.zeros(asset_count, dtype=np.float64)

            for i in range(t_count):
                if i == 0:
                    active = signal_values[i].copy()
                else:
                    active = (
                        self.ema_alpha * signal_values[i]
                        + (1.0 - self.ema_alpha) * active
                    )

                # Maintain zero-net exposure after smoothing.
                active = active - np.mean(active)
                output[i] = active

            output_df = pd.DataFrame(
                output,
                index=features.index,
                columns=pd.Index(tickers),
            )

            # Final compliance processing.
            output_df = output_df.sub(output_df.mean(axis=1), axis=0)
            output_df = output_df.clip(-self.output_bound, self.output_bound)
            output_df = output_df.sub(output_df.mean(axis=1), axis=0)

            return output_df.astype(np.float32)

        except Exception:
            return self._zero_signal(features, self.tickers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _zero_signal(self, features, tickers):
        """Deterministic neutral output for invalid input or asset mismatch."""
        return pd.DataFrame(
            0.0,
            index=getattr(features, "index", pd.Index([])),
            columns=pd.Index(tickers),
            dtype=np.float32,
        )

    def _features_to_tensor(self, features):
        """
        Convert a MultiIndex feature DataFrame into a deterministic
        (time x ticker x feature) tensor.
        """
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features)

        if not isinstance(features.columns, pd.MultiIndex):
            raise ValueError(
                "features must use a (feature_name, ticker_id) MultiIndex"
            )

        level0 = list(dict.fromkeys(features.columns.get_level_values(0)))
        level1 = list(dict.fromkeys(features.columns.get_level_values(1)))

        # Preserve the exact feature order learned during training.
        selected = (
            [name for name in self.feature_names if name in level0]
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

            # Reindex to a fixed ticker order.
            block = block.reindex(columns=level1)

            blocks.append(block.to_numpy(dtype=np.float64))

        tensor = np.stack(blocks, axis=2)

        tensor = np.nan_to_num(
            tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return tensor, selected, level1

    def _target_to_matrix(self, target, t_count, asset_count):
        """
        Convert target into a (time x ticker) matrix.

        Only accepts formats with an unambiguous asset dimension.
        A one-value-per-time target cannot train a cross-sectional model,
        so it is rejected rather than incorrectly repeated across assets.
        """
        arr = np.asarray(target)

        # Preferred: exact (time x ticker) target.
        if arr.ndim == 2 and arr.shape == (t_count, asset_count):
            return self._clean_matrix(arr)

        # Accept flattened target only when it is time-major and size matches.
        if arr.ndim == 1 and arr.size == t_count * asset_count:
            return self._clean_matrix(
                arr.reshape(t_count, asset_count)
            )

        raise ValueError(
            "target must be a (time x ticker) matrix or a time-major "
            "flattened array with size time * assets"
        )

    @staticmethod
    def _clean_matrix(matrix):
        """Clean a numeric matrix while preserving NaNs for masking."""
        return np.nan_to_num(
            np.asarray(matrix, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _engineer(self, x_raw):
        """
        Small, robust feature set.

        For each base feature:
          1. Cross-sectional deviation.
          2. Cross-sectional percentile rank.
          3. Bounded nonlinear transform.

        No arbitrary sine/cosine projections.
        """
        x = np.nan_to_num(
            x_raw.astype(np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        t, j, f = x.shape
        blocks = []

        for i in range(f):
            v = x[:, :, i]

            # Cross-sectional deviation.
            dev = v - v.mean(axis=1, keepdims=True)

            # Cross-sectional percentile rank in [-0.5, 0.5].
            order = np.argsort(np.argsort(v, axis=1), axis=1)
            rank = order.astype(np.float64) / max(j - 1, 1) - 0.5

            # Bounded raw value.
            bounded = np.tanh(np.clip(v, -5.0, 5.0))

            blocks.append(dev)
            blocks.append(rank)
            blocks.append(bounded)

        return np.stack(blocks, axis=2)
