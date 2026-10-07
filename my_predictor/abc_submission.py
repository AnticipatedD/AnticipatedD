# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
import sys
import traceback
import numpy as np
import pandas as pd

from predictor import Predictor


class MyPredictor(Predictor):
    """
    Production-ready MyPredictor for AlphaNova portal.

    - Keep the header exactly as required by the portal (see top of file).
    - Expects predictor.Predictor to be available in the environment.
    - Robust to common input formats: MultiIndex DataFrame, flat DataFrame, or 3D ndarray (T, J, F).
    - train(features, target) fits internal state.
    - predict(features) returns np.ndarray shape (T, J) dtype float32 with per-row mean approximately zero.
    - On unexpected errors during predict, logs the traceback to stderr and returns a safe zero signal of the correct shape.
    """

    def __init__(self, dtype=np.float32):
        self.is_trained = False
        self.n_assets = None
        self.n_features = None
        self.dtype = dtype

        # Robust scaler applied to rows = samples (T*J)
        self.feature_scaler = RobustScaler(quantile_range=(5.0, 85.0))

        # Ridge regression state
        self.coefficients = None
        self.intercept = None
        self.target_mean = 0.0
        self.target_std = 1.0

        # Turnover control (EMA)
        self.alpha_smooth = 0.28

        # Numerical epsilon
        self._eps = 1e-9

    # -------------------------
    # Public API
    # -------------------------
    def train(self, features, target):
        """
        Train the predictor.

        Args:
            features: pd.DataFrame or ndarray convertible to (T, J, F)
            target: array-like (T, J) or flattened (T*J,)
        """
        try:
            self._validate_input(features, target)

            X_raw, y_arr, T, J = self._extract_tensors_and_target(features, target)

            X_eng = self._engineer_features(X_raw).astype(self.dtype, copy=False)

            X_norm = self.feature_scaler.fit_transform(X_eng)
            X_norm = np.clip(X_norm, -10.0, 10.0)

            self._fit_ridge_regression(X_norm, y_arr)

            self.n_assets = J
            self.n_features = X_eng.shape[1]
            self.is_trained = True

        except Exception as e:
            print("TRAINING ERROR: Exception during train()", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Training failed: {e}") from e

    def predict(self, features):
        """
        Predict signals for given features.

        Returns:
            np.ndarray shape (T, J) dtype float32 with cross-sectional sums near zero.
            On internal error, returns zeros of shape (T, J) and logs the exception.
        """
        try:
            if not getattr(self, "is_trained", False):
                raise RuntimeError("Model not trained. Call train() before predict().")

            X_raw, _, T, J = self._extract_tensors_and_target(features, None, allow_no_target=True)

            X_eng = self._engineer_features(X_raw).astype(self.dtype, copy=False)

            if not hasattr(self.feature_scaler, "scale_"):
                raise RuntimeError("feature_scaler not fitted. Train before predict.")

            X_norm = self.feature_scaler.transform(X_eng)
            X_norm = np.clip(X_norm, -10.0, 10.0)

            coef = np.asarray(self.coefficients, dtype=float).ravel()
            if coef.shape[0] != X_norm.shape[1]:
                raise RuntimeError(f"Coefficient dimension mismatch: {coef.shape[0]} vs {X_norm.shape[1]}")

            pred_std = X_norm.dot(coef) + float(self.intercept)  # (T*J,)
            pred_raw = pred_std * self.target_std + self.target_mean

            if pred_raw.size != T * J:
                raise RuntimeError("Prediction size mismatch")

            signal_raw = pred_raw.reshape(T, J)

            # FIRST PASS de-mean
            signal = signal_raw - signal_raw.mean(axis=1, keepdims=True)
            if np.abs(signal.mean(axis=1)).max() > 1e-9:
                signal -= signal.mean(axis=1, keepdims=True)

            # EMA smoothing
            signal_smooth = self._apply_turnover_control(signal)

            # SECOND PASS de-mean and clipping
            signal_final = np.nan_to_num(signal_smooth, nan=0.0)
            signal_final = np.clip(signal_final, -100.0, 100.0)
            signal_final -= signal_final.mean(axis=1, keepdims=True)

            if np.abs(signal_final.mean(axis=1)).max() > 1e-8:
                signal_final -= signal_final.mean(axis=1, keepdims=True)

            return signal_final.astype(self.dtype, copy=False)

        except Exception as e:
            print("PREDICTION ERROR: Exception during predict()", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            try:
                X_raw, _, T, J = self._extract_tensors_and_target(features, None, allow_no_target=True)
                zeros = np.zeros((T, J), dtype=self.dtype)
                return zeros
            except Exception:
                raise RuntimeError(f"Prediction failed and fallback zero signal could not be constructed: {e}") from e

    # -------------------------
    # Internal helpers
    # -------------------------
    def _validate_input(self, features, target):
        if features is None:
            raise ValueError("features cannot be None")
        if target is None:
            return
        tarr = np.array(target)
        if tarr.size == 0:
            raise ValueError("target is empty")
        if np.isnan(tarr).all():
            raise ValueError("target contains only NaN")

    def _extract_tensors_and_target(self, features, target, allow_no_target=False):
        """
        Convert features to X_raw shape (T, J, F) and target to flattened (T*J,).

        Returns:
            X_raw (T, J, F), y_arr (T*J,) or None, T, J
        """
        # Features -> (T, J, F)
        if isinstance(features, pd.DataFrame):
            cols = features.columns
            if isinstance(cols, pd.MultiIndex):
                feat_names = list(cols.get_level_values(0).unique())
                tickers = list(cols.get_level_values(1).unique())
                feat_names_sorted = sorted(feat_names)
                tickers_sorted = sorted(tickers)

                T = len(features)
                J = len(tickers_sorted)
                F = len(feat_names_sorted)
                X_raw = np.empty((T, J, F), dtype=float)
                for fi, feat in enumerate(feat_names_sorted):
                    cols_for_feat = [(feat, tk) for tk in tickers_sorted]
                    try:
                        block = features.loc[:, cols_for_feat].values
                    except KeyError:
                        block = np.full((T, J), np.nan)
                        for j, tk in enumerate(tickers_sorted):
                            if (feat, tk) in features.columns:
                                block[:, j] = features[(feat, tk)].values
                    X_raw[:, :, fi] = block
            else:
                arr = features.values
                T = arr.shape[0]
                # Infer J and F: prefer previously known n_assets, else default F=6
                if self.n_assets is not None and arr.shape[1] % self.n_assets == 0:
                    J = self.n_assets
                    F = arr.shape[1] // J
                else:
                    F = 6
                    if arr.shape[1] % F != 0:
                        raise ValueError(f"Flat feature columns {arr.shape[1]} not divisible by inferred F={F}")
                    J = arr.shape[1] // F
                X_raw = arr.reshape(T, J, F)
        else:
            arr = np.array(features)
            if arr.ndim == 3:
                X_raw = arr
                T, J, F = X_raw.shape
            else:
                raise ValueError("Unsupported features format. Provide DataFrame or 3D array (T, J, F).")

        # Target handling
        if target is None:
            y_arr = None
        else:
            if isinstance(target, pd.DataFrame):
                tarr = target.values
                if tarr.shape[0] != X_raw.shape[0] or tarr.shape[1] != X_raw.shape[1]:
                    raise ValueError("target DataFrame must have shape (T, J) matching features")
                y_arr = tarr.reshape(-1)
            else:
                tarr = np.array(target)
                if tarr.ndim == 1 and tarr.size == X_raw.shape[0] * X_raw.shape[1]:
                    y_arr = tarr.reshape(-1)
                elif tarr.ndim == 1 and tarr.size == X_raw.shape[0]:
                    if X_raw.shape[1] == 1:
                        y_arr = np.repeat(tarr, X_raw.shape[1])
                    else:
                        raise ValueError("target length equals T but not T*J. Provide per-asset target (T, J) or flattened (T*J,).")
                elif tarr.ndim == 2 and tarr.shape == (X_raw.shape[0], X_raw.shape[1]):
                    y_arr = tarr.reshape(-1)
                else:
                    raise ValueError("Unsupported target shape. Provide (T, J) or flattened (T*J,).")

        T, J, F = X_raw.shape
        return X_raw.astype(float, copy=False), (None if y_arr is None else y_arr.astype(float, copy=False)), T, J

    def _engineer_features(self, X_raw):
        """
        Input: X_raw (T, J, F)
        Output: X_flat (T*J, E)
        """
        T, J, F = X_raw.shape
        engineered_list = []

        # Base features: level, deviation, rank_pct-0.5
        for f in range(F):
            feat = X_raw[:, :, f]  # (T, J)
            engineered_list.append(feat)
            engineered_list.append(feat - feat.mean(axis=1, keepdims=True))

            # Per-row ranks implemented with numpy (no scipy)
            ranks = np.empty_like(feat, dtype=float)
            for t in range(T):
                row = feat[t]
                order = np.argsort(row, kind="mergesort")
                ranks_row = np.empty_like(order, dtype=float)
                ranks_row[order] = np.arange(len(order), dtype=float)
                ranks[t] = ranks_row + 1.0
            rank_pct = (ranks - 1.0) / max(1, J - 1)
            engineered_list.append(rank_pct - 0.5)

        # Interactions: limited window to control explosion
        for f1 in range(F):
            for f2 in range(f1 + 1, min(f1 + 3, F)):
                a = X_raw[:, :, f1]
                b = X_raw[:, :, f2]
                engineered_list.append(a * b)
                denom = np.abs(b) + self._eps
                engineered_list.append(a / denom)

        # Temporal features: rolling vol (window=3) and first-difference
        for f in range(F):
            feat = X_raw[:, :, f]
            vol = np.zeros_like(feat)
            for t in range(T):
                start = max(0, t - 2)
                window = feat[start:t + 1]
                if window.shape[0] > 0:
                    vol[t] = np.std(window, axis=0)
                else:
                    vol[t] = 0.0
            engineered_list.append(vol)
            diff = np.diff(feat, axis=0, prepend=0.0)
            engineered_list.append(diff)

        # Stack and flatten
        X_eng = np.stack(engineered_list, axis=2)  # (T, J, E)
        X_flat = X_eng.reshape(T * J, X_eng.shape[2])
        X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=1e3, neginf=-1e3)
        return X_flat

    def _fit_ridge_regression(self, X_norm, y_raw):
        """
        Fit ridge regression on X_norm (N, D) and y_raw (N,)
        """
        X_norm = np.asarray(X_norm, dtype=float)
        y_raw = np.asarray(y_raw, dtype=float).ravel()
        if X_norm.ndim != 2:
            raise ValueError("X_norm must be 2D")
        if y_raw.ndim != 1 or y_raw.shape[0] != X_norm.shape[0]:
            raise ValueError("y_raw must be 1D with same number of rows as X_norm")

        N, D = X_norm.shape

        # Standardize target
        self.target_mean = float(np.mean(y_raw))
        self.target_std = float(np.std(y_raw)) + self._eps
        y_std = (y_raw - self.target_mean) / self.target_std

        lambda_ridge = 35.0 / np.sqrt(max(1, D))
        ridge = Ridge(alpha=float(lambda_ridge), fit_intercept=True, max_iter=10000)
        ridge.fit(X_norm, y_std)

        coef = np.asarray(ridge.coef_, dtype=float).ravel()
        self.coefficients = coef.copy()
        self.intercept = float(ridge.intercept_)

    def _apply_turnover_control(self, signal_raw):
        """
        EMA smoothing on (T, J) signal rows and per-row re-normalization.
        """
        T, J = signal_raw.shape
        signal_smooth = np.empty_like(signal_raw, dtype=float)
        signal_smooth[0] = signal_raw[0].astype(float)

        alpha = float(self.alpha_smooth)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha_smooth must be in [0,1]")

        for t in range(1, T):
            signal_smooth[t] = alpha * signal_raw[t] + (1.0 - alpha) * signal_smooth[t - 1]

        raw_stds = np.std(signal_raw, axis=1)
        smooth_stds = np.std(signal_smooth, axis=1)
        eps = self._eps
        scale = np.ones(T, dtype=float)
        mask = smooth_stds > eps
        scale[mask] = raw_stds[mask] / smooth_stds[mask]
        signal_smooth = signal_smooth * scale[:, None]

        return signal_smooth
