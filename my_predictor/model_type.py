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
    AlphaNova-compatible final predictor.

    Combines:
      1. Short-horizon reversal from realized returns.
      2. A heavily regularized Ridge forecast from engineered features.

    Output:
      Cross-sectional, market-neutral, rank-based predictions.
    """

    def __init__(
        self,
        ridge_alpha: float = 25.0,
        reversal_window: int = 5,
        ema_alpha: float = 0.08,
        target_scale: float = 0.20,
        validation_fraction: float = 0.25,
    ):
        self.ridge_alpha = float(ridge_alpha)
        self.reversal_window = int(reversal_window)
        self.ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self.target_scale = float(target_scale)
        self.validation_fraction = float(validation_fraction)

        self.fitted = False
        self.feature_names = None
        self.tickers = None
        self.center = None
        self.scale = None
        self.coef = None

        self.best_parms = None
        self.best_utility = -np.inf

    # ------------------------------------------------------------------
    # AlphaNova API
    # ------------------------------------------------------------------
    def set_parms(self, **parms):
        """Set model hyperparameters."""
        self.ridge_alpha = float(parms.get("ridge_alpha", self.ridge_alpha))
        self.reversal_window = int(
            parms.get("reversal_window", self.reversal_window)
        )
        self.ema_alpha = float(
            np.clip(parms.get("ema_alpha", self.ema_alpha), 0.01, 1.0)
        )
        self.target_scale = float(
            parms.get("target_scale", self.target_scale)
        )

    def optimize(self, data, target):
        """
        Select hyperparameters using a strictly time-ordered validation split.

        Parameters
        ----------
        data : dict
            Must contain:
              - 'features': MultiIndex (feature, ticker) DataFrame
              - 'returns': optional (time x ticker) returns DataFrame

        target : pd.DataFrame or np.ndarray
            Relative-return target with a matching (time x ticker) structure.

        Returns
        -------
        results : dict
            Validation results for every parameter combination.
        best_parms : dict
            Best parameter set by validation Sharpe.
        best_utility : float
            Best validation Sharpe.
        """
        if not isinstance(data, dict) or "features" not in data:
            raise ValueError("data must contain a 'features' DataFrame")

        features = data["features"]

        if features is None or target is None:
            raise ValueError("features and target are required")

        param_space = {
            "ridge_alpha": [10.0, 25.0, 50.0],
            "reversal_window": [3, 5, 10],
            "ema_alpha": [0.05, 0.08, 0.15],
        }

        results = {}
        best_utility = -np.inf
        best_parms = None

        for ridge_alpha, reversal_window, ema_alpha in product(
            param_space["ridge_alpha"],
            param_space["reversal_window"],
            param_space["ema_alpha"],
        ):
            candidate = {
                "ridge_alpha": ridge_alpha,
                "reversal_window": reversal_window,
                "ema_alpha": ema_alpha,
            }

            self.set_parms(**candidate)

            try:
                predictions = self._fit_and_predict_validation(
                    features,
                    target,
                )

                pnl = self._backtest(predictions, target)
                utility = self._utility_sharpe(pnl)

            except Exception:
                utility = -np.inf

            results[str(candidate)] = {
                "parms": candidate,
                "utility": utility,
            }

            if np.isfinite(utility) and utility > best_utility:
                best_utility = float(utility)
                best_parms = candidate

        if best_parms is None:
            raise ValueError("no valid hyperparameter combination was found")

        # Refit on all provided training data using the selected parameters.
        self.set_parms(**best_parms)
        self._fit(features, target)

        self.best_parms = best_parms
        self.best_utility = best_utility

        return results, best_parms, best_utility

    def predict(self, data):
        """
        Generate out-of-sample predictions.

        Parameters
        ----------
        data : dict
            Must contain:
              - 'features': MultiIndex (feature, ticker) DataFrame
              - 'returns': optional (time x ticker) returns DataFrame

        Returns
        -------
        pd.DataFrame
            Cross-sectional predictions.
        """
        if not self.fitted:
            raise RuntimeError(
                "predictor must be optimized or fitted before predict()"
            )

        if not isinstance(data, dict) or "features" not in data:
            raise ValueError("data must contain a 'features' DataFrame")

        features = data["features"]

        if features is None or features.empty:
            return self._zero_signal(features)

        try:
            x_raw, _, tickers = self._features_to_tensor(features)

            t_count, asset_count, _ = x_raw.shape

            # Reject a changed asset universe rather than guessing alignments.
            if asset_count != len(self.tickers) or not pd.Index(
                tickers
            ).equals(self.tickers):
                return self._zero_signal(features)

            # ---------------- Ridge component ----------------
            x_eng = self._engineer(x_raw)
            x_flat = x_eng.reshape(t_count * asset_count, -1)

            x_flat = np.nan_to_num(
                x_flat,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            x_flat = np.clip((x_flat - self.center) / self.scale, -5.0, 5.0)

            ridge_raw = (x_flat @ self.coef).reshape(t_count, asset_count)
            ridge_raw = np.nan_to_num(
                ridge_raw,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            ridge_raw = ridge_raw - ridge_raw.mean(axis=1, keepdims=True)

            ridge_signal = self._to_rank_signal(
                ridge_raw,
                index=features.index,
                tickers=pd.Index(tickers),
            )

            # ---------------- Reversal component ----------------
            returns = data.get("returns")

            if returns is not None and not returns.empty:
                returns = returns.reindex(
                    index=features.index,
                    columns=pd.Index(tickers),
                )

                recent_returns = returns.rolling(
                    self.reversal_window,
                    min_periods=max(2, self.reversal_window // 2),
                ).sum()

                reversal_raw = -recent_returns
                reversal_raw = reversal_raw.sub(
                    reversal_raw.mean(axis=1),
                    axis=0,
                )

                reversal_signal = self._to_rank_signal(
                    reversal_raw.to_numpy(),
                    index=features.index,
                    columns=pd.Index(tickers),
                )
            else:
                reversal_signal = pd.DataFrame(
                    0.0,
                    index=features.index,
                    columns=pd.Index(tickers),
                )

            # ---------------- Combine signals ----------------
            combined = 0.70 * reversal_signal + 0.30 * ridge_signal
            combined = combined.sub(combined.mean(axis=1), axis=0)

            # Mild conviction normalization; no forced RMS.
            abs_mean = combined.abs().mean(axis=1, keepdims=True)
            abs_mean = abs_mean.clip(lower=1e-8)

            signal = combined.div(abs_mean, axis=0) * self.target_scale
            signal = signal.clip(-0.75, 0.75)
            signal = signal.sub(signal.mean(axis=1), axis=0)

            # Slow causal EMA, only within this prediction window.
            signal = signal.ewm(alpha=self.ema_alpha, adjust=False).mean()
            signal = signal.sub(signal.mean(axis=1), axis=0)

            return signal.astype(np.float32)

        except Exception:
            return self._zero_signal(features)

    # ------------------------------------------------------------------
    # Training and validation
    # ------------------------------------------------------------------
    def _fit_and_predict_validation(self, features, target):
        """
        Fit on the earlier portion and predict on the later portion.
        No random shuffling; this preserves temporal order.
        """
        t_count = len(features.index)
        split = int(t_count * (1.0 - self.validation_fraction))

        if split < 10 or t_count - split < 5:
            raise ValueError("insufficient data for time-ordered validation")

        train_features = features.iloc[:split]
        valid_features = features.iloc[split:]

        train_target = target.iloc[:split]

        self._fit(train_features, train_target)

        return self.predict({"features": valid_features})

    def _fit(self, features, target):
        """Fit Ridge on aligned feature and target matrices."""
        x_raw, feature_names, tickers = self._features_to_tensor(features)

        t_count, asset_count, base_feature_count = x_raw.shape

        if t_count < 10:
            raise ValueError("insufficient training time periods")

        if asset_count < 5:
            raise ValueError("insufficient number of assets")

        if base_feature_count < 1:
            raise ValueError("no features available")

        y = self._target_to_matrix(
            target,
            t_count,
            asset_count,
            time_index=features.index,
            tickers=pd.Index(tickers),
        )

        x_eng = self._engineer(x_raw)
        x_flat = x_eng.reshape(t_count * asset_count, -1)
        y_flat = y.reshape(t_count * asset_count)

        valid = np.isfinite(y_flat) & np.all(np.isfinite(x_flat), axis=1)

        if valid.sum() < max(30, asset_count * 2):
            raise ValueError("too few finite training observations")

        x_fit = x_flat[valid].astype(np.float64)
        y_fit = y_flat[valid].astype(np.float64)

        # Robust scaling: median center and IQR scale.
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
        reg = self.ridge_alpha * np.mean(diag)

        gram = gram + np.eye(gram.shape[1]) * reg

        try:
            self.coef = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.coef = np.linalg.lstsq(gram, rhs, rcond=None)[0]

        self.feature_names = list(feature_names)
        self.tickers = pd.Index(tickers)
        self.fitted = True

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _backtest(self, predictions, relative_returns):
        """
        AlphaNova-style lead-lag backtest.

        A prediction made at time t earns the relative return from t to t+1.
        """
        predictions = predictions.copy()

        relative_returns = relative_returns.reindex(
            index=predictions.index,
            columns=predictions.columns,
        )

        pnl = predictions.shift(1).mul(relative_returns).sum(axis=1)

        if len(pnl) > 0:
            pnl.iloc[0] = 0.0

        return pnl

    @staticmethod
    def _utility_sharpe(returns):
        """Unscaled Sharpe ratio, matching AlphaNova's helper."""
        std = returns.std()

        if not np.isfinite(std) or std < 1e-12:
            return -np.inf

        return float(returns.mean() / std)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
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

    def _target_to_matrix(
        self,
        target,
        t_count,
        asset_count,
        time_index,
        tickers,
    ):
        """
        Convert target into a (time x ticker) matrix.

        Rejects ambiguous targets rather than incorrectly repeating a
        time-series target across assets.
        """
        if isinstance(target, pd.DataFrame):
            arr = (
                target.reindex(index=time_index, columns=tickers)
                .to_numpy(dtype=np.float64)
            )
        else:
            arr = np.asarray(target, dtype=np.float64)

        if arr.ndim == 2 and arr.shape == (t_count, asset_count):
            return self._clean_matrix(arr)

        if arr.ndim == 1 and arr.size == t_count * asset_count:
            return self._clean_matrix(arr.reshape(t_count, asset_count))

        raise ValueError(
            "target must be a (time x ticker) DataFrame/matrix or a "
            "time-major flattened array with matching dimensions"
        )

    @staticmethod
    def _clean_matrix(matrix):
        return np.nan_to_num(
            np.asarray(matrix, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _engineer(self, x_raw):
        """
        Small, robust feature engineering.

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

            dev = v - v.mean(axis=1, keepdims=True)

            order = np.argsort(np.argsort(v, axis=1), axis=1)
            rank = order.astype(np.float64) / max(j - 1, 1) - 0.5

            bounded = np.tanh(np.clip(v, -5.0, 5.0))

            blocks.append(dev)
            blocks.append(rank)
            blocks.append(bounded)

        return np.stack(blocks, axis=2)

    def _to_rank_signal(self, values, index, tickers):
        """Convert raw values into centered cross-sectional percentile ranks."""
        raw_df = pd.DataFrame(values, index=index, columns=tickers)

        ranks = raw_df.rank(axis=1, pct=True, na_option="keep")
        ranks = ranks.sub(0.5, axis=1)

        return ranks

    def _zero_signal(self, features):
        """Deterministic neutral output for invalid input or asset mismatch."""
        try:
            tickers = sorted(features.columns.get_level_values(1).unique())
        except Exception:
            tickers = []

        return pd.DataFrame(
            0.0,
            index=getattr(features, "index", pd.Index([])),
            columns=tickers,
            dtype=np.float32,
        )
