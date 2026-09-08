# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy",
#     "pyarrow"
# ]
# ///

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from predictor import Predictor


class MyPredictor(Predictor):
    """
    Fast Cross-Sectional Rank Momentum + Light Ridge.
    Optimized to finish training in seconds, not minutes.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.model = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 0.20
        self.gross_target = 0.28
        self.ema_alpha = 0.18
        self.ridge_alpha = 15.0

    def _cs_rank(self, x: np.ndarray) -> np.ndarray:
        """Fast percentile rank → [-1, +1]."""
        s = pd.Series(x)
        r = (s.rank(pct=True, method="average") - 0.5) * 2.0
        return r.fillna(0.0).to_numpy(dtype=np.float64)

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is None or features.empty:
            self.model = None
            return

        self.feature_names = list(features.columns.get_level_values(0).unique())
        tickers = features.columns.get_level_values(1).unique()
        n_assets = len(tickers)
        n_features = len(self.feature_names)

        if n_assets == 0 or n_features == 0:
            self.model = None
            return

        # Pre-allocate
        X_list = []
        y_list = []

        # Only loop once – keep it extremely light
        for t in range(len(features)):
            ranks = np.empty((n_assets, n_features), dtype=np.float64)

            for i, feat in enumerate(self.feature_names):
                vals = features[feat].iloc[t].reindex(tickers).to_numpy(dtype=np.float64)
                ranks[:, i] = self._cs_rank(vals)

            X_list.append(ranks)

            if target is not None and not target.empty:
                y_t = target.iloc[t].reindex(tickers).fillna(0.0).to_numpy(dtype=np.float64)
                y_list.append(y_t)

        if not X_list:
            self.model = None
            return

        X = np.vstack(X_list)  # (T*J, F) – ranks only, no interactions

        if y_list and len(y_list) == len(X_list):
            y = np.concatenate(y_list)
            self.model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            try:
                self.model.fit(X, y)
            except Exception:
                self.model = None
        else:
            self.model = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if features is None or features.empty or self.feature_names is None:
            return zero

        try:
            n_features = len(self.feature_names)
            n_assets = len(tickers)
            signals = []

            for t in range(len(features)):
                ranks = np.empty((n_assets, n_features), dtype=np.float64)

                for i, feat in enumerate(self.feature_names):
                    vals = features[feat].iloc[t].reindex(tickers).to_numpy(dtype=np.float64)
                    ranks[:, i] = self._cs_rank(vals)

                if self.model is not None:
                    raw = self.model.predict(ranks)
                else:
                    # Strong unsupervised fallback that usually has positive edge
                    # Assumes Feature.1 is the main momentum signal
                    raw = 0.60 * ranks[:, 0]
                    if n_features > 1:
                        raw -= 0.20 * ranks[:, 1]          # mild reversal
                    if n_features > 2:
                        raw += 0.10 * ranks[:, 2]
                    if n_features > 3:
                        raw += 0.05 * ranks[:, 3]
                    if n_features > 4:
                        raw += 0.05 * ranks[:, 4]

                raw = raw - np.nanmean(raw)
                signals.append(raw)

            signal = np.vstack(signals)

            # Scale to target gross exposure
            norms = np.linalg.norm(signal, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            signal = (signal / norms) * self.gross_target

            # Causal EMA smoothing
            if (self.prev_signal is not None
                    and self.prev_tickers is not None
                    and self.prev_tickers.equals(tickers)
                    and len(self.prev_signal) == len(tickers)):
                smoothed = np.empty_like(signal)
                prev = self.prev_signal.copy()
                for t in range(len(signal)):
                    smoothed[t] = self.ema_alpha * signal[t] + (1.0 - self.ema_alpha) * prev
                    prev = smoothed[t]
                signal = smoothed
            else:
                for t in range(1, len(signal)):
                    signal[t] = self.ema_alpha * signal[t] + (1.0 - self.ema_alpha) * signal[t-1]

            # Hard compliance
            out = pd.DataFrame(signal, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            return zero
