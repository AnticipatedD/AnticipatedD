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
    Robust Cross-Sectional Rank Momentum + Light Ridge.
    Designed to be stable under changing universes and to stay under memory limits.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.model = None
        self.prev_signal = None
        self.prev_tickers = None

        # Platform-friendly constraints
        self.target_bound = 0.20
        self.gross_target = 0.30
        self.ema_alpha = 0.18
        self.ridge_lambda = 8.0          # moderate L2

    def _get_tickers(self, df: pd.DataFrame) -> pd.Index:
        """Safely extract ticker level."""
        if isinstance(df.columns, pd.MultiIndex):
            return df.columns.get_level_values(-1).unique()
        return df.columns

    def _get_feature_names(self, features: pd.DataFrame) -> list:
        """Safely extract feature names (level 0)."""
        if isinstance(features.columns, pd.MultiIndex):
            return list(features.columns.get_level_values(0).unique())
        return list(features.columns)

    def _extract_ranks(self, features: pd.DataFrame, t: int, feature_names: list, tickers: pd.Index) -> np.ndarray:
        """
        Extract cross-sectional ranks for one timestamp.
        Returns array of shape (n_tickers, n_features) with values in [-1, 1].
        Missing values become 0 after ranking.
        """
        ranks_list = []
        for feat in feature_names:
            try:
                if isinstance(features.columns, pd.MultiIndex):
                    series = features[feat].iloc[t]
                else:
                    series = features.iloc[t]

                # Align to the current ticker universe
                vals = series.reindex(tickers).astype(np.float64)
                # Rank only the non-NaN values, then map back
                ranks = vals.rank(pct=True)
                ranks = (ranks - 0.5) * 2.0          # → [-1, 1]
                ranks = ranks.fillna(0.0).values
            except Exception:
                ranks = np.zeros(len(tickers), dtype=np.float64)
            ranks_list.append(ranks)

        if not ranks_list:
            return np.zeros((len(tickers), 0), dtype=np.float64)

        return np.column_stack(ranks_list)           # (J, F)

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is None or features.empty:
            self.model = None
            return

        self.feature_names = self._get_feature_names(features)
        if not self.feature_names:
            self.model = None
            return

        tickers = self._get_tickers(features)
        n_assets = len(tickers)
        if n_assets == 0:
            self.model = None
            return

        X_list = []
        y_list = []

        for t in range(len(features)):
            ranks_mat = self._extract_ranks(features, t, self.feature_names, tickers)
            if ranks_mat.shape[1] == 0:
                continue
            X_list.append(ranks_mat)

            if target is not None and not target.empty:
                try:
                    y_t = target.iloc[t].reindex(tickers).astype(np.float64).fillna(0.0).values
                    y_list.append(y_t)
                except Exception:
                    y_list.append(np.zeros(n_assets, dtype=np.float64))

        if not X_list:
            self.model = None
            return

        X = np.vstack(X_list)                        # (T*J, F)

        if y_list and len(y_list) == len(X_list):
            y = np.concatenate(y_list)
            # Light Ridge – enough regularization to survive shuffle tests
            self.model = Ridge(alpha=self.ridge_lambda, fit_intercept=False)
            try:
                self.model.fit(X, y)
            except Exception:
                self.model = None
        else:
            self.model = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # Always return a valid zero DataFrame on any hard failure
        tickers = self._get_tickers(features) if features is not None else pd.Index([])
        zero = pd.DataFrame(
            0.0,
            index=features.index if features is not None else pd.Index([]),
            columns=tickers,
            dtype=np.float32
        )

        if (features is None or features.empty
                or self.feature_names is None
                or len(self.feature_names) == 0
                or len(tickers) == 0):
            return zero

        try:
            signals = []

            for t in range(len(features)):
                ranks_mat = self._extract_ranks(features, t, self.feature_names, tickers)

                if self.model is not None and ranks_mat.shape[1] == len(self.feature_names):
                    raw = self.model.predict(ranks_mat)
                else:
                    # Fallback: simple average of ranks (pure cross-sectional momentum)
                    raw = ranks_mat.mean(axis=1) if ranks_mat.size > 0 else np.zeros(len(tickers))

                # Center cross-sectionally
                raw = raw - np.nanmean(raw)
                signals.append(raw)

            signal = np.vstack(signals)              # (T, J)

            # Scale to target gross exposure
            norms = np.linalg.norm(signal, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            signal = (signal / norms) * self.gross_target

            # Causal EMA smoothing
            if (self.prev_signal is not None
                    and self.prev_tickers is not None
                    and len(self.prev_tickers) == len(tickers)
                    and self.prev_tickers.equals(tickers)):
                smoothed = np.zeros_like(signal)
                prev = self.prev_signal.copy()
                for t in range(len(signal)):
                    smoothed[t] = self.ema_alpha * signal[t] + (1.0 - self.ema_alpha) * prev
                    prev = smoothed[t]
                signal = smoothed
            else:
                # Mild self-smoothing on first batch
                for t in range(1, len(signal)):
                    signal[t] = self.ema_alpha * signal[t] + (1.0 - self.ema_alpha) * signal[t - 1]

            # Final compliance: demean → clip → demean
            out = pd.DataFrame(signal, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            # Store state for next call
            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            return zero
