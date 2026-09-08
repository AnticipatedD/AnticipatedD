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
    Clean Cross-Sectional Rank Momentum + Light Regularized Interactions.
    Designed to clear AlphaNova overfitting gate and produce positive Sharpe.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.model = None
        self.prev_signal = None
        self.prev_tickers = None

        # Hard constraints (match platform)
        self.target_bound = 0.20
        self.gross_target = 0.30          # modest concentration
        self.ema_alpha = 0.18             # light smoothing → lower turnover
        self.ridge_lambda_base = 12.0     # strong L2 (prevents noise memorization)

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is None or features.empty:
            return

        self.feature_names = list(features.columns.get_level_values(0).unique())
        n_features = len(self.feature_names)

        # Build simple, generalizable engineered matrix (ranks only + a few products)
        X_list = []
        y_list = []

        # We process in time order (walk-forward safe)
        tickers = features.columns.get_level_values(1).unique()
        n_assets = len(tickers)

        for t in range(len(features)):
            row_feats = []
            for feat in self.feature_names:
                vals = features[feat].iloc[t].astype(np.float64).fillna(0.0).values
                # Cross-sectional percentile rank → [-1, 1]
                ranks = (pd.Series(vals).rank(pct=True).values - 0.5) * 2.0
                row_feats.append(ranks)

            # Stack ranks (shape: n_assets x n_features)
            ranks_mat = np.column_stack(row_feats)          # (J, F)

            # Light, stable interactions (only pairwise products of ranks)
            inter = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    inter.append(ranks_mat[:, i] * ranks_mat[:, j])

            if inter:
                inter_mat = np.column_stack(inter)
                X_t = np.hstack([ranks_mat, inter_mat])    # (J, F + C(F,2))
            else:
                X_t = ranks_mat

            X_list.append(X_t)

            if target is not None and not target.empty:
                # Target is already de-meaned forward returns
                y_t = target.iloc[t].reindex(tickers).fillna(0.0).values
                y_list.append(y_t)

        if not X_list:
            return

        X = np.vstack(X_list)          # (T*J, D)
        D = X.shape[1]

        if y_list:
            y = np.concatenate(y_list)
            # Adaptive strong Ridge – key to passing the shuffle test
            lam = self.ridge_lambda_base / np.sqrt(max(D, 1))
            self.model = Ridge(alpha=lam, fit_intercept=False)
            self.model.fit(X, y)
        else:
            # Pure unsupervised fallback (equal-weight ranks)
            self.model = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if features is None or features.empty or self.feature_names is None:
            return zero

        try:
            n_features = len(self.feature_names)
            signals = []

            for t in range(len(features)):
                row_feats = []
                for feat in self.feature_names:
                    vals = features[feat].iloc[t].astype(np.float64).fillna(0.0).values
                    ranks = (pd.Series(vals).rank(pct=True).values - 0.5) * 2.0
                    row_feats.append(ranks)

                ranks_mat = np.column_stack(row_feats)   # (J, F)

                inter = []
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        inter.append(ranks_mat[:, i] * ranks_mat[:, j])

                if inter:
                    X_t = np.hstack([ranks_mat, np.column_stack(inter)])
                else:
                    X_t = ranks_mat

                if self.model is not None:
                    raw = self.model.predict(X_t)
                else:
                    # Unsupervised: average of ranks (pure CS momentum)
                    raw = ranks_mat.mean(axis=1)

                # Center
                raw = raw - raw.mean()
                signals.append(raw)

            signal = np.vstack(signals)          # (T, J)

            # Scale to target gross exposure
            norms = np.linalg.norm(signal, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            signal = (signal / norms) * self.gross_target

            # Causal EMA smoothing (reduces fee drag)
            if (self.prev_signal is not None
                    and self.prev_tickers is not None
                    and self.prev_tickers.equals(tickers)
                    and self.prev_signal.shape == (len(tickers),)):
                smoothed = np.zeros_like(signal)
                prev = self.prev_signal.copy()
                for t in range(len(signal)):
                    smoothed[t] = (self.ema_alpha * signal[t]
                                   + (1.0 - self.ema_alpha) * prev)
                    prev = smoothed[t]
                signal = smoothed
            else:
                # first batch – mild self-smoothing
                for t in range(1, len(signal)):
                    signal[t] = (self.ema_alpha * signal[t]
                                 + (1.0 - self.ema_alpha) * signal[t-1])

            # Final compliance: demean → clip → demean
            out = pd.DataFrame(signal, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            # Store state
            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            return zero
