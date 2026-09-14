# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "scikit-learn",
#     "scipy"
# ]
# ///

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from scipy import stats

warnings.filterwarnings("ignore")

from predictor import Predictor


class MyPredictor(Predictor):
    """
    Cross-sectional rank + non-linear expansion + light supervised Ridge
    + spherical projection + L1 hysteresis.
    Designed to convert negative Sharpe into modestly positive Sharpe
    while preserving high city/global novelty.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # Tunable knobs
        self.target_bound = 0.25
        self.optimal_concentration = 0.30
        self.l1_hysteresis_threshold = 0.85   # slightly tighter than 1.12
        self.ema_alpha = 0.18                 # hybrid smoothing

        # Supervised part
        self.scaler = RobustScaler(quantile_range=(5, 95))
        self.ridge = None
        self.is_trained = False

    def train(self, features: pd.DataFrame, target) -> None:
        if features is None or len(features) < 30:
            return

        self.feature_names = list(features.columns.get_level_values(0).unique())
        X_eng = self._build_engineered(features)
        y = np.asarray(target).ravel()

        # Light supervised fit – this is what usually flips Sharpe positive
        X_norm = self.scaler.fit_transform(X_eng)
        X_norm = np.clip(X_norm, -8, 8)

        alpha = max(2.0, 6.0 / np.sqrt(X_norm.shape[1]))
        self.ridge = Ridge(alpha=alpha, fit_intercept=True)
        self.ridge.fit(X_norm, y)
        self.is_trained = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if len(features) == 0 or self.feature_names is None or len(self.feature_names) < 2:
            return zero

        try:
            # ----------------------------------------------------------
            # 1. Rank transforms (shuffling-immune)
            # ----------------------------------------------------------
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="max") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N, J = list(ranks.values())[0].shape

            # ----------------------------------------------------------
            # 2. Non-linear spatial expansion (your original idea)
            # ----------------------------------------------------------
            blocks = []
            for i, f1 in enumerate(self.feature_names):
                blocks.append(np.tanh(ranks[f1] * 2.0))
                for j in range(i + 1, len(self.feature_names)):
                    f2 = self.feature_names[j]
                    blocks.append(np.sin(ranks[f1] * np.pi * 0.25) * np.cos(ranks[f2] * np.pi * 0.25))
                    blocks.append(ranks[f1] * np.abs(ranks[f2]))

            raw = np.mean(blocks, axis=0)                     # (N, J)

            # ----------------------------------------------------------
            # 3. Optional supervised residual (the missing piece)
            # ----------------------------------------------------------
            if self.is_trained and self.ridge is not None:
                X_eng = self._build_engineered(features)
                X_norm = self.scaler.transform(X_eng)
                X_norm = np.clip(X_norm, -8, 8)
                supervised = self.ridge.predict(X_norm).reshape(N, J)
                # Blend: keep most of the geometric signal, add a little target alignment
                raw = 0.65 * raw + 0.35 * supervised

            # ----------------------------------------------------------
            # 4. Geometric demean + spherical projection
            # ----------------------------------------------------------
            raw = raw - raw.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            sphere = (raw / norms) * self.optimal_concentration

            # ----------------------------------------------------------
            # 5. L1 hysteresis + light EMA (turnover control)
            # ----------------------------------------------------------
            final = np.zeros_like(sphere)

            if (self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (J,)
                and list(self.prev_tickers) == list(tickers)):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(J, dtype=np.float64)

            for t in range(N):
                target_pos = sphere[t]
                l1_delta = np.sum(np.abs(target_pos - active))

                if l1_delta < self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    # Hybrid: hysteresis jump + EMA
                    current = self.ema_alpha * target_pos + (1.0 - self.ema_alpha) * active
                    current -= current.mean()

                final[t] = current
                active = current.copy()

            # ----------------------------------------------------------
            # 6. Final compliance
            # ----------------------------------------------------------
            df = pd.DataFrame(final, index=features.index, columns=tickers)
            df = df.sub(df.mean(axis=1), axis=0)
            df = df.clip(-self.target_bound, self.target_bound)
            df = df.sub(df.mean(axis=1), axis=0)

            # Persist state
            self.prev_signal = df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = df.columns.copy()

            return df.astype(np.float32)

        except Exception:
            return zero

    # ------------------------------------------------------------------
    def _build_engineered(self, features: pd.DataFrame) -> np.ndarray:
        """Flatten the same non-linear features used in predict for the Ridge."""
        ranks = []
        for feat in self.feature_names:
            block = features[feat].astype(np.float64)
            r = (block.rank(axis=1, pct=True, method="max") - 0.5) * 2.0
            ranks.append(r.fillna(0.0).to_numpy())

        ranks = np.stack(ranks, axis=-1)          # (T, J, F)
        T, J, F = ranks.shape

        feats = [ranks]                           # raw ranks
        feats.append(np.tanh(ranks * 2.0))

        # a few products
        for i in range(min(3, F)):
            for j in range(i + 1, min(i + 2, F)):
                feats.append((ranks[:, :, i] * ranks[:, :, j])[:, :, None])

        X = np.concatenate(feats, axis=-1)        # (T, J, n)
        return X.reshape(T, -1)
