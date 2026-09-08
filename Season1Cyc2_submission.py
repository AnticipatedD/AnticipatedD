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
from predictor import Predictor


class MyPredictor(Predictor):
    """
    Clean & Fast Cross-Sectional Rank Momentum + Mild Reversal.
    Optimized for AlphaNova time limits and overfitting gate.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 0.20
        self.gross_target = 0.28
        self.ema_alpha = 0.15          # light smoothing
        self.hysteresis = 0.08         # small L1 threshold

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        # Extremely light – never times out
        if features is not None and not features.empty:
            try:
                self.feature_names = list(features.columns.get_level_values(0).unique())
            except Exception:
                self.feature_names = None
        else:
            self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            return pd.DataFrame(dtype=np.float32)

        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if (features is None or features.empty
                or self.feature_names is None
                or len(self.feature_names) < 1):
            return zero

        try:
            # -------------------------------------------------------
            # 1. Fast vectorized cross-sectional ranks → [-1, +1]
            # -------------------------------------------------------
            rank_arrays = []
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                rank_arrays.append(r.fillna(0.0).to_numpy())

            ranks = np.stack(rank_arrays, axis=2)          # shape: (T, J, F)
            T, J, F = ranks.shape

            # -------------------------------------------------------
            # 2. Simple, robust signal construction
            #    Emphasize first feature (usually momentum)
            # -------------------------------------------------------
            raw = 0.60 * ranks[:, :, 0]                    # main momentum

            if F > 1:
                raw -= 0.22 * ranks[:, :, 1]               # mild reversal
            if F > 2:
                raw += 0.10 * ranks[:, :, 2]
            if F > 3:
                raw += 0.05 * ranks[:, :, 3]
            if F > 4:
                raw += 0.03 * ranks[:, :, 4]
            # ignore remaining features (they are often noisy)

            # Cross-sectional demean
            raw = raw - raw.mean(axis=1, keepdims=True)

            # -------------------------------------------------------
            # 3. Scale to target gross exposure
            # -------------------------------------------------------
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            signal = (raw / norms) * self.gross_target

            # -------------------------------------------------------
            # 4. Light causal EMA + small hysteresis
            # -------------------------------------------------------
            final = np.zeros_like(signal)

            if (self.prev_signal is not None
                    and self.prev_tickers is not None
                    and len(self.prev_tickers) == J
                    and self.prev_tickers.equals(tickers)):
                prev = self.prev_signal.copy()
            else:
                prev = np.zeros(J, dtype=np.float64)

            for t in range(T):
                target = signal[t]

                # Small hysteresis: only move if change is meaningful
                if np.sum(np.abs(target - prev)) < self.hysteresis:
                    current = prev
                else:
                    current = self.ema_alpha * target + (1.0 - self.ema_alpha) * prev
                    current = current - current.mean()     # keep demeaned

                final[t] = current
                prev = current

            # -------------------------------------------------------
            # 5. Final compliance
            # -------------------------------------------------------
            out = pd.DataFrame(final, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            # Save state
            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            return zero
