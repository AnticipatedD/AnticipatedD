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
    Minimal robust baseline for AlphaNova Biweekly.
    Cross-sectional ranks + light non-linearity + EMA smoothing.
    Designed for concentration ~0.20-0.25, low turnover, non-negative Sharpe.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None
        self.ema_alpha = 0.25          # turnover control (very important)
        self.target_concentration = 0.22

    def train(self, features: pd.DataFrame, target) -> None:
        if features is not None and not features.empty:
            self.feature_names = sorted(
                list(features.columns.get_level_values(0).unique())
            )[:6]
        else:
            self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if (
            features is None
            or len(features) == 0
            or self.feature_names is None
            or len(self.feature_names) < 6
        ):
            return zero

        try:
            # 1. Cross-sectional ranks → [-0.5, 0.5]
            ranks = []
            for name in self.feature_names:
                block = features[name].astype(np.float64)
                r = block.rank(axis=1, pct=True, method="average") - 0.5
                ranks.append(np.nan_to_num(r.to_numpy(), nan=0.0))

            ranks = np.stack(ranks, axis=0)          # (6, T, J)

            # 2. Simple combination (average ranks + one light non-linear term)
            signal = ranks.mean(axis=0)              # average of 6 ranks

            # Light novelty term (helps City Novelty without destroying IC)
            f1, f2 = ranks[0], ranks[1]
            signal = signal + 0.15 * (f1 * np.abs(f2))

            # 3. Demean
            signal = signal - signal.mean(axis=1, keepdims=True)

            # 4. Scale to target concentration
            norms = np.linalg.norm(signal, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            signal = signal / norms * self.target_concentration

            # 5. Light causal EMA (critical for net Sharpe)
            T, J = signal.shape
            out = np.zeros_like(signal)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and len(self.prev_signal) == J
                and self.prev_tickers.equals(tickers)
            ):
                active = self.prev_signal.copy()
            else:
                active = signal[0].copy()

            for t in range(T):
                active = self.ema_alpha * signal[t] + (1.0 - self.ema_alpha) * active
                active = active - active.mean()
                out[t] = active

            # 6. Final safety demean
            out = out - out.mean(axis=1, keepdims=True)
            out = np.nan_to_num(out, nan=0.0)

            # Persist state
            self.prev_signal = out[-1].copy()
            self.prev_tickers = pd.Index(tickers)

            return pd.DataFrame(out.astype(np.float32), index=features.index, columns=tickers)

        except Exception:
            return zero
