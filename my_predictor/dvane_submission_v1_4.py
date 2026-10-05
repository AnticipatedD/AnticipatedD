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
    Rank-Difference baseline with correct concentration control.
    Scaling is applied AFTER the EMA so concentration stays in the healthy band.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.ema_alpha = 0.22
        self.target_concentration = 0.23   # healthy target

    def train(self, features: pd.DataFrame, target=None) -> None:
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
            # 1. Cross-sectional ranks → [-1, 1]
            ranks = []
            for name in self.feature_names:
                block = features[name].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks.append(np.nan_to_num(r.to_numpy(), nan=0.0))
            f = ranks  # list of (T, J)

            T, J = f[0].shape

            # 2. Asymmetric combination (not plain average → better novelty)
            signal = (
                1.00 * (f[0] - f[3]) +
                0.75 * (f[1] - f[4]) +
                0.50 * (f[2] - f[5]) +
                0.35 * (f[0] * np.abs(f[5])) +
                0.25 * (f[1] * f[2])
            )

            # 3. Demean
            signal = signal - signal.mean(axis=1, keepdims=True)

            # 4. Light causal EMA
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

            # 5. Final demean
            out = out - out.mean(axis=1, keepdims=True)

            # 6. CRITICAL: Scale to target concentration AFTER EMA
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            out = out / norms * self.target_concentration

            # 7. One last demean (platform safety)
            out = out - out.mean(axis=1, keepdims=True)
            out = np.nan_to_num(out, nan=0.0)

            self.prev_signal = out[-1].copy()
            self.prev_tickers = pd.Index(tickers)

            return pd.DataFrame(out.astype(np.float32), index=features.index, columns=tickers)

        except Exception:
            return zero
