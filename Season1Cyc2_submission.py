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
    Scientist: arifonestop_submission_v96.py
    Maximum turnover control version.
    Goal: positive net Sharpe after 5 bp cost.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # Very strong smoothing / hysteresis
        self.target_bound = 0.12
        self.optimal_concentration = 0.18
        self.l1_hysteresis_threshold = 0.55   # much tighter
        self.smooth_alpha = 0.08              # very slow adaptation

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        if features is not None and not features.empty:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)

        if (
            features is None
            or features.empty
            or self.feature_names is None
            or len(self.feature_names) < 2
        ):
            return zero

        try:
            # 1. Simple, stable cross-sectional ranks
            ranks = []
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks.append(r.fillna(0.0).to_numpy())

            raw = np.mean(ranks, axis=0)

            # light non-linearity (keeps a little edge)
            raw = np.tanh(raw * 1.5)

            N, J = raw.shape

            # 2. Sphere projection (low concentration)
            demeaned = raw - raw.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            sphere = (demeaned / norms) * self.optimal_concentration

            # 3. Extremely heavy causal hysteresis
            final = np.zeros_like(sphere)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (J,)
                and self.prev_tickers.equals(tickers)
            ):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(J, dtype=np.float64)

            for t in range(N):
                target = sphere[t]
                delta = np.sum(np.abs(target - active))

                if delta < self.l1_hysteresis_threshold:
                    current = active
                else:
                    # very slow blend
                    current = self.smooth_alpha * target + (1.0 - self.smooth_alpha) * active
                    current -= current.mean()

                final[t] = current
                active = current.copy()

            # 4. Final compliance
            out = pd.DataFrame(final, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            return zero
