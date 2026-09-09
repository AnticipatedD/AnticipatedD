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
    Higher-novelty variant.
    Stronger non-linear interactions + mild temporal differentiation
    to push City / Global Novelty up while preserving the positive IC.
    Scientist: arifonestop_submission_v95.py
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 0.16
        self.optimal_concentration = 0.22
        self.l1_hysteresis_threshold = 0.85
        self.smooth_alpha = 0.18

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
            # 1. Cross-sectional ranks
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N, J = next(iter(ranks.values())).shape
            n_f = len(self.feature_names)

            # 2. Higher-order / phase-shifted interactions (novelty engine)
            blocks = []
            for i in range(n_f):
                f1 = ranks[self.feature_names[i]]
                blocks.append(np.tanh(f1 * 2.2))
                blocks.append(np.sin(f1 * np.pi * 0.55))          # extra phase

                for j in range(i + 1, n_f):
                    f2 = ranks[self.feature_names[j]]
                    # three distinct non-linear mixes
                    blocks.append(np.sin(f1 * np.pi * 0.4) * np.cos(f2 * np.pi * 0.4))
                    blocks.append(f1 * np.sign(f2) * np.abs(f2) ** 0.7)
                    blocks.append(np.tanh(f1 - f2) * np.cos(f1 + f2))

            raw = np.mean(blocks, axis=0)

            # mild temporal difference (adds trajectory diversity → Global Novelty)
            diff = np.zeros_like(raw)
            diff[1:] = raw[1:] - raw[:-1]
            raw = raw + 0.12 * diff

            # 3. Sphere projection
            demeaned = raw - raw.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            sphere = (demeaned / norms) * self.optimal_concentration

            # 4. L1 hysteresis (turnover control)
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
                    current = self.smooth_alpha * target + (1.0 - self.smooth_alpha) * active
                    current -= current.mean()
                final[t] = current
                active = current.copy()

            # 5. Final compliance
            out = pd.DataFrame(final, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            return zero
