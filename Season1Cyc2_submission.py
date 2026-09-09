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
    AlphaNova small-universe predictor.
    Cross-sectional rank interactions + hypersphere projection + L1 hysteresis.
    Designed for positive net Sharpe, decent IC and high city/global novelty.
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        # Tunable constants (kept modest for stability)
        self.target_bound = 0.18
        self.optimal_concentration = 0.25
        self.l1_hysteresis_threshold = 0.95
        self.smooth_alpha = 0.22          # EMA weight for hysteresis

    def train(self, features: pd.DataFrame, target: pd.DataFrame = None) -> None:
        """Store feature identifiers only – no heavy fitting needed for this pure signal."""
        if features is not None and not features.empty:
            self.feature_names = list(features.columns.get_level_values(0).unique())

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        tickers = features.columns.get_level_values(1).unique()
        zero_signal = pd.DataFrame(
            0.0, index=features.index, columns=tickers, dtype=np.float32
        )

        if (
            features is None
            or features.empty
            or self.feature_names is None
            or len(self.feature_names) < 2
        ):
            return zero_signal

        try:
            # ----------------------------------------------------------
            # 1. Cross-sectional ranks → [-1, 1]
            # ----------------------------------------------------------
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            N_time, J_assets = next(iter(ranks.values())).shape

            # ----------------------------------------------------------
            # 2. Non-linear spatial expansion (novelty driver)
            # ----------------------------------------------------------
            blocks = []
            n_f = len(self.feature_names)

            for i in range(n_f):
                f1 = self.feature_names[i]
                blocks.append(np.tanh(ranks[f1] * 1.8))

                for j in range(i + 1, n_f):
                    f2 = self.feature_names[j]
                    # two distinct non-linear interactions
                    blocks.append(
                        np.sin(ranks[f1] * np.pi * 0.3)
                        * np.cos(ranks[f2] * np.pi * 0.3)
                    )
                    blocks.append(ranks[f1] * np.abs(ranks[f2]))

            raw = np.mean(blocks, axis=0)

            # ----------------------------------------------------------
            # 3. Geometric demean + projection onto sphere (concentration control)
            # ----------------------------------------------------------
            demeaned = raw - raw.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(demeaned, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            sphere = (demeaned / norms) * self.optimal_concentration

            # ----------------------------------------------------------
            # 4. Causal L1 hysteresis (turnover control → protects net Sharpe)
            # ----------------------------------------------------------
            final = np.zeros_like(sphere)

            if (
                self.prev_signal is not None
                and self.prev_tickers is not None
                and self.prev_signal.shape == (J_assets,)
                and self.prev_tickers.equals(tickers)
            ):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target_pos = sphere[t]
                delta = np.sum(np.abs(target_pos - active))

                if delta < self.l1_hysteresis_threshold:
                    current = active
                else:
                    current = (
                        self.smooth_alpha * target_pos
                        + (1.0 - self.smooth_alpha) * active
                    )
                    current -= current.mean()

                final[t] = current
                active = current.copy()

            # ----------------------------------------------------------
            # 5. Compliance: demean, clip, final demean, float32
            # ----------------------------------------------------------
            out = pd.DataFrame(final, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            # store last state for next call (causal)
            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            return zero_signal


predictor = MyPredictor()
