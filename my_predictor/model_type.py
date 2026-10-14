# /// script
# dependencies = [
#   "numpy",
#   "pandas"
# ]
# ///

import numpy as np
import pandas as pd
from predictor import Predictor


class MyPredictor(Predictor):
    """
    Sign-Decoupled Orthogonal Alpha Engine
    Preserves high City / Global Novelty while improving stability and Sharpe
    """

    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 0.18
        self.optimal_concentration = 0.26
        self.l1_hysteresis_threshold = 0.95
        self.smoothing = 0.85

    def train(self, features: pd.DataFrame, target=None) -> None:
        if features is not None and not features.empty:
            try:
                self.feature_names = list(features.columns.get_level_values(0).unique())[:6]
            except Exception:
                self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        zero = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if (features is None or features.empty or 
            self.feature_names is None or len(self.feature_names) < 6):
            return zero

        try:
            # 1. Cross-sectional ranks
            ranks = {}
            for feat in self.feature_names:
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method='average') - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            f1 = ranks[self.feature_names[0]]
            f2 = ranks[self.feature_names[1]]
            f3 = ranks[self.feature_names[2]]
            f4 = ranks[self.feature_names[3]]
            f5 = ranks[self.feature_names[4]]
            f6 = ranks[self.feature_names[5]]

            N_time, J_assets = f1.shape

            # 2. Non-linear interaction blocks (novelty source)
            blocks = []
            for name in self.feature_names:
                blocks.append(np.tanh(ranks[name] * 2.0))

            blocks.append(np.sin(f1 * np.pi * 0.25) * np.cos(f2 * np.pi * 0.25))
            blocks.append(f1 * np.abs(f2))
            blocks.append(np.sin(f3 * np.pi * 0.25) * np.cos(f4 * np.pi * 0.25))
            blocks.append(f3 * np.abs(f4))
            blocks.append(np.arctan(f5) * np.tanh(f6))
            blocks.append(f5 * f6 * np.sign(f1))

            tensor = np.stack(blocks, axis=0)  # (M, T, J)

            # 3. Orthogonal extraction via SVD (keeps novelty)
            raw_velocity = np.zeros((N_time, J_assets), dtype=np.float64)
            for t in range(N_time):
                slice_t = tensor[:, t, :]
                slice_t = slice_t - slice_t.mean(axis=1, keepdims=True)
                try:
                    _, _, vh = np.linalg.svd(slice_t, full_matrices=False)
                    vec = vh[0]
                    raw_velocity[t] = vec * np.sign(np.sum(vec) + 1e-12)
                except Exception:
                    raw_velocity[t] = slice_t.mean(axis=0)

            # 4. Demean + Spherical projection
            velocity = raw_velocity - raw_velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            sphere_target = (velocity / norms) * self.optimal_concentration

            # 5. Stronger L1 Hysteresis (Sharpe-focused)
            final_positions = np.zeros_like(sphere_target)

            if (self.prev_signal is not None and 
                self.prev_tickers is not None and 
                len(self.prev_signal) == J_assets and 
                list(self.prev_tickers) == list(tickers)):
                active = self.prev_signal.copy()
            else:
                active = np.zeros(J_assets, dtype=np.float64)

            for t in range(N_time):
                target = sphere_target[t]
                delta = np.sum(np.abs(target - active))

                if delta < self.l1_hysteresis_threshold:
                    current = active.copy()
                else:
                    current = (1.0 - self.smoothing) * target + self.smoothing * active
                    current -= current.mean()

                final_positions[t] = current
                active = current.copy()

            # 6. Final formatting
            df = pd.DataFrame(final_positions, index=features.index, columns=tickers)
            df = df.sub(df.mean(axis=1), axis=0)
            df = df.clip(-self.target_bound, self.target_bound)
            df = df.sub(df.mean(axis=1), axis=0)

            self.prev_signal = df.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = list(tickers)

            return df.astype(np.float32)

        except Exception:
            return zero


predictor = MyPredictor()
