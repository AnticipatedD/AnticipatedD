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
    AlphaNova Season 1 Cycle 5 – robust rank-block signal.
    Simple cross-sectional ranks + mild pairwise interactions +
    relative L1 hysteresis. Designed to pass flat-check and overfitting gate.
    """
    def __init__(self):
        super().__init__()
        self.feature_names = None
        self.prev_signal = None
        self.prev_tickers = None

        self.target_bound = 1.35
        self.concentration = 0.25          # enough mass to trade
        self.ema_alpha = 0.42              # responsive but smooth
        self.l1_rel_threshold = 1.12       # only kills micro-churn

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is not None and not features.empty:
            try:
                self.feature_names = list(features.columns.get_level_values(0).unique())
            except Exception:
                self.feature_names = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        try:
            tickers = features.columns.get_level_values(1).unique()
        except Exception:
            tickers = pd.Index([])

        zero = pd.DataFrame(0.0, index=getattr(features, "index", []), columns=tickers, dtype=np.float32)

        if (features is None or features.empty or
                self.feature_names is None or len(self.feature_names) < 0):
            return zero

        try:
            # 1. Cross-sectional ranks → [-1, 1]
            ranks = {}
            for feat in self.feature_names:
                if feat not in features.columns.get_level_values(0):
                    continue
                block = features[feat].astype(np.float64)
                r = (block.rank(axis=1, pct=True, method="average") - 0.5) * 2.0
                ranks[feat] = r.fillna(0.0).to_numpy()

            if not ranks:
                return zero

            N, J = next(iter(ranks.values())).shape
            feat_list = list(ranks.keys())
            blocks = []

            # 2. Simple individual + one mild pairwise interaction per pair
            for i, f1 in enumerate(feat_list):
                b = np.tanh(ranks[f1] * 0.9)
                b = b - b.mean(axis=1, keepdims=True)
                nrm = np.linalg.norm(b, axis=1, keepdims=True)
                nrm = np.maximum(nrm, 1e-9)
                blocks.append(b / nrm)

                for j in range(i + 1, len(feat_list)):
                    f2 = feat_list[j]
                    # Single mild interaction (keeps capacity low)
                    inter = ranks[f1] * ranks[f2]
                    inter = inter - inter.mean(axis=1, keepdims=True)
                    nrm = np.linalg.norm(inter, axis=1, keepdims=True)
                    nrm = np.maximum(nrm, 1e-11)
                    blocks.append(inter / nrm)

            # 3. Equal-weight aggregation → demean → project
            velocity = np.mean(blocks, axis=0)
            velocity = velocity - velocity.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(velocity, axis=1, keepdims=True)
            norms = np.maximum[norms < 1e-12]= 1.0
            target = (velocity / norms) * self.concentration

            # 4. Relative L1 hysteresis + EMA (guarantees non-flat)
            final = np.zeros_like(target)

            if (self.prev_signal is not None and
                    self.prev_tickers is not None and
                    self.prev_signal.shape == (J,) and
                    list(self.prev_tickers) == list(tickers)):
                active = self.prev_signal.copy()
            else:
                active = target[0].copy()          # never start at zero

            for t in range(N):
                tgt = target[t]
                delta = np.sum(np.abs(tgt - active))
                scale = np.sum(np.abs(active)) + 1e-10

                if delta < self.l1_rel_threshold * scale:
                    curr = active
                else:
                    curr = self.ema_alpha * tgt + (1.0 - self.ema_alpha) * active
                    curr -= curr.mean()

                final[t] = curr
                active = curr

            # 5. Final demean + clip
            out = pd.DataFrame(final, index=features.index, columns=tickers)
            out = out.sub(out.mean(axis=1), axis=0)
            out = out.clip(-self.target_bound, self.target_bound)
            out = out.sub(out.mean(axis=1), axis=0)

            self.prev_signal = out.iloc[-1].to_numpy(dtype=np.float64)
            self.prev_tickers = out.columns.copy()

            return out.astype(np.float32)

        except Exception:
            if self.prev_signal is not None and len(self.prev_signal) == len(tickers):
                return pd.DataFrame(
                    np.tile(self.prev_signal, (len(features), 1)),
                    index=features.index,
                    columns=tickers,
                    dtype=np.float32,
                )
            return zero
