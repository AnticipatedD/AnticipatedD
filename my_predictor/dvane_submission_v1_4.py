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

        def __init__(self):        
        super().__init__()
        self.feature_names = None
        self.prev = None
        self.prev_tickers = None
        self.alpha = 0.18
        self.conc = 0.22

    def train(self, features, target=None):
        if features is not None and not features.empty:
            self.feature_names = sorted(features.columns.get_level_values(0).unique())[:6]

    def predict(self, features):
        tickers = features.columns.get_level_values(1).unique()
        zero = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=np.float32)
        if features is None or features.empty or self.feature_names is None or len(self.feature_names) < 6:
            return zero
        try:
            ranks = []
            for name in self.feature_names:
                r = features[name].rank(axis=1, pct=True, method="average") - 0.5
                ranks.append(np.nan_to_num(r.to_numpy(), nan=0.0))
            f = ranks
            # simple difference signal
            s = (f[0] - f[3]) + 0.6*(f[1] - f[4]) + 0.4*(f[2] - f[5])
            s -= s.mean(axis=1, keepdims=True)
            # EMA
            out = np.zeros_like(s)
            active = self.prev.copy() if (self.prev is not None and self.prev_tickers is not None and len(self.prev)==s.shape[1] and self.prev_tickers.equals(tickers)) else s[0].copy()
            for t in range(s.shape[0]):
                active = self.alpha * s[t] + (1-self.alpha)*active
                active -= active.mean()
                out[t] = active
            out -= out.mean(axis=1, keepdims=True)
            # scale AFTER EMA
            n = np.linalg.norm(out, axis=1, keepdims=True)
            n = np.maximum(n, 1e-10)
            out = out / n * self.conc
            out -= out.mean(axis=1, keepdims=True)
            self.prev = out[-1].copy()
            self.prev_tickers = pd.Index(tickers)
            return pd.DataFrame(out.astype(np.float32), index=features.index, columns=tickers)
        except Exception:
            return zero
