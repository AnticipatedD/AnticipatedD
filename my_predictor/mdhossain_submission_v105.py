# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///

"""
Conservative ridge-learning predictor (upload as predictor.py).

- Strong ridge (ridge_lambda=200)
- Very slow adaptation: smoothing_alpha small = 0.05 (so changes slowly)
- Large hysteresis band = 2.5
- Low concentration_radius to shrink gross exposure
- No novelty projection
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from predictor import Predictor

class MyPredictor(Predictor):
    def __init__(
        self,
        smoothing_alpha: float = 0.05,
        concentration_radius: float = 0.40,
        tail_clip: float = 0.4,
        ridge_lambda: float = 200.0,
        l1_hysteresis_threshold: float = 2.5,
        rng_seed: int = 42,
    ):
        super().__init__()
        self.smoothing_alpha = float(smoothing_alpha)
        self.concentration_radius = float(concentration_radius)
        self.tail_clip = float(tail_clip)
        self.ridge_lambda = float(ridge_lambda)
        self.l1_hysteresis_threshold = float(l1_hysteresis_threshold)
        self.rng = np.random.RandomState(int(rng_seed))

        self.feature_names = None
        self.weights = None
        self.prev_signal = None
        self.prev_tickers = None

    def _engineer_features(self, features: pd.DataFrame) -> np.ndarray:
        # same robust 3-interaction design
        if isinstance(features.columns, pd.MultiIndex):
            tickers = features.columns.get_level_values(1).unique()
            feat_names = list(features.columns.get_level_values(0).unique())
        else:
            tickers = features.columns
            feat_names = list(features.columns)
        if self.feature_names is None:
            self.feature_names = feat_names[:6]

        ranks = {}
        for feat in self.feature_names:
            if isinstance(features.columns, pd.MultiIndex) and feat in features.columns.get_level_values(0):
                df_feat = features.xs(feat, level=0, axis=1).astype(float).reindex(columns=tickers, fill_value=0.0)
            else:
                df_feat = pd.DataFrame(0.0, index=features.index, columns=tickers, dtype=float)
            r = (df_feat.rank(axis=1, pct=True, method="first") - 0.5) * 2.0
            ranks[feat] = r.fillna(0.0).to_numpy()

        base = [ranks[k] for k in ranks]
        while len(base) < 6:
            base.append(np.zeros_like(base[0]))
        f1, f2, f3, f4, f5, f6 = base[:6]
        inter_1 = f1 * np.sign(f2)
        inter_2 = f3 * np.sign(f4)
        inter_3 = np.tanh(f5 - f6)
        return np.stack([inter_1, inter_2, inter_3], axis=2).astype(float)

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        if features is None or target is None or features.empty or target.empty:
            self.weights = np.zeros(3, dtype=float)
            return
        if isinstance(features.columns, pd.MultiIndex):
            self.feature_names = list(features.columns.get_level_values(0).unique())[:6]
            tickers = features.columns.get_level_values(1).unique()
        else:
            self.feature_names = list(features.columns)[:6]
            tickers = target.columns

        features_sorted = features.sort_index()
        target_sorted = target.sort_index()
        X = self._engineer_features(features_sorted)
        N_time, J_assets, N_feats = X.shape
        target_aligned = target_sorted.reindex(columns=tickers, fill_value=0.0).astype(float)
        y = target_aligned.to_numpy().reshape(-1)
        X_flat = X.reshape(-1, N_feats)
        XTX = X_flat.T.dot(X_flat)
        XTy = X_flat.T.dot(y)
        reg = self.ridge_lambda * np.eye(N_feats)
        try:
            self.weights = np.linalg.solve(XTX + reg, XTy)
        except Exception:
            self.weights = np.linalg.pinv(XTX + reg).dot(XTy)
        if self.weights is None or len(self.weights) != N_feats:
            self.weights = np.zeros(N_feats, dtype=float)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or len(features) == 0:
            return pd.DataFrame()
        if self.feature_names is None:
            if isinstance(features.columns, pd.MultiIndex):
                self.feature_names = list(features.columns.get_level_values(0).unique())[:6]
            else:
                self.feature_names = list(features.columns)[:6]
        if self.weights is None:
            self.weights = np.zeros(3, dtype=float)

        features_sorted = features.sort_index()
        if isinstance(features_sorted.columns, pd.MultiIndex):
            tickers = features_sorted.columns.get_level_values(1).unique()
        else:
            tickers = features_sorted.columns

        X = self._engineer_features(features_sorted)
        if X.size == 0:
            return pd.DataFrame(columns=tickers, index=features_sorted.index).fillna(0.0)

        alpha_raw = np.tensordot(X, self.weights, axes=([2],[0]))
        alpha_dm = alpha_raw - alpha_raw.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(alpha_dm, axis=1, keepdims=True)
        eps = 1e-8
        sphere_velocity = (alpha_dm / (norms + eps)) * self.concentration_radius

        if (
            self.prev_signal is not None
            and self.prev_tickers is not None
            and isinstance(self.prev_tickers, pd.Index)
            and self.prev_tickers.equals(tickers)
            and self.prev_signal.shape == (len(tickers),)
        ):
            active = self.prev_signal.copy()
        else:
            active = np.zeros(len(tickers), dtype=float)

        sm = np.zeros_like(sphere_velocity)
        for t in range(sphere_velocity.shape[0]):
            target = sphere_velocity[t]
            if np.all(np.abs(target) < 1e-12):
                cur = active.copy()
            else:
                l1 = np.sum(np.abs(target - active))
                if l1 < self.l1_hysteresis_threshold:
                    cur = active.copy()
                else:
                    cur = (self.smoothing_alpha * target) + ((1.0 - self.smoothing_alpha) * active)
                    cur = cur - cur.mean()
            sm[t] = cur
            active = cur.copy()
        self.prev_signal = sm[-1].copy()
        self.prev_tickers = pd.Index(tickers)
        df = pd.DataFrame(sm, index=features_sorted.index, columns=tickers)
        df = df.sub(df.mean(axis=1), axis=0).clip(-self.tail_clip, self.tail_clip).sub(df.mean(axis=1), axis=0)
        return df.astype(float)

# required instantiation
predictor = MyPredictor()
