# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///

"""
Conservative rank-based predictor (upload as predictor.py to recover net Sharpe).

- Strong causal smoothing (EWMA span=240)
- Very large L1 hysteresis band (l1_hysteresis_threshold=3.0) to prevent churn
- No train() learning required (train is no-op)
- Returns are cross-sectionally de-meaned and clipped
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from predictor import Predictor

class MyPredictor(Predictor):
    def __init__(
        self,
        ewm_span: int = 240,
        l1_hysteresis_threshold: float = 3.0,
        tail_clip: float = 0.5,
        gross_leverage: float = 1.0,
    ):
        super().__init__()
        self.ewm_span = int(ewm_span)
        self.l1_hysteresis_threshold = float(l1_hysteresis_threshold)
        self.tail_clip = float(tail_clip)
        self.gross_leverage = float(gross_leverage)

        self.prev_signal = None
        self.prev_tickers = None
        self.feature_names = None

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # No learning for this conservative baseline; just record feature names
        if features is not None:
            if isinstance(features.columns, pd.MultiIndex):
                self.feature_names = list(features.columns.get_level_values(0).unique())
            else:
                self.feature_names = list(features.columns)

    def _cs_rank_mean(self, features: pd.DataFrame) -> pd.DataFrame:
        # Compute cross-sectional ranks for each feature and average them
        if isinstance(features.columns, pd.MultiIndex):
            features_by_name = {}
            for name in features.columns.get_level_values(0).unique():
                df = features.xs(name, level=0, axis=1).astype(float)
                # rank per-row to [-1,1]
                r = (df.rank(axis=1, pct=True, method="first") - 0.5) * 2.0
                features_by_name[name] = r
            # align ticker columns by lexicographic order
            # compute mean across features (only those present)
            stack = [features_by_name[n] for n in features_by_name]
            mean_rank = pd.concat(stack, axis=1).groupby(level=1, axis=1).mean()
        else:
            # single-level columns: rank directly
            r = (features.rank(axis=1, pct=True, method="first") - 0.5) * 2.0
            mean_rank = r
        return mean_rank

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or len(features) == 0:
            return pd.DataFrame()

        mean_rank = self._cs_rank_mean(features)
        # EWM smoothing (causal)
        smoothed = mean_rank.ewm(span=self.ewm_span, adjust=False, min_periods=1).mean()

        # Normalize to gross leverage: make L1 per row sum = gross_leverage*2? We'll scale to L1 sum = gross_leverage
        s = smoothed.sub(smoothed.mean(axis=1), axis=0)
        denom = s.abs().sum(axis=1).replace(0, np.nan)
        positions = s.div(denom, axis=0).fillna(0.0) * self.gross_leverage

        # Enforce hysteresis: only change if L1 delta > threshold
        tickers = positions.columns
        out = positions.copy().astype(np.float32)
        if (
            self.prev_signal is not None
            and self.prev_tickers is not None
            and isinstance(self.prev_tickers, pd.Index)
            and self.prev_tickers.equals(tickers)
            and self.prev_signal.shape == (len(tickers),)
        ):
            prev = pd.Series(self.prev_signal, index=tickers)
        else:
            prev = pd.Series(0.0, index=tickers)

        for idx in out.index:
            target = out.loc[idx]
            l1 = (target - prev).abs().sum()
            if l1 < self.l1_hysteresis_threshold:
                # maintain previous to avoid churn
                out.loc[idx] = prev.values
            else:
                # adopt new target but clip and de-mean
                vec = target - target.mean()
                vec = vec.clip(-self.tail_clip, self.tail_clip)
                out.loc[idx] = vec.values
                prev = out.loc[idx]
        # store last state
        self.prev_signal = prev.values.copy()
        self.prev_tickers = tickers.copy()
        # ensure cross-sectional de-mean
        out = out.sub(out.mean(axis=1), axis=0).fillna(0.0)
        return out

# required instantiation
predictor = MyPredictor()
