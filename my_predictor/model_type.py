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
    AlphaNova Season 1 Cycle 5 — robust baseline predictor.

    Design:
      - Ridge regression learns a stable linear mapping from features to target.
      - Predictions are de-meaned cross-sectionally at every timestamp.
      - Final output is a cross-sectional percentile rank, centered at zero.
      - No external files, no hidden state, no look-ahead, no positional bugs.
    """

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
    if features is None or features.empty or not self.is_fitted:
        return self._zero_signal(features)

    try:
        tickers = pd.Index(
            sorted(features.columns.get_level_values(1).unique())
        )

        if len(tickers) < 5:
            return self._zero_signal(features)

        available_features = set(
            features.columns.get_level_values(0).unique()
        )

        # ---------- 1. Ridge residual component ----------
        ridge_preds = np.zeros(
            (len(features.index), len(tickers)), dtype=np.float64
        )

        ridge_used = False

        for feat, coef in zip(self.feature_names, self.model.coef_):
            if feat not in available_features:
                continue

            feat_mat = self._extract_wide_matrix(features, feat, tickers)
            ridge_preds += feat_mat * coef
            ridge_used = True

        ridge_preds = self._demean_cross_sectionally(ridge_preds)

        # Shrink Ridge heavily: it is currently too correlated with the pool.
        ridge_component = 0.20 * ridge_preds

        # ---------- 2. Short-horizon reversal component ----------
        # Use the first available feature as the return-like base.
        base_feature = next(
            (f for f in self.feature_names if f in available_features),
            None,
        )

        if base_feature is None:
            return self._zero_signal(features)

        base = pd.DataFrame(
            self._extract_wide_matrix(features, base_feature, tickers),
            index=features.index,
            columns=tickers,
        )

        # Recent cumulative move; negative sign = reversal.
        recent = base.rolling(5, min_periods=3).sum()
        reversal = -recent
        reversal = reversal.sub(reversal.mean(axis=1), axis=0)

        # Cross-sectional rank for robustness.
        reversal_rank = reversal.rank(axis=1, pct=True)
        reversal_rank = reversal_rank.sub(0.5, axis=1)

        # ---------- 3. Dispersion / regime gate ----------
        cross_std = base.std(axis=1, ddof=0)
        dispersion = cross_std.rolling(20, min_periods=5).mean()

        # High dispersion => more conviction in relative-value signals.
        dispersion_gate = dispersion.rank(pct=True)

        # Expand time-series gate across tickers.
        dispersion_gate_wide = pd.DataFrame(
            np.tile(dispersion_gate.to_numpy()[:, None], (1, len(tickers))),
            index=features.index,
            columns=tickers,
        )

        # Center gate around 0.5 so it does not force a permanent long bias.
        dispersion_gate_wide = dispersion_gate_wide.sub(0.5, axis=1)

        # ---------- 4. Combine components ----------
        combined = (
            0.70 * reversal_rank
            + 0.30 * dispersion_gate_wide
            + 0.10 * ridge_component
        )

        combined = combined.sub(combined.mean(axis=1), axis=0)

        # ---------- 5. Final robust ranking ----------
        signal = combined.rank(axis=1, pct=True)
        signal = signal.sub(0.5, axis=1)

        # Mild temporal smoothing, entirely within this prediction window.
        signal = signal.ewm(span=3, adjust=False).mean()
        signal = signal.sub(signal.mean(axis=1), axis=0)

        # Compliance clipping.
        signal = signal.clip(-self.clip_bound, self.clip_bound)
        signal = signal.sub(signal.mean(axis=1), axis=0)

        return signal.astype(np.float32)

    except Exception:
        return self._zero_signal(features)
