# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "pyarrow",
#   "scikit-learn",
# ]
# ///
import numpy as np
import pandas as pd
from predictor import Predictor
from sklearn.linear_model import Ridge


class MyPredictor(Predictor):
    """
    AlphaNova Season 1 Cycle 5 — robust baseline predictor.

    Design:
      - Ridge regression learns a stable linear mapping from features to target.
      - Predictions are de-meaned cross-sectionally at every timestamp.
      - Final output is a cross-sectional percentile rank, centered at zero.
      - No external files, no hidden state, no look-ahead, no positional bugs.
    """

    def __init__(
        self,
        ridge_alpha: float = 100.0,
        smoothing_span: int = 3,
        clip_bound: float = 1.35,
    ):
        super().__init__()

        self.model = Ridge(alpha=ridge_alpha, fit_intercept=False)
        self.feature_names = None
        self.tickers = None
        self.is_fitted = False

        self.smoothing_span = max(1, int(smoothing_span))
        self.clip_bound = float(clip_bound)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_wide_matrix(
        features: pd.DataFrame,
        feature: str,
        tickers: pd.Index,
    ) -> np.ndarray:
        """
        Extract one feature as a deterministic (time x ticker) matrix.
        Reindexing prevents ticker-order mismatches between train and predict.
        """
        if feature not in features.columns.get_level_values(0):
            raise KeyError(f"Missing feature: {feature}")

        wide = features[feature].reindex(columns=tickers)
        return wide.astype(np.float64).to_numpy()

    def _zero_signal(self, features: pd.DataFrame) -> pd.DataFrame:
        """Deterministic zero output for invalid or empty input."""
        try:
            tickers = sorted(features.columns.get_level_values(1).unique())
        except Exception:
            tickers = []

        return pd.DataFrame(
            0.0,
            index=getattr(features, "index", pd.Index([])),
            columns=tickers,
            dtype=np.float32,
        )

    def _demean_cross_sectionally(self, values: np.ndarray) -> np.ndarray:
        """Subtract each timestamp's cross-sectional mean."""
        return values - np.nanmean(values, axis=1, keepdims=True)

    def _to_rank_signal(self, values: np.ndarray, index, tickers) -> pd.DataFrame:
        """
        Convert raw predictions into a centered cross-sectional percentile rank.
        Ranks are more robust than raw scores to outliers and scale changes.
        """
        raw = pd.DataFrame(values, index=index, columns=tickers)

        ranks = raw.rank(axis=1, pct=True, na_option="keep")
        ranks = ranks.sub(0.5, axis=1)

        # Mild within-window smoothing; does not rely on prior predict() calls.
        if self.smoothing_span > 1 and len(ranks) > 1:
            ranks = ranks.ewm(span=self.smoothing_span, adjust=False).mean()
            ranks = ranks.sub(ranks.mean(axis=1), axis=0)

        return ranks.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit Ridge on aligned (time x ticker) matrices.

        Important:
          - Ticker order is fixed and stored.
          - Feature order is fixed and stored.
          - The model is marked fitted only after a successful fit.
        """
        if features is None or target is None or features.empty or target.empty:
            self.is_fitted = False
            return

        try:
            self.tickers = pd.Index(
                sorted(features.columns.get_level_values(1).unique())
            )

            self.feature_names = sorted(
                features.columns.get_level_values(0).unique()
            )

            if len(self.feature_names) == 0 or len(self.tickers) == 0:
                self.is_fitted = False
                return

            feature_matrices = [
                self._extract_wide_matrix(features, feat, self.tickers)
                for feat in self.feature_names
            ]

            X = np.stack(feature_matrices, axis=-1)  # time x ticker x feature

            # Align target exactly to the same ticker order.
            y_wide = target.reindex(columns=self.tickers)
            y = y_wide.astype(np.float64).to_numpy()  # time x ticker

            # Flatten consistently: every row is one (time, ticker) observation.
            X_flat = X.reshape(-1, len(self.feature_names))
            y_flat = y.reshape(-1)

            valid = np.isfinite(X_flat).all(axis=1) & np.isfinite(y_flat)

            if valid.sum() < 10:
                self.is_fitted = False
                return

            self.model.fit(X_flat[valid], y_flat[valid])

            # Ridge.coef_ is available only after a successful fit.
            if not hasattr(self.model, "coef_"):
                self.is_fitted = False
                return

            coef = np.asarray(self.model.coef_)

            # Guard against multi-target shapes; this template expects one target.
            if coef.ndim != 1 or coef.shape[0] != len(self.feature_names):
                self.is_fitted = False
                return

            self.is_fitted = True

        except Exception:
            # Do not silently use a partially trained model.
            self.is_fitted = False

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features is None or features.empty or not self.is_fitted:
            return self._zero_signal(features)

        try:
            tickers = pd.Index(
                sorted(features.columns.get_level_values(1).unique())
            )

            if len(tickers) == 0:
                return self._zero_signal(features)

            # Use only features seen during training, with their trained order.
            available_features = set(
                features.columns.get_level_values(0).unique()
            )

            raw_preds = np.zeros(
                (len(features.index), len(tickers)),
                dtype=np.float64,
            )

            any_feature_used = False

            for feat, coef in zip(self.feature_names, self.model.coef_):
                if feat not in available_features:
                    continue

                feat_mat = self._extract_wide_matrix(features, feat, tickers)
                raw_preds += feat_mat * coef
                any_feature_used = True

            if not any_feature_used:
                return self._zero_signal(features)

            # Enforce zero-net cross-sectional exposure at every timestamp.
            raw_preds = self._demean_cross_sectionally(raw_preds)

            # Convert to robust, centered cross-sectional ranks.
            signal = self._to_rank_signal(
                raw_preds,
                index=features.index,
                tickers=tickers,
            )

            # Final compliance: de-mean, clip extremes, then de-mean again.
            signal = signal.sub(signal.mean(axis=1), axis=0)
            signal = signal.clip(-self.clip_bound, self.clip_bound)
            signal = signal.sub(signal.mean(axis=1), axis=0)

            return signal.astype(np.float32)

        except Exception:
            return self._zero_signal(features)
