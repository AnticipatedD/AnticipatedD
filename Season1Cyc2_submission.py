# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "pyarrow",
# ]
# ///

from predictor import Predictor
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class MyPredictor(Predictor):
    """
    AlphaNova small-universe (6 features × 20 assets) predictor.
    Design goals
    ------------
    - Positive net Sharpe after 5 bp turnover cost
    - Positive IC vs the proprietary target
    - Moderate concentration (0.1–0.5)
    - Low compression loss (stable direction)
    - High city / global novelty by using non-obvious
      cross-sectional interactions + mild temporal smoothing
      instead of pure short-term return signals.
    """

    def __init__(self):
        # Keep constructor extremely light (called many times by the gate)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.smooth_span = 8          # EWMA span → turnover control
        self.ridge_alpha = 1.5        # strong regularisation
        self.min_periods = 12

    # ------------------------------------------------------------------
    # helpers (all inside the class)
    # ------------------------------------------------------------------
    def _cs_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score (safe for constant columns)."""
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0, np.nan)
        return df.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)

    def _cs_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank normalised to [-1, 1]."""
        r = df.rank(axis=1, method="average")
        n = r.count(axis=1).replace(0, np.nan)
        return (2.0 * (r - 1) / (n - 1) - 1.0).fillna(0.0)

    def _engineer(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Build a compact set of causal, cross-sectional features.
        All operations use only information ≤ t.
        """
        # features is MultiIndex columns (feature, ticker)
        # We work with a wide panel of shape (T, 6*20) internally
        # but keep the original MultiIndex for safety.

        # 1. basic cross-sectional transforms of each raw feature
        f_list = []
        for f in range(1, 7):
            col = features.xs(f"Feature.{f}", level=0, axis=1, drop_level=False)
            # raw
            f_list.append(col)
            # cs-z
            f_list.append(self._cs_zscore(col))
            # cs-rank
            f_list.append(self._cs_rank(col))

        # 2. a few pairwise interactions (products of cs-ranks)
        #    deliberately limited to keep dimensionality modest
        ranks = []
        for f in range(1, 7):
            col = features.xs(f"Feature.{f}", level=0, axis=1, drop_level=False)
            ranks.append(self._cs_rank(col))

        # selected interactions that are less “obvious”
        inter = [
            ranks[0] * ranks[1],          # 1×2
            ranks[2] * ranks[3],          # 3×4
            ranks[4] * ranks[5],          # 5×6
            ranks[0] * ranks[3],          # 1×4
            ranks[1] * ranks[5],          # 2×6
        ]
        f_list.extend(inter)

        # 3. mild temporal smoothing on the engineered panel
        #    (reduces turnover → protects net Sharpe)
        eng = pd.concat(f_list, axis=1)
        eng = eng.ewm(span=self.smooth_span, min_periods=self.min_periods).mean()

        # final cross-sectional de-mean of every column (harmless)
        eng = eng.sub(eng.mean(axis=1), axis=0)
        return eng.fillna(0.0)

    def _to_matrix(self, eng: pd.DataFrame, tickers) -> np.ndarray:
        """
        Convert engineered MultiIndex frame into a dense
        (n_samples, n_features) matrix aligned to the 20 tickers.
        """
        # eng columns are still MultiIndex; we flatten for the linear model
        # while preserving order
        return eng.values

    # ------------------------------------------------------------------
    # required interface
    # ------------------------------------------------------------------
    def train(self, features: pd.DataFrame, target: pd.DataFrame):
        """
        features : MultiIndex columns (Feature.k, ticker)
        target   : (T, 20) forward proprietary target (already xs-z & clipped)
        """
        # Engineer features (causal)
        eng = self._engineer(features)

        # Align target
        y = target.reindex(eng.index).fillna(0.0)

        # Flatten to (T, n_feat) and (T, 20)
        X = eng.values
        Y = y.values

        # Simple per-asset ridge (fast & regularised)
        # We train 20 independent models sharing the same feature matrix.
        # This is cheap and avoids a single huge multi-output model.
        self.models = []
        self.scalers = []

        for j in range(Y.shape[1]):
            sc = StandardScaler()
            Xj = sc.fit_transform(X)
            m = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            # only rows where target is not exactly zero (warm-up)
            mask = np.abs(Y[:, j]) > 1e-8
            if mask.sum() < 30:
                # fallback: zero model
                self.models.append(None)
                self.scalers.append(sc)
                continue
            m.fit(Xj[mask], Y[mask, j])
            self.models.append(m)
            self.scalers.append(sc)

        # remember column order for predict
        self.feature_names_ = eng.columns

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Return a cross-sectionally de-meaned signal of shape (T, 20).
        Must be causal and fast.
        """
        eng = self._engineer(features)
        X = eng.values

        preds = np.zeros((X.shape[0], 20))
        for j, (m, sc) in enumerate(zip(self.models, self.scalers)):
            if m is None:
                continue
            Xj = sc.transform(X)
            preds[:, j] = m.predict(Xj)

        # build DataFrame with correct ticker columns
        tickers = features.columns.get_level_values(1).unique()
        pred_df = pd.DataFrame(preds, index=features.index, columns=tickers)

        # final cross-sectional de-mean (mandatory)
        pred_df = pred_df.sub(pred_df.mean(axis=1), axis=0)

        # optional mild L1 normalisation to keep position sizes stable
        # (helps concentration & turnover)
        abs_sum = pred_df.abs().sum(axis=1).replace(0, np.nan)
        pred_df = pred_df.div(abs_sum, axis=0).fillna(0.0)

        return pred_df

# mandatory instantiation for the runner
predictor = MyPredictor()
