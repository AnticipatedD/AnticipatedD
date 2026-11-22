# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///
import warnings
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class DanteVane_S2_V13Predictor(Predictor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.models = []

        # 1. XGBoost Ensemble Hyperparameters
        self.seeds = self.config.get("seeds", (42, 1024, 2026))
        self.params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": self.config.get("huber_slope", 1.0),
            "max_depth": self.config.get("max_depth", 4),
            "eta": self.config.get("eta", 0.03),
            "subsample": self.config.get("subsample", 0.8),
            "colsample_bytree": self.config.get("colsample_bytree", 0.8),
            "min_child_weight": self.config.get("min_child_weight", 50),
            "reg_lambda": self.config.get("reg_lambda", 5.0),
            "reg_alpha": self.config.get("reg_alpha", 1.0),
            "verbosity": 0,
            "nthread": -1,
        }
        self.n_rounds = self.config.get("n_rounds", 300)

        # 2. Signal Processing & Smoothing Windows
        self.fast_span = self.config.get("fast_span", 5)
        self.slow_span = self.config.get("slow_span", 20)
        self.tail_power = self.config.get("tail_power", 1.25)

        # 3. Price-Based Momentum / Reversal Hyperparameters (CPCV tuned)
        self.mom_short_window = self.config.get("mom_short_window", 5)
        self.mom_long_window = self.config.get("mom_long_window", 21)
        self.vol_window = self.config.get("vol_window", 21)
        self.reversal_window = self.config.get("reversal_window", 3)
        self.l2_penalty = self.config.get("l2_penalty", 0.01)

    # -------------------------------------------------------------------------
    # Cross-Sectional Math Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional rank normalized to [-1, 1]."""
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _orthogonalize(target_df: pd.DataFrame, basis_df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectionally orthogonalizes target_df against basis_df."""
        num = (target_df * basis_df).sum(axis=1)
        den = (basis_df * basis_df).sum(axis=1).replace(0.0, np.nan)
        beta = (num / den).fillna(0.0)
        return target_df.sub(basis_df.mul(beta, axis=0), axis=0)

    def cross_sectional_rank_normalize(self, raw_scores: pd.Series) -> pd.Series:
        """
        Normalizes scores across the asset universe to guarantee dollar neutrality:
        - Outlier clipping (Winsorizing at 5th/95th percentiles)
        - Mean-zero centering: sum(weights) == 0
        - Long book == +1.0, Short book == -1.0, Gross Leverage == 2.0
        """
        valid_scores = raw_scores.dropna()
        if len(valid_scores) < 2:
            return raw_scores.fillna(0.0)

        # Winsorize extreme outliers
        q_low, q_high = valid_scores.quantile(0.05), valid_scores.quantile(0.95)
        clipped = valid_scores.clip(lower=q_low, upper=q_high)

        # Rank-based uniform centering
        ranks = clipped.rank(ascending=True, method="average")
        centered = ranks - ranks.mean()

        pos_sum = centered[centered > 0].sum()
        neg_sum = abs(centered[centered < 0].sum())

        normalized = centered.copy()
        if pos_sum > 0:
            normalized[normalized > 0] = (centered[centered > 0] / pos_sum) * 1.0
        if neg_sum > 0:
            normalized[normalized < 0] = (centered[centered < 0] / neg_sum) * -1.0

        return normalized.reindex(raw_scores.index, fill_value=0.0)

    # -------------------------------------------------------------------------
    # Feature Engineering (Multi-Feature Panel Mode)
    # -------------------------------------------------------------------------
    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # Dynamically detect available Feature.i columns
        feat_indices = sorted(
            [
                int(col.split(".")[1])
                for col in features.columns.get_level_values(0).unique()
                if str(col).startswith("Feature.") and col.split(".")[1].isdigit()
            ]
        )

        g = {i: features[f"Feature.{i}"] for i in feat_indices}
        r = {i: self._csrank(g[i]) for i in feat_indices}

        # Fallback safe indexing for spreads and interactions
        f0 = feat_indices[0]
        f1 = feat_indices[1] if len(feat_indices) > 1 else f0
        f2 = feat_indices[2] if len(feat_indices) > 2 else f1
        f3 = feat_indices[3] if len(feat_indices) > 3 else f2

        # Orthogonalized spreads remove redundant cross-sectional collinearity
        s10 = self._csrank(g[f1] - g[f0])
        s20 = self._csrank(self._orthogonalize(g[f2] - g[f0], g[f1] - g[f0]))
        s30 = self._csrank(self._orthogonalize(g[f3] - g[f0], g[f1] - g[f0]))

        return {
            "s10": s10,
            "s20": s20,
            "s30": s30,
            "i10_2": (r[f1] - r[f0]) * r[f2],
            "i20_3": (r[f2] - r[f0]) * r[f3],
            "i31_0": (r[f3] - r[f1]) * r[f0],
            "p12": r[f1] * r[f2],
            "r0": r[f0],
            "r1": r[f1],
        }

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["s10"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["s10"].index

    # -------------------------------------------------------------------------
    # Feature Engineering (Price-Only Walk-Forward Mode)
    # -------------------------------------------------------------------------
    def compute_price_features(self, df_prices: pd.DataFrame) -> pd.Series:
        """
        Calculates stationary, normalized alpha signals from close prices.
        """
        log_ret = np.log(df_prices / df_prices.shift(1))
        rolling_std = log_ret.rolling(
            window=self.vol_window, min_periods=max(2, self.vol_window // 2)
        ).std()
        rolling_std = rolling_std.replace(0, np.nan)

        # Alpha Component A: Medium-term normalized momentum
        ret_long = df_prices.pct_change(periods=self.mom_long_window)
        alpha_mom = ret_long / (rolling_std * np.sqrt(252))

        # Alpha Component B: Short-term mean-reversion
        ret_short = df_prices.pct_change(periods=self.reversal_window)
        alpha_reversal = -1.0 * (ret_short / (rolling_std * np.sqrt(252)))

        # Combined raw multi-factor composite signal
        raw_signal = 0.6 * alpha_mom + 0.4 * alpha_reversal
        return raw_signal.iloc[-1]

    # -------------------------------------------------------------------------
    # Training & Inference Entrypoints
    # -------------------------------------------------------------------------
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)

        dtrain = xgb.DMatrix(X[ok], label=y[ok])
        self.models = []
        for s in self.seeds:
            p = dict(self.params, seed=s)
            self.models.append(xgb.train(p, dtrain, num_boost_round=self.n_rounds))

    def predict(self, current_data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        """
        Unified AlphaNova entrypoint.
        Automatically routes between trained XGBoost panel inference and
        walk-forward price-based alpha generation depending on input format.
        """
        # Mode A: Trained XGBoost Multi-Feature Panel
        if len(self.models) > 0 and any(
            str(c).startswith("Feature.")
            for c in current_data.columns.get_level_values(0)
        ):
            X, tickers, index = self._design(current_data)
            dtest = xgb.DMatrix(np.nan_to_num(X, nan=0.0))

            raw = np.mean([m.predict(dtest) for m in self.models], axis=0)
            pred = pd.DataFrame(
                raw.reshape(-1, len(tickers)), index=index, columns=tickers
            )

            # Dual-horizon band-pass extraction
            fast = pred.ewm(span=self.fast_span, min_periods=1).mean()
            slow = fast.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
            signal = (fast - slow).fillna(0.0)

            # Cross-sectional de-meaning + scale-free tail sharpening
            p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
            out = np.sign(p) * (p.abs() ** self.tail_power)
            return out.sub(out.mean(axis=1), axis=0).fillna(0.0)

        # Mode B: Walk-Forward Price Series (e.g., OHLCV MultiIndex or Close DataFrame)
        if isinstance(current_data.columns, pd.MultiIndex):
            if "close" in current_data.columns.get_level_values(1):
                prices = current_data.xs("close", level=1, axis=1)
            else:
                prices = current_data.iloc[:, :].droplevel(1, axis=1)
        else:
            prices = current_data

        if len(prices) < self.mom_long_window + 5:
            return pd.Series(0.0, index=prices.columns)

        raw_alpha = self.compute_price_features(prices)
        target_weights = self.cross_sectional_rank_normalize(raw_alpha)
        return target_weights.fillna(0.0).replace([np.inf, -np.inf], 0.0)
