import warnings
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from predictor import Predictor

warnings.filterwarnings("ignore")

class MyPredictor(Predictor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.xgb_models = []
        self.lgb_models = []
        
        # Hyperparameters for XGBoost (Preserving max_depth = 2 from Code 1, or easily adjustable)
        self.xgb_seeds = self.config.get("xgb_seeds", (42, 1024, 2026))
        self.xgb_params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": self.config.get("huber_slope", 1.0),
            "max_depth": self.config.get("max_depth", 2),
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

        # Hyperparameters for LightGBM (Preserving Code 2 configuration)
        self.lgb_seeds = self.config.get("lgb_seeds", (43, 107, 193, 277))
        self.lgb_params = {
            "objective": "huber",
            "alpha": 0.015,
            "boosting_type": "gbdt",
            "extra_trees": False,
            "num_leaves": 5,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.72,
            "min_child_samples": 950,
            "reg_alpha": 2.0,
            "reg_lambda": 15.0,
            "n_estimators": 42,
            "n_jobs": 4,
            "verbose": -1,
        }

        # Smoothing & Signal Spans (Blended from Code 1 & 2 design logic)
        self.fast_span = self.config.get("fast_span", 5)
        self.slow_span = self.config.get("slow_span", 20)
        self.tail_power = self.config.get("tail_power", 1.25)
        self.trade_band = self.config.get("trade_band", 0.28)
        
        # Price Fallback Parameters
        self.mom_long_window = self.config.get("mom_long_window", 21)
        self.vol_window = self.config.get("vol_window", 21)
        self.reversal_window = self.config.get("reversal_window", 3)

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.80 * z)

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _orthogonalize(target_df: pd.DataFrame, basis_df: pd.DataFrame) -> pd.DataFrame:
        num = (target_df * basis_df).sum(axis=1)
        den = (basis_df * basis_df).sum(axis=1).replace(0.0, np.nan)
        beta = (num / den).fillna(0.0)
        return target_df.sub(basis_df.mul(beta, axis=0), axis=0)

    def cross_sectional_rank_normalize(self, raw_scores: pd.Series) -> pd.Series:
        valid_scores = raw_scores.dropna()
        if len(valid_scores) < 2:
            return raw_scores.fillna(0.0)
        q_low, q_high = valid_scores.quantile(0.15), valid_scores.quantile(0.85)
        clipped = valid_scores.clip(lower=q_low, upper=q_high)
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

    def _design(self, features: pd.DataFrame):
        feat_indices = sorted(
            [
                int(col.split(".")[1])
                for col in features.columns.get_level_values(0).unique()
                if str(col).startswith("Feature.") and col.split(".")[1].isdigit()
            ]
        )
        if not feat_indices:
            raise ValueError("No valid Feature.* columns detected.")

        g = {i: features[f"Feature.{i}"] for i in feat_indices}
        r = {i: self._csrank(g[i]) for i in feat_indices}
        t_feat = {i: self._robust_tanh(g[i]) for i in feat_indices}

        f0, f1 = feat_indices[0], feat_indices[1] if len(feat_indices) > 1 else feat_indices[0]
        f2 = feat_indices[2] if len(feat_indices) > 2 else f1
        f3 = feat_indices[3] if len(feat_indices) > 3 else f2
        f4 = feat_indices[4] if len(feat_indices) > 4 else f3
        f5 = feat_indices[5] if len(feat_indices) > 5 else f4

        # Integrated feature engineering capturing robust copula ratios and orthogonalized spreads
        s10 = self._csrank(g[f1] - g[f0])
        s20 = self._csrank(self._orthogonalize(g[f2] - g[f0], g[f1] - g[f0]))
        
        energy = (t_feat.get(f1, r[f1]).abs() + t_feat.get(f3, r[f3]).abs() + t_feat.get(f5, r[f5]).abs()) / 3.0
        rat_31 = self._csrank((t_feat.get(f3, r[f3]) - t_feat.get(f1, r[f1])) / (0.5 + energy))
        rat_45 = self._csrank((t_feat.get(f4, r[f4]) - t_feat.get(f3, r[f3])) / (0.5 + energy))

        blocks = [
            s10,
            s20,
            rat_31,
            rat_45,
            r[f0],
            r[f1],
            (r[f1] - r[f0]) * r[f2],
            t_feat.get(f1, r[f1]) * t_feat.get(f2, r[f2]),
            t_feat.get(f5, r[f5]) * t_feat.get(f6 if len(feat_indices)>5 else f5, r[f5]),
        ]
        
        tickers = list(s10.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, s10.index

    def compute_price_features(self, df_prices: pd.DataFrame) -> pd.Series:
        log_ret = np.log(df_prices / df_prices.shift(1))
        rolling_std = log_ret.rolling(window=self.vol_window, min_periods=max(2, self.vol_window // 2)).std().replace(0, np.nan)
        ret_long = df_prices.pct_change(periods=self.mom_long_window)
        alpha_mom = ret_long / (rolling_std * np.sqrt(252))
        ret_short = df_prices.pct_change(periods=self.reversal_window)
        alpha_reversal = -1.0 * (ret_short / (rolling_std * np.sqrt(252)))
        raw_signal = 0.6 * alpha_mom.fillna(0.0) + 0.4 * alpha_reversal.fillna(0.0)
        return raw_signal.iloc[-1]

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        # Train XGBoost Ensemble
        dtrain = xgb.DMatrix(X_ok, label=y_ok)
        self.xgb_models = []
        for s in self.xgb_seeds:
            p = dict(self.xgb_params, seed=s)
            self.xgb_models.append(xgb.train(p, dtrain, num_boost_round=self.n_rounds))

        # Train LightGBM Ensemble
        self.lgb_models = []
        for s in self.lgb_seeds:
            p = dict(self.lgb_params, random_state=s)
            est = lgb.LGBMRegressor(**p)
            est.fit(X_ok, y_ok)
            self.lgb_models.append(est)

    def predict(self, current_data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        if not any(str(c).startswith("Feature.") for c in current_data.columns.get_level_values(0)):
            prices = current_data.xs("close", level=1, axis=1) if isinstance(current_data.columns, pd.MultiIndex) and "close" in current_data.columns.get_level_values(1) else current_data
            if len(prices) < self.mom_long_window + 5:
                return pd.Series(0.0, index=prices.columns)
            raw_alpha = self.compute_price_features(prices)
            return self.cross_sectional_rank_normalize(raw_alpha).fillna(0.0)

        X, tickers, index = self._design(current_data)
        X_clean = np.nan_to_num(X, nan=0.0)

        dtest = xgb.DMatrix(X_clean)
        xgb_preds = np.mean([m.predict(dtest) for m in self.xgb_models], axis=0) if self.xgb_models else 0
        lgb_preds = np.mean([m.predict(X_clean) for m in self.lgb_models], axis=0) if self.lgb_models else 0

        blended_raw = 0.5 * xgb_preds + 0.5 * lgb_preds
        pred = pd.DataFrame(blended_raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        fast = pred.ewm(span=self.fast_span, min_periods=1).mean()
        slow = fast.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (fast - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        out = np.sign(p) * (p.abs() ** self.tail_power)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
