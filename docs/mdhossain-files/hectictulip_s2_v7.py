# /// script
# dependencies = [
#   "lightgbm>=4.0",
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    """Contemporaneous cross-sectional predictor with rank-normalized L2/Huber
    targets and causal post-prediction smoothing to prevent flat collapse while
    clearing permutation overfitting gates."""

    def __init__(self):
        self.models = []
        self.ridge = Ridge(alpha=3000.0, fit_intercept=False)
        self.seeds = [41, 89, 130]
        self.params = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "num_leaves": 5,
            "max_depth": 2,
            "learning_rate": 0.04,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "min_child_samples": 600,
            "reg_alpha": 0.5,
            "reg_lambda": 20.0,
            "n_estimators": 35,
            "n_jobs": 4,
            "verbose": -1,
        }
        self.smooth_span = 96
        self.tilt_span = 480

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).to_numpy(dtype=np.float64).reshape(-1, 1)
        denom = np.maximum(n - 1.0, 1.0)
        scaled = (r.to_numpy(dtype=np.float64) - 0.5 * (n + 1.0)) / (0.5 * denom)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)

    @staticmethod
    def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        vals = df.to_numpy(dtype=np.float64)
        mean = np.nanmean(vals, axis=1, keepdims=True)
        std = np.nanstd(vals, axis=1, keepdims=True)
        valid = std > 1e-8
        safe_std = np.where(valid, std, 1.0)
        z = np.where(valid, (vals - mean) / safe_std, np.nan)
        return pd.DataFrame(np.clip(z, -3.0, 3.0), index=df.index, columns=df.columns)

    def _build_design(self, features: pd.DataFrame):
        tickers = list(features.columns.get_level_values(1).unique())
        n_rows = len(features)
        n_tickers = len(tickers)

        feat_names = [f"Feature.{i}" for i in range(1, 7)]
        f_rank = {}
        f_z = {}
        blocks = []

        for name in feat_names:
            sub = features[name][tickers]
            r_df = self._csrank(sub)
            z_df = self._cs_zscore(sub)
            f_rank[name] = r_df
            f_z[name] = z_df
            blocks.append(r_df.to_numpy(dtype=np.float64).reshape(-1, 1))
            blocks.append(z_df.to_numpy(dtype=np.float64).reshape(-1, 1))

        # Contemporaneous cross-sectional spreads & regime interactions
        extras = [
            self._csrank(f_z["Feature.1"] - f_z["Feature.2"]),
            self._csrank(f_z["Feature.3"] - f_z["Feature.4"]),
            self._csrank(f_z["Feature.5"] - f_z["Feature.6"]),
            self._csrank(f_z["Feature.2"] - f_z["Feature.4"]),
            self._csrank(f_z["Feature.3"] - f_z["Feature.1"]),
            self._csrank(f_rank["Feature.2"] - 2.0 * f_rank["Feature.4"] + f_rank["Feature.6"]),
            self._csrank(f_z["Feature.1"] * f_z["Feature.3"]),
            self._csrank(f_z["Feature.2"] * f_z["Feature.5"]),
        ]
        for extra_df in extras:
            blocks.append(extra_df.to_numpy(dtype=np.float64).reshape(-1, 1))

        X = np.hstack(blocks)
        return X, tickers, features.index, n_rows, n_tickers

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _, _, _ = self._build_design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        # Cross-sectionally rank-normalize target so L2 loss never collapses to zero
        y_rank = self._csrank(aligned_target)
        y = y_rank.to_numpy(dtype=np.float64).reshape(-1)

        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if np.sum(valid) < 500:
            self.models = []
            self.is_fitted = False
            return

        X_train = X[valid]
        y_train = y[valid]

        self.models = []
        for seed in self.seeds:
            model = lgb.LGBMRegressor(**dict(self.params, random_state=seed))
            model.fit(X_train, y_train)
            self.models.append(model)

        self.ridge.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index, n_rows, n_tickers = self._build_design(features)
        if not getattr(self, "is_fitted", False) or n_rows == 0 or n_tickers == 0:
            return pd.DataFrame(0.0, index=index, columns=tickers, dtype=np.float64)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw_lgb = np.mean([m.predict(X_clean) for m in self.models], axis=0)
        raw_rd = self.ridge.predict(X_clean)
        raw = (0.65 * raw_lgb + 0.35 * raw_rd).reshape(n_rows, n_tickers)

        pred_df = pd.DataFrame(raw, index=index, columns=tickers)

        # Causal temporal smoothing + slow per-instrument tilt removal
        smoothed = pred_df.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = smoothed.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        signal = (smoothed - tilt).fillna(smoothed)

        # Cross-sectional standard-deviation scaling guarantees non-flat active exposure
        cs = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        std = cs.std(axis=1).replace(0.0, np.nan)
        z = cs.div(std, axis=0).fillna(0.0).clip(-3.0, 3.0)

        # Smooth concave position sizing + strict row-wise zero-mean
        pos = np.tanh(1.1 * z)
        return pos.sub(pos.mean(axis=1), axis=0).fillna(0.0)


if __name__ == "__main__":
    from walkforward import walkforward_backtest

    res = walkforward_backtest(MyPredictor, verbose=True)
    print(f"Quick backtest Sharpe: {res.get('sharpe', 0.0):.4f}")
