# /// script
# dependencies = [
#   "lightgbm>=4.0",
#   "numpy",
#   "pandas",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    """Cross-sectional predictor engineered to avoid prolonged flat stretches
    while preserving causal safety, row-position invariance, and low capacity."""

    def __init__(self):
        self.models = []
        self.seeds = [41, 89, 130]
        self.params = {
            "objective": "regression_l1",
            "boosting_type": "gbdt",
            "num_leaves": 6,
            "max_depth": 2,
            "learning_rate": 0.035,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "min_child_samples": 200,
            "reg_alpha": 1.0,
            "reg_lambda": 10.0,
            "n_estimators": 40,
            "n_jobs": 2,
            "verbose": -1,
        }

    def _csrank(self, df: pd.DataFrame) -> pd.DataFrame:
        ranked = df.rank(axis=1, method="average")
        counts = df.notna().sum(axis=1).to_numpy(dtype=np.float64).reshape(-1, 1)
        denom = np.maximum(counts - 1.0, 1.0)
        centered = ranked.to_numpy(dtype=np.float64) - 0.5 * (counts + 1.0)
        scaled = centered / (0.5 * denom)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns).fillna(0.0)

    def _cs_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        vals = df.to_numpy(dtype=np.float64)
        mean = np.nanmean(vals, axis=1, keepdims=True)
        std = np.nanstd(vals, axis=1, keepdims=True)
        safe_std = np.where(std > 1e-8, std, 1.0)
        z = np.where(std > 1e-8, (vals - mean) / safe_std, 0.0)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.DataFrame(np.clip(z, -3.0, 3.0), index=df.index, columns=df.columns)

    def _demean_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        clean = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        vals = clean.to_numpy(dtype=np.float64)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        centered = vals - np.mean(vals, axis=1, keepdims=True)
        return pd.DataFrame(centered, index=df.index, columns=df.columns)

    def _build_design(self, features: pd.DataFrame):
        tickers = list(features.columns.get_level_values(1).unique())
        n_rows = len(features)
        n_tickers = len(tickers)

        feat_names = [f"Feature.{i}" for i in range(1, 7)]
        blocks = []

        f_rank = {}
        f_z = {}
        for name in feat_names:
            sub = features[name][tickers]
            r_df = self._csrank(sub)
            z_df = self._cs_zscore(sub)
            f_rank[name] = r_df
            f_z[name] = z_df
            blocks.append(r_df.to_numpy(dtype=np.float64).reshape(-1, 1))
            blocks.append(z_df.to_numpy(dtype=np.float64).reshape(-1, 1))

        spread_12 = self._cs_zscore(f_z["Feature.1"] - f_z["Feature.2"])
        spread_34 = self._cs_zscore(f_z["Feature.3"] - f_z["Feature.4"])
        spread_56 = self._cs_zscore(f_z["Feature.5"] - f_z["Feature.6"])
        curv_246 = self._cs_zscore(
            f_rank["Feature.2"] - 2.0 * f_rank["Feature.4"] + f_rank["Feature.6"]
        )
        prod_13 = self._cs_zscore(f_z["Feature.1"] * f_z["Feature.3"])
        prod_25 = self._cs_zscore(f_z["Feature.2"] * f_z["Feature.5"])

        for extra_df in (spread_12, spread_34, spread_56, curv_246, prod_13, prod_25):
            blocks.append(extra_df.to_numpy(dtype=np.float64).reshape(-1, 1))

        X = np.hstack(blocks)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, tickers, features.index, n_rows, n_tickers

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _, _, _ = self._build_design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)
        y = aligned_target.to_numpy(dtype=np.float64).reshape(-1)

        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if not np.any(valid):
            self.models = []
            return

        X_train = X[valid]
        y_train = y[valid]

        self.models = []
        for seed in self.seeds:
            model = lgb.LGBMRegressor(**dict(self.params, random_state=seed))
            model.fit(X_train, y_train)
            self.models.append(model)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index, n_rows, n_tickers = self._build_design(features)
        if not self.models:
            zeros = pd.DataFrame(
                np.zeros((n_rows, n_tickers), dtype=np.float64),
                index=index,
                columns=tickers,
            )
            return self._demean_rows(zeros)

        preds_list = [m.predict(X) for m in self.models]
        raw_mean = np.mean(preds_list, axis=0).reshape(n_rows, n_tickers)
        raw_df = pd.DataFrame(raw_mean, index=index, columns=tickers)

        cs = self._demean_rows(raw_df).to_numpy(dtype=np.float64)
        row_std = np.std(cs, axis=1, keepdims=True)
        # Normalize each row by its cross-sectional dispersion when active
        # so positions stay responsive rather than collapsing to near-zero.
        active = row_std > 1e-9
        safe_std = np.where(active, row_std, 1.0)
        scaled = np.where(active, cs / safe_std, cs)
        out_df = pd.DataFrame(scaled, index=index, columns=tickers)
        return self._demean_rows(out_df)


if __name__ == "__main__":
    from walkforward import walkforward_backtest

    res = walkforward_backtest(MyPredictor, verbose=True)
    print(f"Quick backtest Sharpe: {res.get('sharpe', 0.0):.4f}")
