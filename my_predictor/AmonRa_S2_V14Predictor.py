import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V15Predictor(Predictor):
    def __init__(
        self,
        feature_mode: str = "v8_plus_ortho3",
        smooth_span: int = 120,
        tilt_span: int = 900,
        n_rounds: int = 40,
        max_depth: int = 2,
        min_child_weight: int = 600,
        eta: float = 0.04,
        shrink_tails: bool = True,
    ):
        super().__init__()
        self.model = None
        self.feature_mode = feature_mode
        self.smooth_span = smooth_span
        self.tilt_span = tilt_span
        self.n_rounds = n_rounds
        self.shrink_tails = shrink_tails
        self.params = {
            "objective": "reg:squarederror",
            "max_depth": max_depth,
            "eta": eta,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": min_child_weight,
            "reg_lambda": 8.0,
            "reg_alpha": 1.0,
            "seed": 42,
            "verbosity": 0,
            "nthread": 4,
        }

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    def _blocks(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        g1 = features["Feature.1"]
        g2 = features["Feature.2"]
        g3 = features["Feature.3"]
        g4 = features["Feature.4"]
        g5 = features["Feature.5"]
        g6 = features["Feature.6"]

        r1 = self._csrank(g1)
        r2 = self._csrank(g2)
        r3 = self._csrank(g3)
        r4 = self._csrank(g4)
        r5 = self._csrank(g5)
        r6 = self._csrank(g6)

        blocks = {
            "d24": self._csrank(g2 - g4),
            "d12": self._csrank(g1 - g2),
            "d13": self._csrank(g1 - g3),
            "m23_6": (r2 - r3) * r6,
            "m24_5": (r2 - r4) * r5,
            "p56": r5 * r6,
            "r6": r6,
            "r5": r5,
        }

        if self.feature_mode == "v8_plus_ortho3":
            ema12_f1 = g1.ewm(span=12, min_periods=1).mean()
            blocks["nl_14_6"] = self._csrank(g1 + g4) * self._csrank(g6.abs())
            blocks["nl_35_2"] = np.tanh(2.0 * r3) * (r5 - r2)
            blocks["ts_rev_13"] = -self._csrank(g1 - ema12_f1) * r3

        return blocks

    def _design(self, features: pd.DataFrame):
        b = self._blocks(features)
        tickers = list(b["d24"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in b.values()])
        return X, tickers, b["d24"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.model = xgb.train(
            self.params,
            xgb.DMatrix(X[ok], label=y[ok]),
            num_boost_round=self.n_rounds,
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.model is not None
        X, tickers, index = self._design(features)
        raw = self.model.predict(xgb.DMatrix(np.nan_to_num(X)))
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        pred = pred.ewm(span=self.smooth_span, min_periods=1).mean()
        tilt = pred.shift(1).ewm(span=self.tilt_span, min_periods=1).mean()
        pred = (pred - tilt).fillna(0.0)
        pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

        if self.shrink_tails:
            scale = pred.abs().quantile(0.90, axis=1).replace(0.0, 1.0).fillna(1.0)
            pred = np.tanh(pred.div(scale * 1.5, axis=0))
            pred = pred.sub(pred.mean(axis=1), axis=0).fillna(0.0)

        return pred


if __name__ == "__main__":
    import os

    if os.path.exists("features.parquet") and os.path.exists("target.parquet"):
        features = pd.read_parquet("features.parquet")
        target = pd.read_parquet("target.parquet")
    elif os.path.exists("data/features.parquet") and os.path.exists("data/target.parquet"):
        features = pd.read_parquet("data/features.parquet")
        target = pd.read_parquet("data/target.parquet")
    else:
        raise FileNotFoundError("Could not locate features.parquet and target.parquet.")

    dates = features.index.unique()
    split_date = dates[int(0.70 * len(dates))]
    feat_train, targ_train = features.loc[:split_date], target.loc[:split_date]
    feat_val, targ_val = features.loc[split_date:], target.loc[split_date:]

    experiments = [
        ("1_V8_Exact_Baseline", dict(feature_mode="v8_base", smooth_span=300, tilt_span=1800, n_rounds=30, shrink_tails=False)),
        ("2_V8_Faster_EMA_120", dict(feature_mode="v8_base", smooth_span=120, tilt_span=900, n_rounds=30, shrink_tails=False)),
        ("3_V8_Faster_EMA_60",  dict(feature_mode="v8_base", smooth_span=60,  tilt_span=600, n_rounds=30, shrink_tails=False)),
        ("4_V8_Low_Conc_Tanh",  dict(feature_mode="v8_base", smooth_span=120, tilt_span=900, n_rounds=30, shrink_tails=True)),
        ("5_V8_Plus_Ortho3",    dict(feature_mode="v8_plus_ortho3", smooth_span=120, tilt_span=900, n_rounds=40, shrink_tails=False)),
        ("6_V8_Ortho3_LowConc", dict(feature_mode="v8_plus_ortho3", smooth_span=120, tilt_span=900, n_rounds=40, shrink_tails=True)),
    ]

    print(f"{'Experiment':<22} | {'OOS_Sharpe':>10} | {'Mean_IC':>9} | {'IC_Std':>8} | {'Conc':>8}")
    print("-" * 68)

    for name, cfg in experiments:
        model = AmonRaS2V14Predictor(**cfg)
        model.train(feat_train, targ_train)
        preds = model.predict(feat_val)

        common_idx = preds.index.intersection(targ_val.index)
        common_cols = preds.columns.intersection(targ_val.columns)
        P = preds.loc[common_idx, common_cols]
        Y = targ_val.loc[common_idx, common_cols]

        daily_ic = P.corrwith(Y, axis=1).dropna()
        pnl = (P * Y).mean(axis=1).dropna()
        sharpe = float((pnl.mean() / (pnl.std() + 1e-12)) * np.sqrt(252))
        abs_w = P.abs()
        w_norm = abs_w.div(abs_w.sum(axis=1).replace(0.0, np.nan), axis=0)
        conc = float((w_norm ** 2).sum(axis=1).mean())

        print(f"{name:<22} | {sharpe:>10.4f} | {daily_ic.mean():>9.4f} | {daily_ic.std():>8.4f} | {conc:>8.4f}")
