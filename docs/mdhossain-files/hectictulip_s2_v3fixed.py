# /// script
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.et_models = []
        self.ridge_model = None
        self.seeds = (19, 67, 131, 211)

        self.et_params = {
            "n_estimators": 80,
            "max_depth": 3,
            "min_samples_leaf": 600,
            "max_features": 0.60,
            "bootstrap": True,
            "max_samples": 0.75,
            "n_jobs": 4,
        }

        self.ridge_alpha = 1500.0

        # Calibrated spans: ~29-bar composite lag (down from 134 bars) to capture
        # cross-sectional alpha before decay while suppressing hourly noise
        self.fast_span = 36
        self.mid_span = 24
        self.base_span = 480
        self.deadzone = 0.22

    @staticmethod
    def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    @staticmethod
    def _cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0)
        return z.clip(-clip, clip) / clip

    @staticmethod
    def _orthogonalize_multi(target_df: pd.DataFrame, bases: list[pd.DataFrame]) -> pd.DataFrame:
        """Cross-sectional sequential orthogonalization against baseline factor subspace."""
        res = target_df.copy()
        for b in bases:
            num = (res * b).sum(axis=1)
            den = (b * b).sum(axis=1).replace(0.0, np.nan)
            beta = (num / den).fillna(0.0)
            res = res.sub(b.mul(beta, axis=0), axis=0)
        return res

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        z = {i: self._cs_zscore(g[i]).fillna(0.0) for i in range(1, 7)}
        r = {i: self._cs_rank(g[i]) for i in range(1, 7)}

        # Baseline subspace to strip out crowded linear exposure (boosts City/Global Novelty)
        baseline_subspace = [z[1], z[2], z[3], z[4], z[5], z[6]]

        # 1. Multi-horizon causal velocity & acceleration (pure temporal dynamics)
        ema_12 = {i: z[i].ewm(span=12, min_periods=1).mean() for i in range(1, 7)}
        ema_48 = {i: z[i].ewm(span=48, min_periods=1).mean() for i in range(1, 7)}
        ema_168 = {i: z[i].ewm(span=168, min_periods=1).mean() for i in range(1, 7)}

        vel_fast = {i: ema_12[i] - ema_48[i] for i in range(1, 7)}
        vel_slow = {i: ema_48[i] - ema_168[i] for i in range(1, 7)}
        accel = {i: vel_fast[i] - vel_slow[i] for i in range(1, 7)}

        # 2. Nonlinear cross-feature curvature & conditional interactions
        # Captures second-order regime shifts invisible to linear baseline models
        interact_1 = (z[1] * z[6]) - (z[2] * z[4])
        interact_2 = (z[3] * z[5]) - (z[1] * z[4])
        abs_asym = (z[4].abs() - z[5].abs()) * np.sign(z[6])
        rank_spread_acc = (accel[4] - accel[5]) + 0.5 * (accel[1] - accel[2])
        flow_div = (vel_slow[3] - vel_slow[1]) * (1.0 - z[6].abs())
        tail_agree = np.tanh(2.0 * z[4]) * np.tanh(2.0 * z[5]) * z[3]
        vol_regime = (vel_fast[2] - vel_fast[6]) / (0.5 + (z[1].abs() + z[3].abs()))
        cross_mom = (r[4] - r[6]).ewm(span=36, min_periods=1).mean() - (r[4] - r[6])

        raw_blocks = [
            interact_1,
            interact_2,
            abs_asym,
            rank_spread_acc,
            flow_div,
            tail_agree,
            vol_regime,
            cross_mom,
        ]

        # Orthogonalize every engineered block against the 6 baseline features,
        # then rank-normalize so tree splits and ridge weights are scale-invariant
        blocks = [
            self._cs_rank(self._orthogonalize_multi(blk, baseline_subspace))
            for blk in raw_blocks
        ]

        tickers = list(blocks[0].columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, blocks[0].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        # Cross-sectional rank-normalized target reduces outlier distortion & improves IC Dtd
        y_cs = aligned_target.sub(aligned_target.mean(axis=1), axis=0)
        y_std = y_cs.std(axis=1).replace(0.0, np.nan)
        y_norm = y_cs.div(y_std, axis=0).clip(-3.0, 3.0)

        y = y_norm.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.et_models = []
        self.ridge_model = None
        if len(y_ok) < 500:
            return

        for s in self.seeds:
            et = ExtraTreesRegressor(**dict(self.et_params, random_state=s))
            et.fit(X_ok, y_ok)
            self.et_models.append(et)

        self.ridge_model = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        self.ridge_model.fit(X_ok, y_ok)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._design(features)
        n_rows, n_tickers = len(index), len(tickers)

        if not self.et_models or self.ridge_model is None or n_rows == 0 or n_tickers == 0:
            return pd.DataFrame(0.0, index=index, columns=tickers, dtype=np.float64)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw_et = np.mean([m.predict(X_clean) for m in self.et_models], axis=0)
        raw_rg = self.ridge_model.predict(X_clean)

        # Equal blend of nonlinear tree interactions and smooth L2 linear projection
        raw = 0.55 * raw_et + 0.45 * raw_rg
        pred = pd.DataFrame(raw.reshape(n_rows, n_tickers), index=index, columns=tickers)

        # Causal band-pass filter tuned to preserve IC while removing slow drift
        lp1 = pred.ewm(span=self.fast_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.mid_span, min_periods=1).mean()
        baseline = lp2.shift(1).ewm(span=self.base_span, min_periods=1).mean()
        signal = (lp2 - baseline).fillna(0.0)

        # Cross-sectional standardization with concave tanh shaping (cuts Concentration & Compression loss)
        cs = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = cs.std(axis=1).replace(0.0, np.nan)
        z = cs.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.tanh(0.85 * z)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Neutrality-preserving hysteresis deadzone (zero phantom turnover on locked tickers)
        arr = target_pos.values
        held = np.zeros_like(arr)
        curr = arr[0].copy()
        curr -= curr.mean()
        held[0] = curr
        band = self.deadzone

        for t in range(1, n_rows):
            diff = arr[t] - curr
            active = np.abs(diff) > band
            if np.any(active):
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                # Re-center only across active movers so locked tickers do not incur phantom trades
                active_mean = curr.mean()
                n_act = active.sum()
                curr[active] -= active_mean * (n_tickers / n_act)
            held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
