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
        self.ridge_model = Ridge(alpha=1800.0, fit_intercept=False)
        self.seeds = (17, 53, 97, 149)

        self.et_params = {
            "n_estimators": 80,
            "max_depth": 3,
            "min_samples_leaf": 950,
            "max_features": 0.50,
            "bootstrap": True,
            "max_samples": 0.70,
            "n_jobs": 4,
        }

        # Parallel medium/slow filter: keeps Run 1's 0.044+ IC without Run 1's phase lag
        self.med_span = 72
        self.slow_span = 180
        self.base_span = 1600
        self.med_weight = 0.45
        self.trade_band = 0.28

    @staticmethod
    def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return ((r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))).fillna(0.0)

    @staticmethod
    def _robust_tanh(df: pd.DataFrame, scale: float = 0.67) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(scale * z)

    @staticmethod
    def _orthogonalize_subspace(target_df: pd.DataFrame, bases: list[pd.DataFrame]) -> pd.DataFrame:
        """Removes linear projection onto all basis factors cross-sectionally at each bar."""
        res = target_df.copy()
        for b in bases:
            num = (res * b).sum(axis=1)
            den = (b * b).sum(axis=1).replace(0.0, np.nan)
            beta = (num / den).fillna(0.0)
            res = res.sub(b.mul(beta, axis=0), axis=0)
        return res

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        t = {i: self._robust_tanh(g[i]) for i in range(1, 7)}
        r = {i: self._cs_rank(g[i]) for i in range(1, 7)}

        # Full 6-factor baseline subspace (used to push predictions into an unvisited City)
        baseline_basis = [t[i] for i in range(1, 7)]

        # Smooth multi-horizon state features (no noisy 5-bar/20-bar rank jitter)
        e48 = {i: t[i].ewm(span=48, min_periods=1).mean() for i in range(1, 7)}
        e144 = {i: t[i].ewm(span=144, min_periods=1).mean() for i in range(1, 7)}

        # Dispersion-normalized SNR & nonlinear cross-feature regime interactions
        disp = (t[1].abs() + t[3].abs() + t[4].abs() + t[6].abs()) / 4.0

        raw_blocks = [
            (e48[4] - e48[5]) / (0.45 + disp),
            (e48[1] - e48[6]) / (0.45 + disp),
            (e144[2] - e144[4]) / (0.45 + disp),
            (e144[3] - e144[1]) / (0.45 + disp),
            (e48[6] - e144[6]) * (1.0 - t[4].abs()),
            (e48[4] - e144[4]) * (1.0 - t[6].abs()),
            t[4] * t[6] - t[2] * t[5],
            t[1] * t[5] - t[3] * t[6],
            np.sign(t[4] - t[5]) * np.sqrt((t[4] - t[5]).abs() * r[6].abs()),
            np.sign(t[1] - t[6]) * np.sqrt((t[1] - t[6]).abs() * r[5].abs()),
            (e48[1] - 2.0 * e48[3] + e48[5]) / (0.45 + disp),
            (e144[2] - 2.0 * e144[4] + e144[6]) / (0.45 + disp),
        ]

        # Orthogonalize EVERY block against the 6 baseline features so trees cannot
        # default back to crowded City 15.3° splits
        blocks = [
            self._cs_rank(self._orthogonalize_subspace(blk, baseline_basis))
            for blk in raw_blocks
        ]

        tickers = list(blocks[0].columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, blocks[0].index, baseline_basis

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _, _ = self._design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        # Cross-sectionally standardize target so every bar has equal variance weight
        y_cs = aligned_target.sub(aligned_target.mean(axis=1), axis=0)
        y_std = y_cs.std(axis=1).replace(0.0, np.nan)
        y_norm = y_cs.div(y_std, axis=0).clip(-3.0, 3.0)

        y = y_norm.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.et_models = []
        self.is_fitted = False
        if len(y_ok) < 500:
            return

        for s in self.seeds:
            et = ExtraTreesRegressor(**dict(self.et_params, random_state=s))
            et.fit(X_ok, y_ok)
            self.et_models.append(et)

        self.ridge_model.fit(X_ok, y_ok)
        self.is_fitted = True

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index, baseline_basis = self._design(features)
        n_rows, n_tickers = len(index), len(tickers)

        if not getattr(self, "is_fitted", False) or n_rows == 0 or n_tickers == 0:
            return pd.DataFrame(0.0, index=index, columns=tickers, dtype=np.float64)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw_et = np.mean([m.predict(X_clean) for m in self.et_models], axis=0)
        raw_rd = self.ridge_model.predict(X_clean)

        raw = 0.60 * raw_et + 0.40 * raw_rd
        pred = pd.DataFrame(raw.reshape(n_rows, n_tickers), index=index, columns=tickers)

        # Parallel medium/slow filter: avoids sequential phase lag while keeping IC smooth
        med_sig = pred.ewm(span=self.med_span, min_periods=1).mean()
        slow_sig = pred.ewm(span=self.slow_span, min_periods=1).mean()
        baseline = slow_sig.shift(1).ewm(span=self.base_span, min_periods=1).mean()

        blended = self.med_weight * med_sig + (1.0 - self.med_weight) * (slow_sig - baseline)

        # Strip residual nonlinear reconstruction of the 6 baseline features to lock in New City novelty
        signal = self._orthogonalize_subspace(blended, baseline_basis).fillna(0.0)

        # Concave tanh sizing prevents tail-concentration turnover drag
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.tanh(0.90 * z)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Neutrality-preserving hysteresis loop (zero phantom turnover on locked tickers)
        arr = target_pos.values
        held = np.zeros_like(arr)
        curr = arr[0].copy()
        curr -= curr.mean()
        held[0] = curr
        band = self.trade_band

        for t in range(1, n_rows):
            diff = arr[t] - curr
            active = np.abs(diff) > band
            if np.any(active):
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                n_act = active.sum()
                curr[active] -= curr.mean() * (n_tickers / n_act)
            held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
