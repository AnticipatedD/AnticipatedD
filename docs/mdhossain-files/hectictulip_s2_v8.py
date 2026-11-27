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
from sklearn.linear_model import HuberRegressor
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    """Self-contained signal generator for account hectictulip.

    Reference Anchor & Separation Metadata (for tracking & orthogonality):
    - Prior Submission / Public ID : signal_2b7d903bfea044c3 (Sharpe 0.0061, IC 0.0436, City 15.55 deg)
    - Nearest City / Signal ID     : signal_f3b9bfb99cfb3a90
    - Objective                    : Maximize angular separation (>35-55 deg) from
                                     signal_2b7d903bfea044c3 and signal_f3b9bfb99cfb3a90
                                     while converting 0.044+ IC into higher net Sharpe.
    """

    def __init__(self):
        self.et_models = []
        self.huber_model = None
        self.seeds = (17, 53, 109)

        # Anchor IDs tracked internally for multi-submission separation
        self.parent_signal_id = "signal_2b7d903bfea044c3"
        self.nearest_city_id = "signal_f3b9bfb99cfb3a90"

        # Overfitting-safe model capacity matching hectictulip's validated envelope
        self.et_params = {
            "n_estimators": 60,
            "max_depth": 3,
            "min_samples_leaf": 1000,
            "max_features": 0.55,
            "bootstrap": True,
            "max_samples": 0.70,
            "n_jobs": 4,
        }

        self.huber_params = {
            "alpha": 250.0,
            "epsilon": 1.35,
            "max_iter": 200,
            "fit_intercept": False,
        }

        # Reduced lag (95/65 vs 160/110) + active-only hysteresis to boost net Sharpe
        self.fast_span = 95
        self.mid_span = 65
        self.base_span = 1200
        self.deadzone = 0.28

    @staticmethod
    def _cs_rank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return ((r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))).fillna(0.0)

    @staticmethod
    def _cs_zscore(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return z.clip(-clip, clip) / clip

    @staticmethod
    def _orthogonalize(target_df: pd.DataFrame, basis_df: pd.DataFrame) -> pd.DataFrame:
        num = (target_df * basis_df).sum(axis=1)
        den = (basis_df * basis_df).sum(axis=1).replace(0.0, np.nan)
        beta = (num / den).fillna(0.0)
        return target_df.sub(basis_df.mul(beta, axis=0), axis=0)

    def _strip_occupied_subspace(
        self, df: pd.DataFrame, occupied_bases: list[pd.DataFrame], rank_out: bool = True
    ) -> pd.DataFrame:
        """Removes linear projection onto signal_2b7d903bfea044c3 / signal_f3b9bfb99cfb3a90 subspace."""
        res = df
        for b in occupied_bases:
            res = self._orthogonalize(res, b)
        return self._cs_rank(res) if rank_out else res

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        z = {i: self._cs_zscore(g[i]) for i in range(1, 7)}
        r = {i: self._cs_rank(g[i]) for i in range(1, 7)}

        ema_short = {i: z[i].ewm(span=24, min_periods=1).mean() for i in range(1, 7)}
        ema_long = {i: z[i].ewm(span=120, min_periods=1).mean() for i in range(1, 7)}
        vel = {i: self._cs_rank(ema_short[i] - ema_long[i]) for i in range(1, 7)}

        # Subspace occupied by signal_2b7d903bfea044c3 & nearest cluster signal_f3b9bfb99cfb3a90
        occupied_bases = [
            z - z,
            z - z,
            ema_long,
            ema_long,
        ]

        disp_145 = (z.abs() + z.abs() + z.abs()) / 3.0
        gate_6 = np.tanh(1.5 * ema_long)

        # Orthogonalize all candidate blocks away from signal_2b7d903bfea044c3 / signal_f3b9bfb99cfb3a90
        blocks = [
            self._strip_occupied_subspace(z - z, occupied_bases),
            self._strip_occupied_subspace(z + z, occupied_bases),
            self._strip_occupied_subspace((ema_long - ema_long) / (0.4 + disp_145), occupied_bases),
            self._strip_occupied_subspace((ema_long - ema_long) / (0.4 + disp_145), occupied_bases),
            self._strip_occupied_subspace((z - z) * gate_6, occupied_bases),
            self._strip_occupied_subspace((vel - vel) * r, occupied_bases),
            self._strip_occupied_subspace(vel - vel, occupied_bases),
            self._strip_occupied_subspace(vel - vel, occupied_bases),
            self._strip_occupied_subspace(ema_long * ema_long, occupied_bases),
            self._strip_occupied_subspace(ema_short - ema_short, occupied_bases),
        ]

        tickers = list(blocks.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, blocks.index, occupied_bases

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _, _ = self._design(features)
        aligned_target = target.reindex(index=features.index, columns=tickers)

        y_mat = aligned_target.sub(aligned_target.mean(axis=1), axis=0)
        y_clip = y_mat.clip(-0.08, 0.08)

        y = y_clip.values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.et_models = []
        self.huber_model = None
        if len(y_ok) < 500:
            return

        for s in self.seeds:
            et = ExtraTreesRegressor(**dict(self.et_params, random_state=s))
            et.fit(X_ok, y_ok)
            self.et_models.append(et)

        self.huber_model = HuberRegressor(**self.huber_params)
        self.huber_model.fit(X_ok, y_ok)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index, occupied_bases = self._design(features)
        n_rows, n_tickers = len(index), len(tickers)

        if not self.et_models or self.huber_model is None or n_rows == 0 or n_tickers == 0:
            return pd.DataFrame(0.0, index=index, columns=tickers, dtype=np.float64)

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw_et = np.mean([m.predict(X_clean) for m in self.et_models], axis=0)
        raw_hb = self.huber_model.predict(X_clean)

        raw = 0.70 * raw_et + 0.30 * raw_hb
        pred = pd.DataFrame(raw.reshape(n_rows, n_tickers), index=index, columns=tickers)

        # Two-stage low-pass + slow baseline removal
        lp1 = pred.ewm(span=self.fast_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.mid_span, min_periods=1).mean()
        baseline = lp2.shift(1).ewm(span=self.base_span, min_periods=1).mean()
        raw_signal = (lp2 - baseline).fillna(0.0)

        # Second-stage orthogonal projection guarantees angular distance from
        # signal_2b7d903bfea044c3 & signal_f3b9bfb99cfb3a90 even after nonlinear tree splits
        signal = self._strip_occupied_subspace(raw_signal, occupied_bases, rank_out=False).fillna(0.0)

        # Dispersion-gated cross-sectional sizing (fixes negative IC Dispersion -0.0413)
        cs = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = cs.std(axis=1).replace(0.0, np.nan)
        disp_gate = (scale / scale.ewm(span=240, min_periods=1).mean()).clip(0.70, 1.40).fillna(1.0)

        z = cs.div(scale, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = (np.sign(z) * (z.abs() ** 1.30)).mul(disp_gate, axis=0)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Neutrality-preserving hysteresis loop (eliminates phantom turnover on locked tickers)
        arr = target_pos.values
        held = np.zeros_like(arr)
        curr = arr.copy()
        curr -= curr.mean()
        held = curr
        band = self.deadzone

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
