# /// script
# dependencies = ["catboost>=1.2"]
# ///

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.models = []
        self.seeds = (7, 41, 97)
        self.lgb_params = {
            "objective": "huber",
            "alpha": 0.015,
            "boosting_type": "gbdt",
            "extra_trees": True,
            "num_leaves": 4,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.72,
            "subsample_freq": 1,
            "colsample_bytree": 0.70,
            "min_child_samples": 950,
            "reg_alpha": 2.5,
            "reg_lambda": 25.0,
            "n_estimators": 30,
            "n_jobs": 4,
            "verbose": -1,
        }
        self.fir_window = 320
        self.base_window = 2000
        self.deadband = 0.22

    @staticmethod
    def _robust_cs(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.65 * z)

    @staticmethod
    def _cs_pct(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, pct=True)
        return (2.0 * r - 1.0).fillna(0.0)

    def _build_features(self, features: pd.DataFrame) -> dict[str, pd.DataFrame]:
        raw = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        rob = {i: self._robust_cs(raw.get(i)) for i in range(1, 7)}
        pct = {i: self._cs_pct(raw.get(i)) for i in range(1, 7)}

        # Nonlinear spatial ratios & asymmetric dispersion features
        core_energy = (rob.get(1).abs() + rob.get(2).abs() + rob.get(4).abs()) / 3.0
        ratio_21 = self._cs_pct((rob.get(2) - rob.get(1)) / (0.5 + core_energy))
        ratio_24 = self._cs_pct((rob.get(2) - rob.get(4)) / (0.5 + core_energy))
        ratio_34 = self._cs_pct((rob.get(3) - rob.get(4)) / (0.5 + core_energy))

        # Multi-horizon acceleration on Feature.6
        f6_fast = pct.get(6).rolling(12, min_periods=1).mean()
        f6_slow = pct.get(6).rolling(96, min_periods=1).mean()
        accel_6 = self._cs_pct(f6_fast - f6_slow)

        # Signed geometric interactions
        geom_216 = np.sign(rob.get(2) - rob.get(1)) * np.sqrt((rob.get(2) - rob.get(1)).abs() * pct.get(6).abs())
        geom_245 = np.sign(rob.get(2) - rob.get(4)) * np.sqrt((rob.get(2) - rob.get(4)).abs() * pct.get(5).abs())

        # Cross-sectional regime dispersion multiplier (positive IC Dispersion Corr)
        cs_disp = raw.get(2).std(axis=1) / raw.get(2).std(axis=1).rolling(240, min_periods=1).mean().replace(0.0, np.nan)
        cs_disp = cs_disp.fillna(1.0).clip(0.5, 2.0)

        return {
            "ratio_21": ratio_21,
            "ratio_24": ratio_24,
            "ratio_34": ratio_34,
            "accel_6": accel_6,
            "geom_216": geom_216,
            "geom_245": geom_245,
            "prod_56": rob.get(5) * rob.get(6),
            "regime_21": ratio_21.mul(cs_disp, axis=0),
            "rob_6": rob.get(6),
        }

    def _design(self, features: pd.DataFrame):
        blocks = self._build_features(features)
        tickers = list(blocks["ratio_21"].columns)
        X = np.hstack([d[tickers].values.reshape(-1, 1) for d in blocks.values()])
        return X, tickers, blocks["ratio_21"].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.models = []
        for s in self.seeds:
            params = dict(self.lgb_params, random_state=s)
            est = lgb.LGBMRegressor(**params)
            est.fit(X_ok, y_ok)
            self.models.append(est)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.models) > 0
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw = np.mean([m.predict(X_clean) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Causal Triangular FIR smoothing (distinct phase response vs cascaded EMA)
        w = np.arange(1, self.fir_window + 1, dtype=np.float64)
        w /= w.sum()
        arr_pred = pred.values
        n_rows, n_cols = arr_pred.shape
        fir_out = np.zeros_like(arr_pred)
        w_rev = w[::-1]
        w_cum = np.cumsum(w_rev)
        norm_fac = np.minimum(np.arange(1, n_rows + 1), self.fir_window)
        scale_prefix = np.where(norm_fac < self.fir_window, w_cum[norm_fac - 1], 1.0)

        for c in range(n_cols):
            conv = np.convolve(arr_pred[:, c], w_rev, mode="full")[:n_rows]
            fir_out[:, c] = conv / np.maximum(scale_prefix, 1e-12)

        fir_df = pd.DataFrame(fir_out, index=index, columns=tickers)
        baseline = fir_df.shift(1).rolling(self.base_window, min_periods=1).mean()
        signal = (fir_df - baseline).fillna(0.0)

        # Cross-sectional de-meaning + smooth sigmoid saturation
        centered = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        cs_std = centered.std(axis=1).replace(0.0, np.nan)
        z = centered.div(cs_std, axis=0).fillna(0.0)
        target_pos = 1.6 * np.tanh(0.85 * z)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Causal Hysteresis filter to suppress 5 bp hourly turnover cost
        pos_arr = target_pos.values
        held = np.zeros_like(pos_arr)
        if n_rows > 0:
            curr = pos_arr[0].copy()
            held[0] = curr
            band = self.deadband
            for t in range(1, n_rows):
                diff = pos_arr[t] - curr
                curr = curr + np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
        f4 = features["Feature.4"]
        f5 = features["Feature.5"]
        f6 = features["Feature.6"]

        # Cross-sectional quantile/rank transforms
        r1 = 2.0 * f1.rank(axis=1, pct=True) - 1.0
        r2 = 2.0 * f2.rank(axis=1, pct=True) - 1.0
        r3 = 2.0 * f3.rank(axis=1, pct=True) - 1.0
        r4 = 2.0 * f4.rank(axis=1, pct=True) - 1.0
        r5 = 2.0 * f5.rank(axis=1, pct=True) - 1.0
        r6 = 2.0 * f6.rank(axis=1, pct=True) - 1.0

        # Cross-sectional z-scores clipped for tail stability
        def csz(df: pd.DataFrame) -> pd.DataFrame:
            m = df.mean(axis=1)
            s = df.std(axis=1).replace(0.0, np.nan)
            return df.sub(m, axis=0).div(s, axis=0).fillna(0.0).clip(-3.5, 3.5)

        z1, z2, z4, z5, z6 = csz(f1), csz(f2), csz(f4), csz(f5), csz(f6)

        # Orthogonalized spreads & regime-modulated interactions
        s21 = 2.0 * (f2 - f1).rank(axis=1, pct=True) - 1.0
        s24 = 2.0 * (f2 - f4).rank(axis=1, pct=True) - 1.0
        s31 = 2.0 * (f3 - f1).rank(axis=1, pct=True) - 1.0

        # Short-horizon velocity on Feature.6 & regime dispersion
        v6 = 2.0 * (r6 - r6.ewm(span=24, min_periods=1).mean()).rank(axis=1, pct=True) - 1.0
        disp = (f1.std(axis=1) + f2.std(axis=1) + f4.std(axis=1)) / 3.0
        disp_scale = (disp / disp.ewm(span=240, min_periods=1).mean().replace(0.0, np.nan)).fillna(1.0).clip(0.5, 2.0)

        blocks = [
            s21,
            s24,
            s31,
            v6,
            (r2 - r1) * r6,
            (r2 - r4) * r5,
            np.tanh(z5 * z6),
            s21.mul(disp_scale, axis=0),
            r6,
        ]

        tickers = list(f1.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, f1.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._extract_panel(features)
        y = target[tickers].values.reshape(-1)
        valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok = X[valid]
        y_ok = y[valid]

        self.models = []
        for seed in self.seeds:
            params = dict(self.cb_params, random_seed=seed)
            cb = CatBoostRegressor(**params)
            cb.fit(X_ok, y_ok)
            self.models.append(cb)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        X, tickers, index = self._extract_panel(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw = np.mean([m.predict(X_clean) for m in self.models], axis=0)
        pred_mat = raw.reshape(-1, len(tickers))
        n_rows, n_cols = pred_mat.shape

        # Causal 1D Kalman state estimator (smooths noise with minimal lag vs cascaded EMA)
        est = np.zeros_like(pred_mat)
        if n_rows > 0:
            x_hat = pred_mat[0].copy()
            p_cov = np.ones(n_cols, dtype=np.float64)
            est[0] = x_hat
            q = self.kalman_q
            r = self.kalman_r
            for t in range(1, n_rows):
                p_pred = p_cov + q
                k_gain = p_pred / (p_pred + r)
                x_hat = x_hat + k_gain * (pred_mat[t] - x_hat)
                p_cov = (1.0 - k_gain) * p_pred
                est[t] = x_hat

        est_df = pd.DataFrame(est, index=index, columns=tickers)
        slow_drift = est_df.shift(1).ewm(span=self.drift_window, min_periods=1).mean()
        signal = (est_df - slow_drift).fillna(0.0)

        # Cross-sectional de-meaning + unit-variance scaling + tail sharpening
        centered = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        cs_std = centered.std(axis=1).replace(0.0, np.nan)
        z = centered.div(cs_std, axis=0).fillna(0.0).clip(-3.0, 3.0)
        target_pos = np.sign(z) * (z.abs() ** 1.35)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Causal hysteresis deadband to suppress 5 bp hourly turnover drag
        pos_arr = target_pos.values
        held = np.zeros_like(pos_arr)
        if n_rows > 0:
            curr = pos_arr[0].copy()
            held[0] = curr
            b = self.band
            for t in range(1, n_rows):
                diff = pos_arr[t] - curr
                curr = curr + np.sign(diff) * np.maximum(np.abs(diff) - b, 0.0)
                held[t] = curr

        preds = pd.DataFrame(held, index=index, columns=tickers)
        return preds.sub(preds.mean(axis=1), axis=0).fillna(0.0)
