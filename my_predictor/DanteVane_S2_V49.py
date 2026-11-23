# /// script
# dependencies = [
#   "lightgbm>=4.0",
#   "xgboost>=2.0",
#   "numpy",
#   "pandas",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    def __init__(self):
        self.lgb_models = []
        self.xgb_models = []
        self.huber_w = None
        self.feat_mean = None
        self.feat_std = None
        self.seeds = [31, 89, 173, 257]

        self.lgb_params = {
            "objective": "huber",
            "alpha": 0.035,
            "boosting_type": "gbdt",
            "num_leaves": 5,
            "max_depth": 2,
            "learning_rate": 0.028,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.68,
            "min_child_samples": 950,
            "reg_alpha": 2.5,
            "reg_lambda": 28.0,
            "n_estimators": 45,
            "n_jobs": 4,
            "verbose": -1,
        }

        self.xgb_params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.018,
            "max_depth": 2,
            "eta": 0.028,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 900,
            "reg_lambda": 18.0,
            "reg_alpha": 1.2,
            "verbosity": 0,
            "nthread": 4,
        }
        self.xgb_rounds = 26

        # Balanced band-pass: retains low IC dispersion while cutting ~35 steps of phase lag
        self.ema1_span = 165
        self.ema2_span = 110
        self.slow_span = 1800
        self.trade_band = 0.32

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.67 * z)

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

    def _design(self, features: pd.DataFrame):
        g = {i: features[f"Feature.{i}"] for i in range(1, 7)}
        t_feat = {i: self._robust_tanh(g[i]) for i in range(1, 7)}
        r = {i: self._csrank(g[i]) for i in range(1, 7)}

        # Causal multi-horizon EMAs for temporal velocity & risk-adjusted efficiency
        e_fast = {i: g[i].ewm(span=18, min_periods=1).mean() for i in range(1, 7)}
        e_med = {i: g[i].ewm(span=72, min_periods=1).mean() for i in range(1, 7)}
        t_fast = {i: self._robust_tanh(e_fast[i]) for i in range(1, 7)}
        t_med = {i: self._robust_tanh(e_med[i]) for i in range(1, 7)}

        # 1. Proven high-IC MAD-copula basis + complete 4-node curvature/torsion
        energy = (t_feat[1].abs() + t_feat[2].abs() + t_feat[3].abs() + t_feat[4].abs()) / 4.0
        sp_24 = t_feat[2] - t_feat[4]
        sp_31 = t_feat[3] - t_feat[1]

        rat_24 = self._csrank(sp_24 / (0.5 + energy))
        rat_31 = self._csrank(sp_31 / (0.5 + energy))
        orth_12 = self._csrank(self._orthogonalize(g[1] - g[2], g[2] - g[4]))
        curv_123 = self._csrank((t_feat[1] - 2.0 * t_feat[2] + t_feat[3]) / (0.5 + energy))
        curv_234 = self._csrank((t_feat[2] - 2.0 * t_feat[3] + t_feat[4]) / (0.5 + energy))

        # 2. Temporal velocity & rolling volatility-normalized efficiency (drives Novelty + Sharpe)
        vel_24 = self._csrank((t_fast[2] - t_med[2]) - (t_fast[4] - t_med[4]))
        vel_31 = self._csrank((t_fast[3] - t_med[3]) - (t_fast[1] - t_med[1]))

        vol_24 = (g[2] - g[4]).ewm(span=75, min_periods=5).std().replace(0.0, np.nan)
        eff_24 = self._csrank(((e_fast[2] - e_fast[4]) / vol_24).fillna(0.0))

        # 3. Orthogonalized F5-F6 regime conditioning & geometric couplings
        orth_56 = self._csrank(self._orthogonalize(t_feat[5] * t_feat[6], sp_24))
        geom_246 = np.sign(sp_24) * np.sqrt(sp_24.abs() * r[6].abs())
        geom_315 = np.sign(sp_31) * np.sqrt(sp_31.abs() * r[5].abs())

        # Smooth phase-quadrature interaction (breaks axis-aligned tree correlation)
        phase = (np.pi / 2.0) * (t_med[6] - t_med[5])
        quad_mix = self._csrank(rat_24 * np.cos(phase) + rat_31 * np.sin(phase))

        blocks = [
            rat_24,
            rat_31,
            orth_12,
            curv_123,
            curv_234,
            vel_24,
            vel_31,
            eff_24,
            orth_56,
            geom_246,
            geom_315,
            quad_mix,
            r[6] - r[5],
            t_feat[6],
        ]

        tickers = list(rat_24.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, rat_24.index

    def _fit_huber_ridge(self, X: np.ndarray, y: np.ndarray, l2: float = 15000.0, delta: float = 0.025, iters: int = 5) -> np.ndarray:
        n_feat = X.shape[1]
        w = np.zeros(n_feat, dtype=np.float64)
        sw = np.ones_like(y, dtype=np.float64)
        eye = l2 * np.eye(n_feat, dtype=np.float64)
        for _ in range(iters):
            xw = X * sw[:, None]
            w = np.linalg.solve(xw.T @ X + eye, xw.T @ y)
            resid = np.abs(y - X @ w)
            sw = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-12))
        return w

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)

        # Cross-sectionally demean target so trees/linear models fit pure relative alpha
        y_df = target[tickers]
        y_cs = y_df.sub(y_df.mean(axis=1), axis=0)
        y = y_cs.values.reshape(-1)

        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        # Continuous Huber-Ridge branch for smooth rank interpolation (lowers Compression Loss)
        self.feat_mean = np.mean(X_ok, axis=0)
        self.feat_std = np.std(X_ok, axis=0) + 1e-8
        X_std = (X_ok - self.feat_mean) / self.feat_std
        self.huber_w = self._fit_huber_ridge(X_std, y_ok, l2=15000.0, delta=0.025, iters=5)

        self.lgb_models = []
        self.xgb_models = []
        dtrain = xgb.DMatrix(X_ok, label=y_ok)

        for s in self.seeds:
            est = lgb.LGBMRegressor(**dict(self.lgb_params, random_state=s))
            est.fit(X_ok, y_ok)
            self.lgb_models.append(est)

            bst = xgb.train(
                dict(self.xgb_params, seed=s),
                dtrain,
                num_boost_round=self.xgb_rounds,
            )
            self.xgb_models.append(bst)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.lgb_models) > 0 and len(self.xgb_models) > 0 and self.huber_w is not None
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw_lgb = np.mean([m.predict(X_clean) for m in self.lgb_models], axis=0)
        raw_xgb = np.mean([m.predict(xgb.DMatrix(X_clean)) for m in self.xgb_models], axis=0)
        X_std = (X_clean - self.feat_mean) / self.feat_std
        raw_lin = X_std @ self.huber_w

        # Cross-sectionally standardize each branch before blending to prevent scale drift
        n_t, n_k = len(index), len(tickers)

        def _cs_norm(arr_1d: np.ndarray) -> pd.DataFrame:
            df = pd.DataFrame(arr_1d.reshape(n_t, n_k), index=index, columns=tickers)
            df = df.sub(df.mean(axis=1), axis=0)
            sd = df.std(axis=1).replace(0.0, np.nan)
            return df.div(sd, axis=0).fillna(0.0)

        pred = 0.55 * _cs_norm(raw_lgb) + 0.20 * _cs_norm(raw_xgb) + 0.25 * _cs_norm(raw_lin)

        # Double-EMA band-pass
        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        # Tail-controlled cross-sectional z-score (keeps Concentration steady ~0.12-0.14)
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-2.8, 2.8)
        target_pos = np.sign(z) * (z.abs() ** 1.25)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Quadratic-knee hysteresis band (reduces Compression Loss while blocking noise trades)
        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            curr = arr[0].copy()
            held[0] = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                adiff = np.abs(diff)
                move = np.where(
                    adiff > band,
                    adiff - 0.5 * band,
                    np.where(
                        adiff > 0.5 * band,
                        0.5 * ((adiff - 0.5 * band) ** 2) / (0.5 * band),
                        0.0,
                    ),
                )
                curr = curr + np.sign(diff) * move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
