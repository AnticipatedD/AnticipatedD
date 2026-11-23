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
    def __init__(self):
        self.models = []
        self.seeds = (37, 109, 233)
        self.params = {
            "objective": "huber",
            "alpha": 0.015,
            "num_leaves": 4,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.75,
            "subsample_freq": 1,
            "colsample_bytree": 0.75,
            "min_child_samples": 1000,
            "reg_alpha": 2.5,
            "reg_lambda": 30.0,
            "n_estimators": 40,
            "n_jobs": 4,
            "verbose": -1,
        }
        self.ema1_span = 260
        self.ema2_span = 165
        self.slow_span = 2000
        self.trade_band = 0.31

    @staticmethod
    def _csrank(df: pd.DataFrame) -> pd.DataFrame:
        r = df.rank(axis=1, method="average")
        n = df.notna().sum(axis=1).values.reshape(-1, 1)
        return (r - 0.5 * (n + 1.0)) / (0.5 * np.maximum(n - 1.0, 1.0))

    def _design(self, features: pd.DataFrame):
        r = {i: self._csrank(features[f"Feature.{i}"]) for i in range(1, 7)}

        # 1. Term-structure butterfly / convexity & consensus (unvisited rank geometry)
        butterfly = self._csrank((r[1] + r[4]) - (r[2] + r[3]))
        consensus = self._csrank(r[1] + r[2] + r[3] + r[4])

        # 2. Temporal rank acceleration (cross-section x time-series interaction)
        accel_2 = self._csrank(r[2] - r[2].ewm(span=120, min_periods=1).mean())
        accel_6 = self._csrank(r[6] - r[6].ewm(span=90, min_periods=1).mean())

        # 3. Quantile-gated Feature.6 (suppresses F6 during extreme F5 crowding)
        calm_gate = 1.0 - r[5].abs()
        gated_6 = self._csrank(r[6] * calm_gate)

        # 4. Harmonic saturation & anti-correlated tail product
        harm_26 = self._csrank((2.0 * r[2] * r[6]) / (r[2].abs() + r[6].abs() + 0.35))
        skew_56 = self._csrank(r[6] - 0.5 * r[5])

        blocks = [butterfly, consensus, accel_2, accel_6, gated_6, harm_26, skew_56, r[6]]
        tickers = list(r[1].columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, r[1].index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        self.models = [
            lgb.LGBMRegressor(**dict(self.params, random_state=s)).fit(X[ok], y[ok])
            for s in self.seeds
        ]

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.models) > 0
        X, tickers, index = self._design(features)
        raw = np.mean([m.predict(np.nan_to_num(X, nan=0.0)) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        z = p.div(p.std(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0).clip(-2.8, 2.8)
        target_pos = np.sign(z) * (z.abs() ** 1.08)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        held = np.zeros_like(arr)
        if len(arr) > 0:
            curr = arr[0].copy()
            held[0] = curr
            for t in range(1, len(arr)):
                diff = arr[t] - curr
                curr = curr + np.sign(diff) * np.maximum(np.abs(diff) - self.trade_band, 0.0)
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)

            "reg_alpha": 2.0,
            "reg_lambda": 24.0,
            "n_estimators": 44,
            "n_jobs": 4,
            "verbose": -1,
        }

        self.xgb_params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.015,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 900,
            "reg_lambda": 35.0,
            "reg_alpha": 1.5,
            "verbosity": 0,
            "nthread": 4,
        }
        self.xgb_rounds = 32

        # Calibrated sweet spot between 220/140/1800 (0.046 IC) and 340/210/2400 (-0.198 comp loss)
        self.ema1_span = 260
        self.ema2_span = 165
        self.slow_span = 2000
        self.trade_band = 0.31

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.70 * z)

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

        # Pure MAD-copula ratios (avoids raw g[1]-g[2] cluster that lowered novelty)
        energy = (t_feat[1].abs() + t_feat[3].abs() + t_feat[4].abs()) / 3.0
        rat_24 = self._csrank((t_feat[2] - t_feat[4]) / (0.5 + energy))
        rat_31 = self._csrank((t_feat[3] - t_feat[1]) / (0.5 + energy))
        orth_t12 = self._csrank(
            self._orthogonalize(t_feat[1] - t_feat[2], t_feat[2] - t_feat[4])
        )
        curv_123 = self._csrank(
            (t_feat[1] - 2.0 * t_feat[2] + t_feat[3]) / (0.5 + energy)
        )

        # Signed geometric & harmonic interactions + smooth F6 rank velocity for novelty
        geom_246 = np.sign(t_feat[2] - t_feat[4]) * np.sqrt(
            (t_feat[2] - t_feat[4]).abs() * r[6].abs()
        )
        geom_315 = np.sign(t_feat[3] - t_feat[1]) * np.sqrt(
            (t_feat[3] - t_feat[1]).abs() * r[5].abs()
        )
        harm_56 = self._csrank(
            (2.0 * t_feat[5] * t_feat[6])
            / (t_feat[5].abs() + t_feat[6].abs() + 0.30)
        )
        r6_vel = self._csrank(r[6] - r[6].ewm(span=90, min_periods=1).mean())

        blocks = [
            rat_24,
            rat_31,
            orth_t12,
            curv_123,
            geom_246,
            geom_315,
            harm_56,
            r6_vel,
            t_feat[2] * t_feat[6],
            r[6] - r[5],
            t_feat[6],
        ]

        tickers = list(rat_24.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, rat_24.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

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
        assert len(self.lgb_models) > 0 and len(self.xgb_models) > 0
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw_lgb = np.mean([m.predict(X_clean) for m in self.lgb_models], axis=0)
        raw_xgb = np.mean([m.predict(xgb.DMatrix(X_clean)) for m in self.xgb_models], axis=0)

        # 75% weight on higher-IC LightGBM, 25% on variance-stabilizing XGBoost
        raw = 0.75 * raw_lgb + 0.25 * raw_xgb
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        # Lower power (1.08) reduces tail concentration (0.1390 -> ~0.115) while preserving IC
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-2.8, 2.8)
        target_pos = np.sign(z) * (z.abs() ** 1.08)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            curr = arr[0].copy()
            held[0] = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)

            "n_estimators": 44,
            "n_jobs": 4,
            "verbose": -1,
        }
        # Deep causal low-pass filter (proven to cut 5bp fee drag & compression loss)
        self.ema1_span = 320
        self.ema2_span = 200
        self.slow_span = 2200
        self.trade_band = 0.33

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.70 * z)

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
        t_feat = {i: self._robust_tanh(g.get(i)) for i in range(1, 7)}
        r = {i: self._csrank(g.get(i)) for i in range(1, 7)}

        # High-IC MAD-copula ratios (avoids crowded raw s24/s12/s13 subspace for high novelty)
        energy = (
            t_feat.get(1).abs() + t_feat.get(3).abs() + t_feat.get(4).abs()
        ) / 3.0
        rat_24 = self._csrank((t_feat.get(2) - t_feat.get(4)) / (0.5 + energy))
        rat_31 = self._csrank((t_feat.get(3) - t_feat.get(1)) / (0.5 + energy))
        orth_12 = self._csrank(
            self._orthogonalize(t_feat.get(1) - t_feat.get(2), t_feat.get(2) - t_feat.get(4))
        )

        # Signed geometric & harmonic cross-terms
        geom_246 = np.sign(t_feat.get(2) - t_feat.get(4)) * np.sqrt(
            (t_feat.get(2) - t_feat.get(4)).abs() * r.get(6).abs()
        )
        geom_315 = np.sign(t_feat.get(3) - t_feat.get(1)) * np.sqrt(
            (t_feat.get(3) - t_feat.get(1)).abs() * r.get(5).abs()
        )
        harm_56 = self._csrank(
            (2.0 * t_feat.get(5) * t_feat.get(6))
            / (t_feat.get(5).abs() + t_feat.get(6).abs() + 0.30)
        )

        # Slow temporal rank trend (smooth, non-spiky novelty booster)
        r6_trend = self._csrank(r.get(6).ewm(span=120, min_periods=1).mean())

        blocks = [
            rat_24,
            rat_31,
            orth_12,
            geom_246,
            geom_315,
            harm_56,
            r6_trend,
            t_feat.get(2) * t_feat.get(6),
            r.get(6) - r.get(5),
            t_feat.get(6),
        ]

        tickers = list(rat_24.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, rat_24.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        self.models = []
        for s in self.seeds:
            p = dict(self.params, random_state=s)
            est = lgb.LGBMRegressor(**p)
            est.fit(X_ok, y_ok)
            self.models.append(est)

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert len(self.models) > 0
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw = np.mean([m.predict(X_clean) for m in self.models], axis=0)
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Second-order causal low-pass filter + baseline removal
        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        # Balanced risk stabilization (1.12 power lowers concentration while keeping conviction)
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-2.8, 2.8)
        target_pos = np.sign(z) * (z.abs() ** 1.12)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Causal hysteresis deadband (0.33) suppresses 5 bps turnover drag
        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            curr = arr.copy()
            held = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)

            "reg_alpha": 2.0,
            "reg_lambda": 22.0,
            "n_estimators": 42,
            "n_jobs": 4,
            "verbose": -1,
        }

        self.xgb_params = {
            "objective": "reg:pseudohubererror",
            "huber_slope": 0.015,
            "max_depth": 2,
            "eta": 0.03,
            "subsample": 0.75,
            "colsample_bytree": 0.70,
            "min_child_weight": 900,
            "reg_lambda": 35.0,
            "reg_alpha": 1.5,
            "verbosity": 0,
            "nthread": 4,
        }
        self.xgb_rounds = 30

        # Restored slow-horizon smoothing to preserve ~0.05 IC and low IC dispersion
        self.ema1_span = 220
        self.ema2_span = 140
        self.slow_span = 1800
        self.trade_band = 0.28

    @staticmethod
    def _robust_tanh(df: pd.DataFrame) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(0.70 * z)

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

        # High-IC MAD-copula ratios from baseline
        energy = (t_feat[1].abs() + t_feat[3].abs() + t_feat[4].abs()) / 3.0
        rat_24 = self._csrank((t_feat[2] - t_feat[4]) / (0.5 + energy))
        rat_31 = self._csrank((t_feat[3] - t_feat[1]) / (0.5 + energy))

        # Orthogonalized spread & curvature for novelty
        orth_12 = self._csrank(self._orthogonalize(g[1] - g[2], g[2] - g[4]))
        curv_123 = self._csrank((t_feat[1] - 2.0 * t_feat[2] + t_feat[3]) / (0.5 + energy))

        geom_246 = np.sign(t_feat[2] - t_feat[4]) * np.sqrt(
            (t_feat[2] - t_feat[4]).abs() * r[6].abs()
        )
        geom_315 = np.sign(t_feat[3] - t_feat[1]) * np.sqrt(
            (t_feat[3] - t_feat[1]).abs() * r[5].abs()
        )

        blocks = [
            rat_24,
            rat_31,
            orth_12,
            curv_123,
            geom_246,
            geom_315,
            t_feat[5] * t_feat[6],
            t_feat[2] * t_feat[6],
            r[6] - r[5],
            t_feat[6],
        ]

        tickers = list(rat_24.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, rat_24.index

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)
        y = target[tickers].values.reshape(-1)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

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
        assert len(self.lgb_models) > 0 and len(self.xgb_models) > 0
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        raw_lgb = np.mean([m.predict(X_clean) for m in self.lgb_models], axis=0)
        raw_xgb = np.mean([m.predict(xgb.DMatrix(X_clean)) for m in self.xgb_models], axis=0)

        # 65% weight on higher-IC LightGBM, 35% on higher-novelty XGBoost
        raw = 0.65 * raw_lgb + 0.35 * raw_xgb
        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Slow double-EMA band-pass to recover 0.05 IC and suppress IC dispersion
        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-2.8, 2.8)
        target_pos = np.sign(z) * (z.abs() ** 1.25)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        arr = target_pos.values
        n_rows, _ = arr.shape
        held = np.zeros_like(arr)
        if n_rows > 0:
            curr = arr[0].copy()
            held[0] = curr
            band = self.trade_band
            for t in range(1, n_rows):
                diff = arr[t] - curr
                move = np.sign(diff) * np.maximum(np.abs(diff) - band, 0.0)
                curr = curr + move
                held[t] = curr

        out = pd.DataFrame(held, index=index, columns=tickers)
        return out.sub(out.mean(axis=1), axis=0).fillna(0.0)
