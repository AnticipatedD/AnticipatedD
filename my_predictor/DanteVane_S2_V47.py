# /// script
# dependencies = [
#   "numpy",
#   "pandas",
# ]
# ///

import warnings
import numpy as np
import pandas as pd
from predictor import Predictor

warnings.filterwarnings("ignore")


class MyPredictor(Predictor):
    """
    Season 2 Orthogonal Architecture:
    - Cross-Asset Graph Laplacian Peer-Spillover Residuals
    - Multi-Scale Causal Signal-to-Noise & Phase-Quadrature Basis
    - Random Fourier Feature (RFF) + Iteratively Reweighted Huber-Ridge Solver
    - L2-Bounded Concentration & Quadratic Hysteresis Execution
    """

    def __init__(self):
        self.weights = None
        self.rff_W = None
        self.rff_b = None
        self.feat_mean = None
        self.feat_std = None

        # RFF & Regularization hyperparameters (strictly bounded DoF for K=0.5 gate)
        self.n_rff = 96
        self.rff_gamma = 0.35
        self.l2_reg = 18000.0
        self.huber_delta = 0.025
        self.seed = 42

        # Execution & Smoothing parameters (low turnover, bounded concentration)
        self.ema1_span = 110
        self.ema2_span = 75
        self.slow_span = 1750
        self.trade_band = 0.28
        self.max_z = 2.5

    @staticmethod
    def _robust_tanh(df: pd.DataFrame, scale: float = 0.65) -> pd.DataFrame:
        med = df.median(axis=1)
        mad = (df.sub(med, axis=0)).abs().median(axis=1).replace(0.0, np.nan)
        z = df.sub(med, axis=0).div(1.4826 * mad, axis=0).fillna(0.0)
        return np.tanh(scale * z)

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
        raw = {i: features[f"Feature.{i}"] for i in range(1, 7)}

        # 1. Causal Multi-Horizon Frequency Decomposition
        f_fast = {i: raw[i].ewm(span=16, min_periods=1).mean() for i in range(1, 7)}
        f_med = {i: raw[i].ewm(span=64, min_periods=1).mean() for i in range(1, 7)}
        f_slow = {i: raw[i].ewm(span=192, min_periods=1).mean() for i in range(1, 7)}

        t_fast = {i: self._robust_tanh(f_fast[i]) for i in range(1, 7)}
        t_med = {i: self._robust_tanh(f_med[i]) for i in range(1, 7)}
        t_slow = {i: self._robust_tanh(f_slow[i]) for i in range(1, 7)}

        # 2. Cross-Asset Graph Spillover (Peer-Residual Alpha)
        # Uses rolling cross-sectional rank similarity of Feature.1..4 to diffuse peer signal
        composite = (t_fast[1] + t_fast[2] + t_fast[3] + t_fast[4]) / 4.0
        comp_vals = composite.fillna(0.0).values
        n_t, n_k = comp_vals.shape
        peer_spill = np.zeros_like(comp_vals)

        # Causal exponentially weighted cross-asset affinity matrix (decay=0.985)
        cov = np.eye(n_k, dtype=np.float64)
        decay = 0.985
        for t in range(n_t):
            row = comp_vals[t]
            if t > 10:
                d = np.sqrt(np.maximum(np.diag(cov), 1e-6))
                corr = np.clip(cov / np.outer(d, d), 0.0, 1.0)
                np.fill_diagonal(corr, 0.0)
                deg = corr.sum(axis=1, keepdims=True)
                norm_adj = corr / np.maximum(deg, 1e-6)
                peer_spill[t] = row - (norm_adj @ row)
            cov = decay * cov + (1.0 - decay) * np.outer(row, row)

        graph_res = self._csrank(
            pd.DataFrame(peer_spill, index=composite.index, columns=composite.columns)
        )

        # 3. Rolling Signal-to-Noise Efficiency (Risk-Adjusted Momentum per Asset)
        vol_14 = (raw[1] - raw[4]).ewm(span=80, min_periods=5).std().replace(0.0, np.nan)
        snr_14 = self._csrank(((f_fast[1] - f_fast[4]) / vol_14).fillna(0.0))

        vol_23 = (raw[2] - raw[3]).ewm(span=80, min_periods=5).std().replace(0.0, np.nan)
        snr_23 = self._csrank(((f_fast[2] - f_fast[3]) / vol_23).fillna(0.0))

        # 4. Orthogonalized Multi-Scale Trajectory & Phase-Quadrature Features
        vel_14 = self._csrank((t_fast[1] - t_med[1]) - (t_fast[4] - t_med[4]))
        vel_23 = self._csrank((t_fast[2] - t_med[2]) - (t_fast[3] - t_med[3]))
        curv_1234 = self._csrank(t_fast[1] - 2.0 * t_fast[2] + 2.0 * t_fast[3] - t_fast[4])

        # Orthogonalize slow drift against fast impulse to capture pure regime shifts
        orth_drift_6 = self._csrank(self._orthogonalize(t_slow[6], t_fast[6]))
        orth_spread_56 = self._csrank(
            self._orthogonalize(t_fast[5] - t_fast[6], t_fast[1] - t_fast[4])
        )

        # Smooth trigonometric coupling (avoids axis-aligned GBDT city overlap)
        angle = (np.pi / 2.0) * (t_med[5] - t_med[6])
        trig_sin = self._csrank(snr_14 * np.sin(angle))
        trig_cos = self._csrank(snr_23 * np.cos(angle))

        blocks = [
            graph_res,
            snr_14,
            snr_23,
            vel_14,
            vel_23,
            curv_1234,
            orth_drift_6,
            orth_spread_56,
            trig_sin,
            trig_cos,
        ]

        tickers = list(graph_res.columns)
        X = np.hstack([b[tickers].values.reshape(-1, 1) for b in blocks])
        return X, tickers, graph_res.index

    def _rff_transform(self, X_std: np.ndarray) -> np.ndarray:
        """Maps standardized features into smooth Random Fourier Feature space + linear skip."""
        proj = X_std @ self.rff_W + self.rff_b
        z_rff = np.sqrt(2.0 / self.n_rff) * np.cos(proj)
        return np.hstack([X_std, z_rff])

    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        X, tickers, _ = self._design(features)

        # Cross-sectionally demean target to isolate pure relative alpha
        y_df = target[tickers]
        y_cs = y_df.sub(y_df.mean(axis=1), axis=0)
        y = y_cs.values.reshape(-1)

        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X_ok, y_ok = X[ok], y[ok]

        # Standardize input basis
        self.feat_mean = np.mean(X_ok, axis=0)
        self.feat_std = np.std(X_ok, axis=0) + 1e-8
        X_std = (X_ok - self.feat_mean) / self.feat_std

        # Initialize deterministic Random Fourier projection
        rng = np.random.default_rng(self.seed)
        d_in = X_std.shape[1]
        self.rff_W = rng.normal(0.0, self.rff_gamma, size=(d_in, self.n_rff))
        self.rff_b = rng.uniform(0.0, 2.0 * np.pi, size=(1, self.n_rff))

        Phi = self._rff_transform(X_std)
        n_basis = Phi.shape[1]

        # Iteratively Reweighted Least Squares (IRLS) Huber-Ridge Solver
        w = np.zeros(n_basis, dtype=np.float64)
        sample_w = np.ones_like(y_ok, dtype=np.float64)
        eye = self.l2_reg * np.eye(n_basis, dtype=np.float64)

        for _ in range(6):
            Phi_w = Phi * sample_w[:, None]
            w = np.linalg.solve(Phi_w.T @ Phi + eye, Phi_w.T @ y_ok)
            resid = np.abs(y_ok - Phi @ w)
            sample_w = np.where(
                resid <= self.huber_delta,
                1.0,
                self.huber_delta / np.maximum(resid, 1e-12),
            )

        self.weights = w

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        assert self.weights is not None
        X, tickers, index = self._design(features)
        X_clean = np.nan_to_num(X, nan=0.0)

        X_std = (X_clean - self.feat_mean) / self.feat_std
        Phi = self._rff_transform(X_std)
        raw = Phi @ self.weights

        pred = pd.DataFrame(raw.reshape(-1, len(tickers)), index=index, columns=tickers)

        # Low-lag double-EMA band-pass to suppress noise turnover
        lp1 = pred.ewm(span=self.ema1_span, min_periods=1).mean()
        lp2 = lp1.ewm(span=self.ema2_span, min_periods=1).mean()
        slow = lp2.shift(1).ewm(span=self.slow_span, min_periods=1).mean()
        signal = (lp2 - slow).fillna(0.0)

        # Penalty-safe cross-sectional z-score (concavity 1.10 keeps Concentration low & steady)
        p = signal.sub(signal.mean(axis=1), axis=0).fillna(0.0)
        scale = p.std(axis=1).replace(0.0, np.nan)
        z = p.div(scale, axis=0).fillna(0.0).clip(-self.max_z, self.max_z)
        target_pos = np.sign(z) * (z.abs() ** 1.10)
        target_pos = target_pos.sub(target_pos.mean(axis=1), axis=0).fillna(0.0)

        # Quadratic hysteresis band (suppresses micro-rebalances while tracking macro shifts)
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
        out = out.sub(out.mean(axis=1), axis=0).fillna(0.0)

        # Strict cross-sectional L2 normalization to lock Concentration steady across regimes
        l2 = np.sqrt((out ** 2).sum(axis=1)).replace(0.0, np.nan)
        return out.div(l2, axis=0).fillna(0.0)
