# /// script
# dependencies = ["xgboost"]
# ///

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from predictor import Predictor

warnings.filterwarnings("ignore")


class AmonRaS2V9Predictor(Predictor):
Local Diagnostic & Backtest Script (AmonRaS2V12Predictor)
# Mirrors the 13-signal feature engineering, 70% GBDT + 30% Ridge blend,
# EMA smoothing/de-tilting, and cross-sectional standardization.

clear; clc; rng(42);

1. Load Data (file_name,sharpe,n_observations,timestamp,ic,ic_std,ic_dispersion_corr,concentration,compression_loss,city_novelty_deg,nearest_city)
Expected format: 6 matrices (F1..F6) and Target matrix Y, each of size [T x 20]
T = 4000;
N = 20;

F1 = randn(T, N);
F2 = randn(T, N);
F3 = randn(T, N);
F4 = randn(T, N);
F5 = randn(T, N);
F6 = randn(T, N);

# Synthetic target with weak signal on (F2 - F4)
raw_signal = F2 - F4;
Y = max(min(0.08 * raw_signal + randn(T, N), 5.0), -5.0);

## 2. Build the 13 Orthogonal Alpha Signals
[X_3d, signal_names] = build_13_signals(F1, F2, F3, F4, F5, F6);
num_signals = numel(signal_names);

## 3. Chronological 70/30 Train / Validation Split
split_idx = floor(0.70 * T);
T_train = split_idx;
T_val   = T - split_idx;

# Flatten panel [T x N x 13] -> [(T*N) x 13]
X_train = reshape(X_3d(1:split_idx, :, :), [], num_signals);
y_train = reshape(Y(1:split_idx, :), [], 1);

valid_mask = isfinite(y_train) & all(isfinite(X_train), 2);
X_train = X_train(valid_mask, :);
y_train = y_train(valid_mask);

X_val = reshape(X_3d(split_idx+1:end, :, :), [], num_signals);
X_val(~isfinite(X_val)) = 0.0;
Y_val = Y(split_idx+1:end, :);

## 4. Train 70% Gradient Boosted Trees + 30% Closed-Form Ridge
# 4a. Gradient Boosted Trees (requires Statistics and Machine Learning Toolbox)
t_tree = templateTree('MaxNumSplits', 7, ...      % Equivalent to max_depth = 3 (2^3 - 1)
                      'MinLeafSize', 350, ...     % Matches min_child_weight = 350
                      'NumVariablesToSample', floor(0.55 * num_signals));

gbdt_model = fitrensemble(X_train, y_train, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 65, ...
    'LearnRate', 0.06, ...
    'Learners', t_tree);

# 4b. Closed-Form L2 Ridge Regression
ridge_alpha = 5000.0;
feat_mean = mean(X_train, 1);
feat_std  = std(X_train, 0, 1);
feat_std(feat_std == 0) = 1.0;

X_train_std = (X_train - feat_mean) ./ feat_std;
y_train_ctr = y_train - mean(y_train);

gram = (X_train_std' * X_train_std) + ridge_alpha * eye(num_signals);
ridge_w = gram \ (X_train_std' * y_train_ctr);

## 5. Out-of-Sample Prediction & Post-Processing
raw_gbdt  = predict(gbdt_model, X_val);
X_val_std = (X_val - feat_mean) ./ feat_std;
raw_ridge = X_val_std * ridge_w;

# Scale-match before 70/30 blend
gbdt_norm  = (raw_gbdt - mean(raw_gbdt)) / (std(raw_gbdt) + 1e-8);
ridge_norm = (raw_ridge - mean(raw_ridge)) / (std(raw_ridge) + 1e-8);
raw_blend  = 0.70 * gbdt_norm + 0.30 * ridge_norm;

# Reshape back to [T_val x N]
pred = reshape(raw_blend, T_val, N);

# EMA Smoothing (span = 45) & Lagged Drift De-tilting (span = 450)
pred_smooth = ewm_mean(pred, 45);
pred_lagged = [zeros(1, N); pred_smooth(1:end-1, :)];
tilt        = ewm_mean(pred_lagged, 450);
pred        = pred_smooth - tilt;

# Cross-sectional Z-Score, [-4, 4] Clip, and Mandatory De-meaning
cs_mean = mean(pred, 2);
cs_std  = std(pred, 0, 2);
cs_std(cs_std == 0) = 1.0;

pred = (pred - cs_mean) ./ cs_std;
pred = max(min(pred, 4.0), -4.0);
pred = pred - mean(pred, 2);   % Mandatory zero row-mean

## 6. Diagnostic Report: Orthogonality, Weights & OOS Scores
# 6a. Pairwise Signal Correlation
corr_mat = abs(corr(X_val));
corr_mat(1:num_signals+1:end) = 0;
max_corr  = max(corr_mat(:));
mean_corr = mean(corr_mat(:));

fprintf('=== 1. SIGNAL UNIQUENESS (ORTHOGONALITY) ===\n');
fprintf('Total Signals: %d\n', num_signals);
fprintf('Mean Abs Pairwise Correlation: %.3f (Ideal: < 0.25)\n', mean_corr);
fprintf('Max Pairwise Correlation:      %.3f\n\n', max_corr);

# 6b. Feature Importance & Ridge Weights
imp = predictorImportance(gbdt_model);
fprintf('=== 2. SIGNAL IMPORTANCE & RIDGE WEIGHTS ===\n');
fprintf('%-14s | %-12s | %-12s | %-10s\n', 'Signal', 'GBDT_Imp', 'Ridge_Wt', 'Max_Corr');
fprintf('---------------------------------------------------------\n');
for k = 1:num_signals
    fprintf('%-14s | %12.6f | %12.6f | %10.3f\n', ...
        signal_names{k}, imp(k), ridge_w(k), max(corr_mat(k, :)));
end

# 6c. Out-of-Sample IC, IC Std & Annualized Sharpe
daily_ic = zeros(T_val, 1);
for t = 1:T_val
    valid_n = isfinite(pred(t, :)) & isfinite(Y_val(t, :));
    if sum(valid_n) > 2
        daily_ic(t) = corr(pred(t, valid_n)', Y_val(t, valid_n)');
    else
        daily_ic(t) = NaN;
    end
end
daily_ic = daily_ic(isfinite(daily_ic));

pnl = mean(pred .* Y_val, 2, 'omitnan');
ann_sharpe = (mean(pnl) / (std(pnl) + 1e-8)) * sqrt(252);

fprintf('\n=== 3. OUT-OF-SAMPLE VALIDATION METRICS ===\n');
fprintf('OOS Mean IC:      %.4f\n', mean(daily_ic));
fprintf('OOS IC Std:       %.4f\n', std(daily_ic));
fprintf('OOS Proxy Sharpe: %.4f\n', ann_sharpe);
fprintf('Max Abs Row Mean: %.2e\n', max(abs(mean(pred, 2))));


## ================= HELPER FUNCTIONS =================

function [X_3d, names] = build_13_signals(F1, F2, F3, F4, F5, F6)
    [T, N] = size(F1);
    R1 = csrank(F1); R2 = csrank(F2); R3 = csrank(F3);
    R4 = csrank(F4); R5 = csrank(F5); R6 = csrank(F6);

    s24 = F2 - F4;
    s12 = F1 - F2;
    s13 = F1 - F3;

    # Signal 12: 5-bar difference of s24
    diff5_s24 = [zeros(5, N); s24(6:end, :) - s24(1:end-5, :)];

    # Signal 13: Mean-reversion against 12-bar EMA of F1
    ema12_F1 = ewm_mean(F1, 12);

    names = {'d24', 'd12', 'd13', ...
             'm23_6', 'm24_5', 'p56', ...
             'r6', 'r5', ...
             'nl_14_6', 'nl_35_2', 'ratio_16', ...
             'ts_delta_24', 'ts_rev_13'};

    X_3d = zeros(T, N, 13);
    X_3d(:, :, 1)  = csrank(s24);
    X_3d(:, :, 2)  = csrank(s12);
    X_3d(:, :, 3)  = csrank(s13);
    X_3d(:, :, 4)  = (R2 - R3) .* R6;
    X_3d(:, :, 5)  = (R2 - R4) .* R5;
    X_3d(:, :, 6)  = R5 .* R6;
    X_3d(:, :, 7)  = R6;
    X_3d(:, :, 8)  = R5;
    X_3d(:, :, 9)  = csrank(F1 + F4) .* csrank(abs(F6));
    X_3d(:, :, 10) = tanh(2.0 * R3) .* (R5 - R2);
    X_3d(:, :, 11) = csrank(cs_robust_z(F1)) - R6;
    X_3d(:, :, 12) = csrank(diff5_s24);
    X_3d(:, :, 13) = -csrank(F1 - ema12_F1) .* R3;
end

function R = csrank(M)
    # Normalized cross-sectional rank in [-1, 1] matching pandas rank(axis=1)
    [T, N] = size(M);
    R = nan(T, N);
    for t = 1:T
        row = M(t, :);
        valid = isfinite(row);
        n = sum(valid);
        if n > 0
            rk = tiedrank(row(valid));
            R(t, valid) = (rk - 0.5 * (n + 1.0)) / (0.5 * max(n - 1.0, 1.0));
        end
    end
end

function Z = cs_robust_z(M)
    # Cross-sectional median/MAD robust z-score clipped to [-4, 4]
    med = median(M, 2, 'omitnan');
    mad_val = median(abs(M - med), 2, 'omitnan');
    mad_val(mad_val == 0) = 1.0;
    Z = (M - med) ./ (mad_val * 1.4826);
    Z = max(min(Z, 4.0), -4.0);
end

function E = ewm_mean(M, span)
    # Matches pandas .ewm(span=span, min_periods=1).mean()
    alpha = 2.0 / (span + 1.0);
    [T, N] = size(M);
    E = zeros(T, N);
    E(1, :) = M(1, :);
    for t = 2:T
        E(t, :) = alpha * M(t, :) + (1.0 - alpha) * E(t-1, :);
    end
end
