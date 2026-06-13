Objective
Develop a cross-sectional signal P(i)=(P1(i),…,PJ(i))P(i)=(P1​(i),…,PJ​(i)) at each timestamp ii that forecasts the relative returns of JJ assets. The signal must be cross-sectionally de-meaned at every timestamp:
Used outline a method for developing the signal, which includes several steps: identifying features, selecting a model, normalizing data, determining the training target based on relative returns, and establishing a loss function like ranking or MAE. Also, I have addressed scaling, missing data, and evaluated methods like IC and rank IC. It's important to backtest and consider regularization and constraints. I have provided equations in LaTeX and incorporate implementation details like minibatch processing and online updates.
Good — here's a practical, production-ready blueprint (math + engineering) I have used to built a cross-sectionally de-meaned forecasting signal
\(P(i)=\big(P_1(i),\dots,P_J(i)\big)\) at each timestamp \(i\) so that
\(\sum_{j=1}^J P_j(i)=0\) for every \(i\). I included objectives, loss choices, preprocessing, implementation details for streaming/real-time production, evaluation metrics, and common pitfalls.

Problem framing and target: Defined the relative return target at timestamp \(i\) for asset \(j\). A common choice is forward log return over horizon \(h\):
\[
  r_j(i+h) \;=\; \ln\!\left(\frac{S_j(i+h)}{S_j(i)}\right).
  \]
Converted to a cross-sectional target (demeaned forward return) so the model learns relative performance:
\[
  \tilde{r}_j(i) \;=\; r_j(i+h) \;-\; \frac{1}{J}\sum_{k=1}^J r_k(i+h).
  \]

Used \(\tilde{r}_j(i)\) as needed for supervised label.

Model output constraint (enforce cross-sectional zero-mean)
You want model outputs \(P_j(i)\) that satisfy \(\sum_j P_j(i)=0\). Two robust approaches:

A. Predict unconstrained scores and subtract cross-sectional mean (recommended, simple and stable)
Let raw model output be \(s_j(i)\). Then defined
\[
  P_j(i) \;=\; s_j(i) \;-\; \frac{1}{J}\sum_{k=1}^J s_k(i).
  \]

This enforces \(\sum_j P_j(i)=0\) exactly.

B. Model with explicit constraint layer
Design a final layer that maps unconstrained vector \(s(i)\in\mathbb{R}^J\) to the subspace orthogonal to \(\mathbf{1}\) by projecting:
\[
  P(i) \;=\; \left(I - \frac{1}{J}\mathbf{1}\mathbf{1}^\top\right) s(i),
  \]

where \(\mathbf{1}=(1,\dots,1)^\top\). This is identical to mean-subtraction but expressed as a linear operator (useful if integrating in linear algebra pipelines).
Loss choices (optimize for cross-sectional predictive power)

Mean Squared Error (MSE) on demeaned labels:
\[
  \mathcal{L}_{\text{MSE}} \;=\; \frac{1}{N_{\text{train}}}\sum_{(i,j)}\bigl(P_j(i) - \tilde{r}_j(i)\bigr)^2.
  \]
Rank-based losses (often better for relative trading signals):

Pairwise ranking (hinge or logistic): penalize pairs with wrong relative order.
Listwise loss (e.g., softmax cross-entropy over ranks):
\[
    \mathcal{L}_{\text{softmax}}(i) \;=\; -\sum_{j=1}^J y_j(i)\log\frac{e^{P_j(i)}}{\sum_k e^{P_k(i)}},
    \]

where \(y_j(i)\) is a target distribution derived from \(\tilde r_j(i)\) (e.g., soft labels).

Regularize for stability: add \(L_2\) on model weights. Optionally add temporal smoothing penalty on \(P(i)\) changes.

Feature engineering & normalization

Cross-sectional features: price momentum, recent returns, volume/turnover, volatility, factor exposures.

Time-series features: exponentially weighted moving averages, z-scored returns over lookbacks.

Normalize features to avoid leakages:

Per-asset time-series scaling (e.g., divide by asset volatility).

Cross-sectional standardization at each \(i\) if features vary widely across assets:
\[
    \hat{x}_j(i) = \frac{x_j(i) - \mu_x(i)}{\sigma_x(i)},
    \quad
    \mu_x(i)=\frac{1}{J}\sum_{k} x_k(i).
    \]

Missing data: forward-fill where sensible, or mask features and let the model receive masks.
Architecture & training recommendations
Model families: gradient-boosted trees (LightGBM/CatBoost) for tabular; MLP/Gated MLP / Transformer for cross-asset interactions; Siamese/ranking networks for pairwise ranking.

If you use a neural net and J varies or is large, treat the model to predict per-asset score from per-asset features (shared weights) and then apply cross-sectional projection:

Per-asset encoder \(f_\theta\) maps asset features \(x_j(i)\) to scalar \(s_j(i)=f_\theta(x_j(i))\).

Aggregate de-mean to get \(P_j(i)=s_j(i)-\overline{s(i)}\).

Mini-batching: batch over timestamps; compute cross-sectional mean within each timestamp in the batch.

Avoid leakage: ensure forward returns \(r_j(i+h)\) are strictly out-of-sample for training timestamp \(i\).

Evaluation metrics for cross-sectional signals

Information Coefficient (IC):
\[
  \text{IC}(i)=\operatorname{corr}\bigl(P(i),\tilde r(i)\bigr),
  \]
report mean IC and t-stat.

Rank IC (Spearman correlation) — robust to scaling.

Hit rate (fraction where sign matches).

Simulated PnL/backtest: apply simple normalized portfolio construction (see section 8).

Stability: autocorrelation of per-asset signals, turnover.

Portfolio construction (mapping signal to positions)

Simple cross-sectional long-short with leverage neutralization:

Normalize signal to unit standard deviation across assets at each \(i\):
\[
    \bar P_j(i)=\frac{P_j(i)}{\sqrt{\frac{1}{J}\sum_k P_k(i)^2}}.
    \]

Positions \(w_j(i)=\bar P_j(i)\) produce dollar-neutral exposure because \(\sum_j w_j(i)=0\).

If anyone needs market-neutral (beta neutral), regress \(w(i)\) on factor exposures and orthogonalize.

Control turnover by applying shrinkage (exponential smoothing) to signals:
\[
  P^{\text{sm}}_j(i)=\alpha P_j(i) + (1-\alpha) P^{\text{sm}}_j(i-1).
  \]

Backtesting and transaction-cost aware evaluation

Include realistic transaction costs and slippage.

Execute portfolio rebalancing frequency consistent with your signal horizon \(h\) (e.g., daily/weekly).

Metrics: annualized return, Sharpe, max drawdown, turnover, information ratio vs benchmark.

Practical production implementation (scalable)

Data pipeline:

Feature store (per-asset features) fed into model inference service.

Compute per-asset scores in parallel (shared encoder) on GPUs/CPU cluster.

Compute cross-sectional mean reduction (a single all-reduce) to subtract mean: scalable across many assets using a reduce operation (MapReduce / parameter server / all-reduce in Ray or Spark).

Real-time constraints:

Use streaming feature updates, compute \(s_j(i)\) online, then run a reduce step to get \(\overline{s(i)}\), then broadcast subtraction to produce \(P_j(i)\).

Keep numerical precision stable (use float64 for reduction if J large).

Monitoring:

Live IC, turnover, distributional shift of features and P values; alert when mean of raw scores drifts.

Re-training cadence:

Retrain model on rolling window, validate on holdout. Consider online-learning updates for model weights.
Here is the direct, minimal solution you can implement immediately.

Produce unconstrained scores

For each timestamp \(i\) and asset \(j\), compute an unconstrained score \(s_j(i)\) from your model/encoder:

\[
  s_j(i)=f_\theta\bigl(x_j(i)\bigr),
  \]

where \(x_j(i)\) are the features for asset \(j\) at time \(i\). 

Cross-sectionally de-mean the scores (exactly)

Compute the cross-sectional mean at time \(i\):
\[
  \overline{s}(i)=\frac{1}{J}\sum_{k=1}^J s_k(i).
  \]

Define the final signal vector \(P(i)=(P_1(i),\dots,P_J(i))\) by subtracting that mean:

\[
  P_j(i)=s_j(i)-\overline{s}(i)\qquad\text{for }j=1,\dots,J.
  \]

This enforces

\[
  \sum_{j=1}^J P_j(i)=0\quad\text{for every }i.
  \]

(Equivalent linear operator form)
\[
P(i)=\Bigl(I - \frac{1}{J}\mathbf{1}\mathbf{1}^\top\Bigr)\,s(i),
\]

where \(\mathbf{1}=(1,\dots,1)^\top\).

Optional scale / portfolio mapping (recommended)

Normalize to control risk/turnover:

\[
  \tilde P_j(i)=\frac{P_j(i)}{\sqrt{\frac{1}{J}\sum_{k=1}^J P_k(i)^2}}.
  \]

Use positions \(w_j(i)=\tilde P_j(i)\) for a dollar-neutral long-short portfolio (still \(\sum_j w_j(i)=0\)).

Training target (if you need supervised labels)

Use demeaned forward returns as labels:

\[
  \tilde r_j(i)=r_j(i+h)-\frac{1}{J}\sum_{k=1}^J r_k(i+h),
  \]

and train to minimize e.g. MSE between \(P_j(i)\) and \(\tilde r_j(i)\) or a rank-based loss.

That's the complete, production-safe answer: compute per-asset scores, subtract the cross-sectional mean at each timestamp, (optionally) scale, and use those \(P_j(i)\).

Step definitions used:
\[
s(i)=\bigl(s_1(i),s_2(i),s_3(i)\bigr),
\qquad
\overline{s}(i)=\frac{1}{3}\sum_{j=1}^3 s_j(i),
\qquad
P_j(i)=s_j(i)-\overline{s}(i).
\]

Optional scaling:
\[
\text{RMS}_i=\sqrt{\tfrac{1}{3}\sum_{j=1}^3 P_j(i)^2},\qquad
\tilde P_j(i)=\frac{P_j(i)}{\text{RMS}_i}.
\]

Raw scores \(s\) (rows = timestamps \(i=1\ldots5\); columns = assets \(j=1\ldots3\)):
\[
s=\begin{bmatrix}
0.10 & 0.05 & -0.02\\[4pt]
0.20 & 0.15 & 0.10\\[4pt]
-0.05 & 0.00 & 0.10\\[4pt]
0.50 & -0.10 & 0.00\\[4pt]
0.02 & 0.02 & 0.02
\end{bmatrix}.
\]
Cross-sectional means \(\overline{s}(i)\) for each timestamp:
\[
\overline{s}=
\begin{bmatrix}
0.0433 \\[4pt]
0.1500 \\[4pt]
0.0167 \\[4pt]
0.1333 \\[4pt]
0.0200
\end{bmatrix}
\quad\text{(values rounded to 4 d.p.)}.
\]

Demeaned signals \(P = s - \overline{s}\) (each row sums to exactly \(0\)):
\[
P=
\begin{bmatrix}
0.0567 & 0.0067 & -0.0633\\[4pt]
0.0500 & 0.0000 & -0.0500\\[4pt]
-0.0667 & -0.0167 & 0.0833\\[4pt]
0.3667 & -0.2333 & -0.1333\\[4pt]
0.0000 & 0.0000 & 0.0000
\end{bmatrix}.
\]

Check: for each \(i\), \(\sum_{j=1}^3 P_j(i)=0\).

Row RMS (for scaling) and scaled signals \(\tilde P\):

Row RMS values
\[
  \text{RMS} = \bigl[0.0492,\; 0.0408,\;0.0624,\;0.2625,\;0.0000\bigr].
  \]

Scaled signals \(\tilde P_j(i)=P_j(i)/\text{RMS}_i\) (rows with zero RMS produce undefined entries — interpreted as all-zero signal):

\[
  \tilde P=
  \begin{bmatrix}
  1.1514 & 0.1355 & -1.2868\\[4pt]
  1.2247 & 0.0000 & -1.2247\\[4pt]
  -1.0690 & -0.2673 & 1.3363\\[4pt]
  1.3970 & -0.8890 & -0.5080\\[4pt]
  \text{undefined (all zeros)} & \text{undefined} & \text{undefined}
  \end{bmatrix}.
  \]
Good — I walked through a concrete numeric example with \(J=3\) assets and \(i=1,\dots,5\) timestamps. I compute raw scores \(s_j(i)\), the cross-sectional mean \(\overline{s}(i)\), the demeaned signal \(P_j(i)=s_j(i)-\overline{s}(i)\), and an optional row-wise RMS scaling to unit risk.
Step definitions used:
\[
s(i)=\bigl(s_1(i),s_2(i),s_3(i)\bigr),
\qquad
\overline{s}(i)=\frac{1}{3}\sum_{j=1}^3 s_j(i),
\qquad
P_j(i)=s_j(i)-\overline{s}(i).
\]

Optional scaling:
\[
\text{RMS}_i=\sqrt{\tfrac{1}{3}\sum_{j=1}^3 P_j(i)^2},\qquad
\tilde P_j(i)=\frac{P_j(i)}{\text{RMS}_i}.
\]
Raw scores \(s\) (rows = timestamps \(i=1\ldots5\); columns = assets \(j=1\ldots3\)):
\[
s=\begin{bmatrix}
0.10 & 0.05 & -0.02\\[4pt]
0.20 & 0.15 & 0.10\\[4pt]
-0.05 & 0.00 & 0.10\\[4pt]
0.50 & -0.10 & 0.00\\[4pt]
0.02 & 0.02 & 0.02
\end{bmatrix}.
\]
Cross-sectional means \(\overline{s}(i)\) for each timestamp:
\[
\overline{s}=
\begin{bmatrix}
0.0433 \\[4pt]
0.1500 \\[4pt]
0.0167 \\[4pt]
0.1333 \\[4pt]
0.0200
\end{bmatrix}
\quad\text{(values rounded to 4 d.p.)}.
\]
Demeaned signals \(P = s - \overline{s}\) (each row sums to exactly \(0\)):
\[
P=
\begin{bmatrix}
0.0567 & 0.0067 & -0.0633\\[4pt]
0.0500 & 0.0000 & -0.0500\\[4pt]
-0.0667 & -0.0167 & 0.0833\\[4pt]
0.3667 & -0.2333 & -0.1333\\[4pt]
0.0000 & 0.0000 & 0.0000
\end{bmatrix}.
\]
Check: for each \(i\), \(\sum_{j=1}^3 P_j(i)=0\).
Row RMS (for scaling) and scaled signals \(\tilde P\):¹
Row RMS values:

\[
  \text{RMS} = \bigl[0.0492,\; 0.0408,\;0.0624,\;0.2625,\;0.0000\bigr].
  \]
Scaled signals \(\tilde P_j(i)=P_j(i)/\text{RMS}_i\) (rows with zero RMS produce undefined entries — interpreted as all-zero signal):
\[
  \tilde P=
  \begin{bmatrix}
  1.1514 & 0.1355 & -1.2868\\[4pt]
  1.2247 & 0.0000 & -1.2247\\[4pt]
  -1.0690 & -0.2673 & 1.3363\\[4pt]
  1.3970 & -0.8890 & -0.5080\\[4pt]
  \text{undefined (all zeros)} & \text{undefined} & \text{undefined}
  \end{bmatrix}.
  \]

Python snippets to (A) compute angular distance between your signal’s city and every existing city in data/signal_cities.parquet, (B) check the 60° novelty rule, and (C) an iterative workflow + practical tips to increase novelty without using city coordinates as model inputs.

Key math (angular distance)

If two city coordinates are unit vectors \(u,v\in\mathbb{R}^3\) on the sphere, the angular distance \(\theta\) (in degrees) is

\[
  \theta \;=\; \arccos\!\bigl(u\cdot v\bigr)\times\frac{180}{\pi}.
  \]

Equivalently, to test the 60° threshold use the dot product:

\[
  \theta \ge 60^\circ \quad\Longleftrightarrow\quad u\cdot v \le \cos(60^\circ) = 0.5.
  \]

So we can often avoid computing the expensive \(\arccos\) and just check whether the dot product is \(\le 0.5\).

If city coordinates are given as latitude/longitude \((\phi,\lambda)\), convert to unit 3D vectors first:
\[
u=\begin{bmatrix}
\cos\phi\cos\lambda\\[4pt]
\cos\phi\sin\lambda\\[4pt]
\sin\phi
\end{bmatrix},
\]

where \(\phi\) = latitude in radians and \(\lambda\) = longitude in radians.

Python snippet — read Parquet, compute angular distances, report nearest
(assumes cities are stored either as 3 columns x,y,z OR lat/lon; adapt if your schema differs)

```python
import numpy as np
import pandas as pd

# Path to parquet
path = "data/signal_cities.parquet"
df = pd.read_parquet(path)

# Inspect columns: either expect ['x','y','z'] as unit vectors or ['lat','lon'] in degrees
print(df.columns)

# Example: if df has 'x','y','z'
if set(['x','y','z']).issubset(df.columns):
    existing = df[['x','y','z']].to_numpy(dtype=float)
    # ensure unit normalization (numerical safety)
    norms = np.linalg.norm(existing, axis=1, keepdims=True)
    existing = existing / np.maximum(norms, 1e-12)

# Or if df has 'lat','lon' in degrees:
elif set(['lat','lon']).issubset(df.columns):
    lat = np.deg2rad(df['lat'].to_numpy(dtype=float))
    lon = np.deg2rad(df['lon'].to_numpy(dtype=float))
    existing = np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
else:
    raise ValueError("Unexpected schema: expected ('x','y','z') or ('lat','lon').")

# Your city: either as unit vector or lat/lon. Example: provide as unit vector:
# Replace this with your actual city coordinates
your_city = np.array([0.1, 0.9, 0.4], dtype=float)
your_city = your_city / np.linalg.norm(your_city)

# Compute dot products with all existing cities
dots = existing.dot(your_city)

# Clip for numerical safety
dots = np.clip(dots, -1.0, 1.0)

# Angular distances in degrees (optional)
angles_deg = np.degrees(np.arccos(dots))

# Find nearest
idx_min = np.argmin(angles_deg)
min_angle = angles_deg[idx_min]
min_dot = dots[idx_min]

print(f"Nearest city index: {idx_min}, angular distance = {min_angle:.3f} deg, dot = {min_dot:.6f}")

# Check 60 degree novelty criterion
threshold_deg = 60.0
ok = min_angle >= threshold_deg
if ok:
    print("Novelty check PASSED: nearest city >= 60° away.")
else:
    print("Novelty check FAILED: nearest city closer than 60° (consider iterating).")

# Optionally list all cities within 60 deg
close_mask = angles_deg < threshold_deg
print(f"{close_mask.sum()} existing cities are within {threshold_deg}°.")
```

```
# AmonRa_final_submission_V1_4

This repository contains a single script `AmonRa_final_submission_V1_4.py` that computes the contest prize function
and provides CI-friendly test output.

Prize definition (defaults):
- \(U\): number of new users (non-negative integer)
- \(Q\): number of quality signals (non-negative integer)
  \[
  \operatorname{Prize}(U,Q)=
  \begin{cases}
  0, & Q=0,\\[6pt]
  2000 + 48000\cdot (U\cdot Q)^{0.75}, & Q>0.
  \end{cases}
  \]

Rounding: results are rounded to nearest dollar with "round half-up" semantics (ties of .5 round up).

## Files
- `AmonRa_final_submission_V1_4.py` — main script containing implementations, CLI, and unit tests.
- `README.md` — this file.

## Usage

Run interactively:
```bash
python AmonRa_final_submission_V1_4.py 100 10
```
