# Competition 6: Biweekly Cross-Sectional Signal Forecasting

Forecast a deliberately hard, undisclosed cross-sectional target.
Read `COMPETITION.md` for the rules and prize structure — this file gets you running.

## Setup

Python 3.12 is recommended.

```bash
pip install -r requirements.txt
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
```

## Quick start

```bash
# See the data and a worked example end to end
python demo.py

# Quick check on the latest period (holdout split)
python runner.py demo_engineered.py

# Full walk-forward across every period — this is the number to iterate on
python runner.py demo_engineered.py --full

# Full walk-forward + city / novelty metrics
python runner.py demo_engineered.py --gauge-fix
```

`demo.py` is the minimal example; `demo_engineered.py` is the stronger starting point
(slow feature windows, turnover control) and is
validated against the same overfitting gate your submission must pass before the
package ships — see "Entry checks and the overfitting gate" in `COMPETITION.md`.
Both are starting points, not a solution: bring the signal yourself.

## Writing your predictor

```python
# /// script
# dependencies = ["catboost>=1.2"]     # optional extra deps, PEP 723
# ///

import numpy as np
import pandas as pd

from predictor import Predictor


class MyPredictor(Predictor):
    def train(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        # features: MultiIndex columns (feature_name, ticker), 6 x 20
        # target:   20 columns, the forward target (xs-z, clipped +-5)
        ...

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        preds = ...                                   # (n_rows, 20)
        return preds.sub(preds.mean(axis=1), axis=0)  # de-meaning is mandatory
```

Rules that get submissions rejected if broken:

- everything lives **inside the class** (imports at module level are fine);
- predictions are **cross-sectionally de-meaned** (each row sums to zero);
- no future-looking operations (`shift(-n)`, `bfill`, centred rolling windows…);
- `train()` under 4 minutes, `predict()` under 60 seconds, no GPU.

## Three things that decide your score

1. **The target is not disclosed.** It is constructed so that the obvious first
   ideas earn ~0 — see the "What you are predicting" section of `COMPETITION.md`.
2. **Turnover costs 5 bp.** Position churn is taxed on every rebalance; smoothing your
   signal over time is often worth more than a better forecast.
3. **Novelty gates the prize.** Your signal must stay uncorrelated with everything
   already submitted this season. `--gauge-fix` tells you how close you are.

## Files

| File | Purpose |
|---|---|
| `COMPETITION.md` | rules, prizes, format, submission checks |
| `METRICS.md` | what each reported metric means |
| `predictor.py` | base class — inherit, do not modify |
| `demo.py`, `demo_engineered.py` | worked examples |
| `runner.py` | local validation CLI |
| `demo.ipynb` | guided walkthrough notebook (start here) |
| `requirements.txt` | local environment dependencies |
| `walkforward.py`, `data_loader.py`, `evaluation.py`, `city_tools.py` | machinery used by the runner |
| `data/train/` | training periods (features, returns, target) + gauge matrices |
| `signal_db/` | city coordinates of existing signals, for novelty checks |

Questions: the contest discussion page.
