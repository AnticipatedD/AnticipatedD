# Biweekly Competition: Cross-Sectional Signal Forecasting — Season 2

> **These are the rules for biweekly SEASON 2.** They govern every Season 2 cycle and
> replace the Season 1 rules for them. Different files will be issued for future
> seasons; later seasons may differ — the target, the parameters, and the prize
> structure can all change between seasons.

## What is new in Season 2

- **A new target and new returns.** Season 2 has its own target, and the returns panel
  your signal is scored on is Season 2's own. A model fitted for Season 1 is not
  expected to keep its edge: retrain it and validate it again on the Season 2 package
  before you submit.
- **A new data package.** The Season 2 package is built from a new price feed and is
  the download on each Season 2 cycle's page. Its files are not comparable with Season
  1's — do not mix the two seasons' files, and train on the package of the season you
  enter.
- **A new city space.** Season 2 cities are not comparable with Season 1's: Season 1
  cities, including any city file or map you downloaded during Season 1, say nothing
  about novelty in Season 2.
- **Season 1 entries stay in Season 1.** Season 1 submissions are not entered into
  Season 2 and do not occupy its legacy pot. To compete, submit to a Season 2 cycle; a
  model you entered in Season 1 may be submitted again, retrained on the Season 2
  package. Season 1 cycles still in their live windows finish and settle under the
  Season 1 rules.

**Unchanged from Season 1:** the format, the prize formula, the per-cycle budget and
the 60/25/15 split, the quality threshold $K = 0.5$, the 20-submission cap, the entry
checks and the overfitting gate, the computation limits, and the kit's tools
(`runner.py`, `--full`, `--gauge-fix`).

## Format: seasons of biweekly contests

The Biweekly Competition runs as a **season of five biweekly contests**. Each biweekly is its own
contest with its own leaderboard and its own prize pot:

| | |
|---|---|
| **Submission window** | 2 weeks (boundaries land on the 1st and the 15th) |
| **Live scoring window** | 1 month from the moment submissions close |
| **Submissions per contest** | up to **20 per person**, reset every cycle |
| **Settlement** | at the end of the live window — one month after close, as the cycle after next is wrapping up |
| **Season** | 5 consecutive biweeklies under one target |

Because each live window runs a month, **two of them overlap at any time**: while
cycle N is being scored live, cycle N+1 is already accepting submissions.

### The legacy pot — why each cycle needs new alpha

When a cycle closes, **every submission that passed the entry checks moves into the
season's legacy pot**, regardless of how well it scored. Pot signals are no longer
prize-eligible, but they keep occupying their neighbourhood in signal space for the
rest of the season.

The practical consequence: **resubmitting a previous cycle's entry blocks itself**, as
it correlates ~1.0 with its own copy in the pot. Small perturbations of it fare barely
better. Every cycle rewards signals that are new relative to everything already
submitted this season — including your own earlier work.

When the season ends, its pot ends with it, and the next season starts with an empty
one. The fund may fold a selection of the season's signals into its own book, and a
later season may start against a correspondingly harder target.

## Prize

Each biweekly has its own pot, determined by the number of **quality signals** $Q$ that
cycle admits:

$$\text{Prize}(Q) = \begin{cases} 0 & \text{if } Q = 0 \\ 100 + 2400 \cdot q^{0.75} & \text{if } Q > 0 \end{cases}, \qquad q = \min(Q/13,\ 1)$$

The pot ranges from **\$0** (no quality signals) to **\$2,500 USD** per cycle at
saturation (13 quality signals). Payment is by stablecoin or bank transfer.

**Unspent budget rolls forward.** Each cycle has a prize budget of **\$2,500**; the
formula above sets what the cycle earns on its own. Whatever part of the budget is
not paid out in a cycle — because fewer quality signals were admitted, or fewer than
three places existed — **rolls into the pot of the season's next contest**. A cycle
that follows underspent ones can therefore pay out more than its own formula pot.

Note there is **no user-count term** — unlike comp5, the pot depends only on the alpha
actually found, not on how many people registered.

### Prize split within a cycle

The cycle's pot goes to the **top 3 signals in that cycle's quality set**, ranked by
Sharpe — the same split comp5 used:

| Place | Share |
|-------|-------|
| 1st   | 60%   |
| 2nd   | 25%   |
| 3rd   | 15%   |

Winners are drawn from the quality set, so you must be **both high-Sharpe and
sufficiently uncorrelated** with every higher-ranked winner. A signal that correlates
with a better one never enters the quality set and cannot win, however high its Sharpe.

**If a cycle admits fewer than three quality signals, only the stated shares are
paid to the places that exist**: with one quality signal the winner receives 60% of
the pot; with two, the top two receive 60% and 25%. Shares of places that do not
exist are not paid out in that cycle; the unpaid remainder rolls into the pot of
the season's next contest (see the budget note above).

### Quality signal selection

$Q$ is decided by a **greedy selection**:

1. Signals with a positive Sharpe are ranked by Sharpe.
2. Going down the list, a signal joins the quality set only if its time-averaged
   cross-sectional correlation with **every** signal already occupying space does
   not exceed **$K = 0.5$** in absolute value. This is comp5's threshold, carried
   over unchanged.
3. "Already occupying space" means: signals admitted earlier in the same ranking, and
   the season's legacy pot.

So a signal counts toward $Q$ — and grows the pot for everyone — only if it brings
genuinely new information, not a marginal variation on something already submitted.

---

## Objective

At each timestamp $i$ produce a cross-sectional signal $P(i) = (P_1(i), \ldots, P_J(i))$
over $J = 20$ assets, **cross-sectionally de-meaned**:

$$\sum_{j=1}^{J} P_j(i) = 0$$

You are scored on the Sharpe ratio of the portfolio return, **net of transaction cost**:

$$r(i) = \langle P(i-1),\ X(i) \rangle - c \cdot \lVert P(i-1) - P(i-2) \rVert_1, \qquad c = 5\ \text{bp}$$

$$U = \frac{\mathbb{E}[r(i)]}{\sqrt{\mathrm{Var}[r(i)]}}$$

Here $X(i) = (X_1(i), \ldots, X_J(i))$ is the vector of **AlphaNova proprietary
returns** at time $i$ — the shipped `returns` panel described in "What you are
predicting" below.

Official scoring may include additional execution-cost adjustments; official and
local numbers can differ systematically.

The cost term is not decorative. On Season 1's shipped example, rebalancing every hour
cost about **6% of the strategy's PnL volatility** — i.e. it dragged Sharpe down by
~0.06, which is larger than most edges you will find on this target. Smoothing
positions over time cut that to ~1% and is usually worth more than a slightly better
forecast. The demo notebook measures both on the Season 2 package.

## What you are predicting — read this twice

**The `returns` panel shipped with this competition is not raw price returns.** It is
a proprietary return series constructed by us; how it is constructed is deliberately
not disclosed, like the feature identities. This panel is what the backtest multiplies
your signal against, so it is the only "returns" that matter in this competition —
replicating it from outside data will not work and is not the game.

The **target** you fit is an **unknown target**: a forward-looking series built from
the returns panel, cross-sectionally z-scored and clipped at ±5. Its horizon and
exact construction are not disclosed. Treat it as unknown and let the data tell you
what predicts it.

Three things follow, and they are the whole character of this competition:

1. **The obvious first ideas score ~0.** The signals a quant reaches for first carry
   almost no edge on this target — not because they are wrong, but because the target
   is constructed so that they are not rewarded. (The demo notebook shows this
   happening to a textbook signal.)
2. **The bar is real.** Simple, marginal transformations of the shipped features
   carry little edge against this target. Expect to need structure: interactions,
   nonlinearities, cross-sectional relationships.
3. **Early rows of a month's panel can be exactly 0.0.** The construction needs a
   warm-up window of history before its first value. A flat stretch at the start of a
   panel is expected, not a data bug.

**The Season 2 target is not Season 1's.** It is a new series; as above, it is
cross-sectionally z-scored and clipped at ±5, and its horizon and exact construction are
not disclosed. Edges that paid in Season 1 may be worth little here.

---

## Competition Data

| File | What it is |
|---|---|
| `data/train/{NNN}.small.features.parquet` | 6 features × 20 assets, MultiIndex `(feature, ticker)` |
| `data/train/{NNN}.small.returns.parquet` | the hourly return series (see above) — what the backtest multiplies your signal against |
| `data/train/{NNN}.small.target.parquet` | the forward target, xs-z-scored, clipped ±5 (horizon and construction not disclosed) |
| `data/train/gauge/{NNN}.*.npy` | precomputed gauge matrices for `--gauge-fix` |
| `signal_db/signal_cities_ambient_coords.parquet` | city coordinates of existing signals — **not in this release**, see `signal_db/NO_CITIES_IN_THIS_RELEASE.txt` |

The six feature columns are anonymised on purpose: what each one measures, and how
each is computed, is not disclosed — in this package or anywhere else. They are
features, nothing more is stated. Treat them as `Feature.1`–`Feature.6` and let the
data tell you what they do.

Each period file is **cumulative**: period `NNN` contains all history up to that point,
so there is no need to concatenate periods (doing so duplicates early months heavily).

**The Season 2 package replaces Season 1's.** It is built from a new price feed. Season 1
and Season 2 files are different data: never combine them, and do not reuse a model's
fitted state across seasons.

**Ticker identities are not stable across periods.** `ticker.1` in period 001 is
unrelated to `ticker.1` in period 002 — each period is independently rotated. Your model
must learn cross-sectional structure, never per-ticker habits. Scoring is
invariant under the anonymization, so nothing you need for a good signal is lost.

### The data you get ends before the data you are scored on

Your package contains a fixed set of training periods. Internal scoring runs over a
longer history — including a portion you never receive — plus the live window of your
cycle. That hidden band is deliberate: it is the part of your score that cannot be
reproduced, tuned against, or reverse-engineered offline. Expect your local
validation numbers to differ from the leaderboard.

---

## Signal Cities and Novelty

Compressed to unit norm and gauge-fixed, a signal traces a path on a high-dimensional
sphere; the time-average of that path is a single point — the signal's **city**. It is a
compact summary of what your signal does on average.

- Cities **far apart** ⇒ the signals are very likely uncorrelated.
- Cities **close together** ⇒ they *may* be correlated (closeness is necessary but not
  sufficient).

Because the pot rewards the number of mutually uncorrelated signals, a city far from all
others is your best chance of being admitted. **Aim for at least 60° from the nearest
existing city.** (Measured on our side when you submit. This release ships no local
city database, so check yourself against the hourly map described below.)

`runner.py your_submission.py --gauge-fix` computes your city and reports the distance
to the nearest one in the shipped database. **This release ships no city
database** (see `signal_db/NO_CITIES_IN_THIS_RELEASE.txt`), so that distance reads `nan` locally — the
hourly map below is the one to steer by this cycle, and unzipping it into
`signal_db/` restores the local comparison. An **hourly refreshed** map of all season
cities is published on the contest page, so you can steer away from occupied
territory instead of discovering the collision after submitting.

**Season 2 has its own city space.** A Season 1 city is not comparable with a Season 2
one. The Season 2 map on the contest page shows Season 2 signals only; it fills as
Season 2 submissions are accepted.

**City coordinates are for offline navigation only.** They describe other signals, not
future returns; feeding them into your predictor will not help and may get your
submission flagged.

---

## Entry checks and the overfitting gate

Every submission runs through automated checks. Failures are reported as:

| Result | What went wrong | How to fix |
|--------|-----------------|------------|
| **PEEK_THE_FUTURE** | Data leakage — a future-looking operation such as `shift(-n)`, `rolling(center=True)`, or `bfill`/`backfill`. Those are examples, not the list: the reject set is broader, is not published, and is extended as new forms appear. Any transform whose output at hour *t* can depend on hours after *t* — centered kernels, zero-phase filters, wrap-arounds, in any library — is a violation, and ambiguous forms are reported as warnings. Passing the scan is not a certification: the no-look-ahead rule applies however a transform is expressed, and forms the scan cannot see are still violations — screening also continues after acceptance. | Remove every forward-looking transform — the examples here and any centered or zero-phase equivalent — from anywhere in the file, including inside `train()` and in comments. Make each feature causal by construction: at hour *t*, use only hours ≤ *t* (backward differences, trailing windows). The check is a line-by-line scan of the whole submission as the runner executes it, not of `predict()` alone — comments included, notebook magic and `!shell` lines excluded, because the runner strips those before your code runs. |
| **CODE_OUTSIDE_CLASS** | A top-level `def`, a non-Predictor class, or any other module-level statement that makes a call — bare or assigned, in any spelling, `with`, `for`, `try` — unless the call instantiates your Predictor subclass or is one of the permitted configuration calls. A statement with no call in it (`SEED = 42`, `X += 1`) only warns and does not fail. Passing this check is not a certification either: the rule is that no work happens at import time and no logic lives outside the class, however it is written, and forms the check cannot see are still violations. | Move helper functions and non-Predictor classes inside your subclass. `data = pd.read_parquet(...)` and `MODEL = build_model()` fail because what they call is not your Predictor subclass; `predictor = MyPredictor()` is fine. The full list of what may stay at module level is under Submission Format below. |
| **NOT_DEMEANED** | `predict()` output is not cross-sectionally de-meaned. Apply `pred.sub(pred.mean(axis=1), axis=0)` before returning. | Apply the de-mean before returning. The default `python runner.py your_submission.py` prints a `Cross-sectional de-mean check: PASS`/`FAIL` line, so you can catch this before submitting — note that neither `--full` nor `--gauge-fix` runs that check. |
| **TRAINING_TOO_SLOW** | One of two triggers. Either `train()` finished but took longer than 4 minutes — the elapsed time is compared with the limit once the call returns — or your model's combined cost per retrain (constructor + `train()` + `predict()`), multiplied across the overfitting gate's repeated retraining, is projected to exceed the gate's time ceiling; in that case the rejection happens before the gate runs, and — like every failed submission — it does not count against your cycle's 20-submission cap (only pending and accepted submissions do). | Cut boosting rounds/epochs, features, or training rows, and avoid recomputation, so `train()` finishes inside 4 minutes on CPU. For the gate trigger, the whole retrain loop must stay affordable: keep `__init__` light (it runs on every retrain) and reduce the combined constructor + train + predict cost — staying well inside the per-call limits keeps the gate comfortable. |
| **PREDICTION_TOO_SLOW** | `predict()` finished, but took longer than 60 seconds for a call — measured the same way as training. | Move heavy work into `train()` and keep `predict()` light — no refitting, no expensive per-call computation. |
| **OVERFITTING_FAILED** | Your signal did not beat its own shuffled-label null (see below). | Two causes look identical here. If you may be memorizing, cut complexity and regularize (shallower trees, stronger L2, fewer rounds) and prefer engineered, generalizable features. If the signal is simply weak, the answer is a stronger signal — see below. |
| **ROW_MASK_DETECTED** | Your predictions depend on where a row sits in the file rather than on the market history before it. After training, we call `predict()` again on the same rows with extra rows padded before and after them, so that every hour keeps its whole history and only its row number changes, and compare the forecasts for the same hours — direction and size — within a small tolerance. A model that reads the history returns the same forecasts; a model whose output is gated, scaled or switched by row number, block index or any other schedule keyed on position — counted from the start or from the end of the file — changes its answer when the rows shift, and is rejected. A forecast of the wrong shape for the padded rows counts as a changed answer. A model the check cannot measure — `predict()` raises on the padded rows, answers two identical calls differently, or cannot predict once reloaded from its saved state the way the daily scorer reloads it — is not accepted either; it is recorded as **Failed to run**, so seed any randomness, accept any frame of the same shape, and keep everything `predict()` needs in the trained object. | Remove every schedule that reads the row index: block masks, alternating on/off periods, per-block multipliers, position-keyed switches, counters of rows since the start of the frame. Anchor any state or normalisation you carry forward to timestamps and a finite look-back, not to the first row you were given: the start of the frame is not a fixed point, and a dependence on it is a position dependence under this rule. |
| **FLAT_PERIODS** | Your scored history contains too many flat rows (near-zero positions relative to your signal's own typical size), or a flat stretch that runs too long. A signal has to take positions to be evaluated; going flat for whole periods holds a rank without exposure and is rejected whatever the reason. | If your model abstains, make sure it does so only when the market gives it nothing to trade, and never on a fixed schedule. Check your own backtest output for runs of zero rows before submitting. |
| **NO_PREDICTOR_CLASS** | The file we received defines no class that subclasses `Predictor`, so there is nothing to run. An empty or placeholder `.py`, a notebook with no code in it, or the wrong file picked for upload, lands here too: the check reads the exact bytes your upload delivered, before any of its code runs. (A notebook file that is not notebook JSON at all, a 0-byte one included, is a SYNTAX_ERROR.) | Make sure you uploaded the file you meant to, and that it defines your subclass — `from predictor import Predictor`, then `class MyPredictor(Predictor):`, in a notebook inside one of its code cells. `python runner.py your_submission.py` loads it the same way: if the file you meant to upload loads there, the upload delivered something else, so upload that file again. |
| **SYNTAX_ERROR** | Python could not parse the file we received, so none of the structure rules could be checked and it could not run. A notebook is parsed as one script of its code cells, with lines starting with `%` or `!` removed as the runner removes them, so other notebook-only syntax fails here too. A file that is not UTF-8 text or that starts with a byte-order mark (BOM), or a notebook that is not valid notebook JSON (a 0-byte file included), lands here as well. | Fix the syntax error and resubmit: `python -m py_compile your_submission.py` names the line (for a notebook, run `python runner.py your_notebook.ipynb`). Keep notebook-only syntax out of the code cells, and save the file as UTF-8 without a BOM — `py_compile` accepts a BOM and other declared encodings, `python runner.py` refuses a BOM. Save a notebook that is not valid notebook JSON again from your notebook editor. Only if the file you meant to upload is UTF-8 without a BOM and `python runner.py` loads it did the upload deliver something else: then upload that file again. |
| **CANT_RUN** | Load failure, exception during train/predict, or another runtime error. A call cut off by the sandbox's absolute timeout lands here too (see Computation limits). | Reproduce with `python runner.py your_submission.py`, confirm your subclass implements `train()`/`predict()`, and declare extra dependencies via [PEP 723](https://peps.python.org/pep-0723/). Running out of memory also lands here — keep peak usage under 8 GB. |

### Row position and flat periods

Two further checks run on every submission, alongside the overfitting gate and on
the same trained model. The first offers your model the feature rows twice, once as
the pipeline does and once padded with extra rows before and after, so that every hour
keeps its whole history and only its row number changes, and requires the forecasts for
the same hours to agree — in direction and in size — within a tolerance; a model the
check cannot measure is recorded as Failed to run. The rule this enforces: a forecast is a
function of the market history before its hour, not of the row's position in the file.
A model that reads the history is unaffected — the padding sits more than a year before
any compared hour; anything anchored to a row number — a row counter, a block schedule,
a normalisation against the opening rows, a count of rows from the end — moves with the
rows and is a position dependence, whether or not it was meant as one.
The second reads your full scored history and rejects a signal that is flat — no
position, or a negligible one relative to its own typical size — for too much of it, or
for too long a stretch at a time. Both are reported as their own results (ROW_MASK_DETECTED,
FLAT_PERIODS) with the same mechanics as the other rejections. The exact offsets,
tolerances and caps are not published, and an accepted submission is not a
certification: a schedule the checks cannot see is still a violation, and screening
continues after acceptance.

### What the overfitting gate actually does

We retrain your model many times on data whose labels have been shuffled, so that no
predictable structure remains, and compare your real in-sample result against that null
distribution. A model that scores well on shuffled labels is memorizing, and is rejected.

Because the gate retrains your model repeatedly, its runtime is your `train()` +
`predict()` cost multiplied many times over. The gate's time budget scales with your
model's own measured cost up to an operator ceiling — a model that is inside the
per-call limits but too expensive for repeated retraining to finish within that
ceiling is rejected as TRAINING_TOO_SLOW before the gate runs; like every
failed submission, it does not count against your cycle's submission cap. Staying well inside the per-call limits keeps the gate
comfortable.

Because the gate retrains on variants whose time rows are permuted, your code —
`train()` and `predict()` alike — must tolerate data it would never see in a
real run. An `assert` about row order, about any quantity you derive from it
(autocorrelation, rolling statistics, trends, the ranking of computed
features), or about the target's values can fire on a shuffled retrain: the
features are permuted row-wise and the target is recomputed from the permuted
data, so its distribution differs from the real batch. A run in which any
single retrain crashes is refused rather than scored, because a data-dependent
crash could bias the test. What does stay fixed is the column layout: every
batch in a run carries the same columns, in the same order, as the unshuffled
one, so checks on column names are safe. Row counts are not: a batch is the
training slice of one particular month, so any length your local runs happen to
produce says nothing about what arrives here. Handle surprising values
tolerantly instead of asserting them away — for a sort you rely on, add a
stable tiebreaker rather than asserting that ties cannot happen. And note that
the local runner does not shuffle, so a clean local run cannot tell you whether
your code survives the gate.

**Be aware that this gate also rejects honest signals that are simply weak.** On a
target this hard a near-zero-edge model cannot beat its own null. This is a
property of a genuinely hard target, not a fault in your code or in our pipeline. If
you are rejected here, the answer is a stronger signal, not a bug report.

Passing the gate is a floor, not a score. Your ranking comes from Sharpe over the full
scored history plus your cycle's live window.

**The live window is a verification period, not a scoring period.** You are not ranked
on the live month by itself — it enters your Sharpe at its natural weight (about 2% of
the scored hours), and its real job is different: a signal built by de-obfuscating the
data gives itself away when its live behaviour diverges from its historical record.
Honest signals have nothing to do here but wait.

---

## Submission Format

A single **`.py` file** (or a notebook) containing your Predictor:

```python
from predictor import Predictor

class MyPredictor(Predictor):
    def train(self, features, target): ...
    def predict(self, features): ...   # must be cross-sectionally de-meaned
```

- Inherit from `Predictor` in `predictor.py`; do not modify the provided `.py` files.
- **All logic inside the class.** No top-level helper functions — including one
  written as `helper = lambda: ...`, which is a helper function however it is
  spelled — and no module-level statement that makes a call. These may stay at
  module level:
  - imports, a docstring or PEP 723 header, the Predictor subclass
  - `predictor = MyPredictor()` — any *other* call in that statement must itself be
    a permitted configuration call
  - **configuration calls**: `warnings.filterwarnings`/`simplefilter`,
    `logging.basicConfig`/`getLogger`, `pd.set_option`, `np.seterr` — written dotted,
    via `from … import …`, or as up to two chained method calls on the result of one
  - an **import guard**: `try: import x` / `except ImportError:` — raise a builtin
    exception (quoting the error with `str`/`repr`/`format` is fine), bind a
    call-free fallback, or import a fallback module
  - `if __name__ == "__main__":` — either operand order, and **no** `else:` (its
    `else` runs when your file is imported for scoring)
- Extra dependencies via [PEP 723](https://peps.python.org/pep-0723/) at the top of the
  file.

Pre-installed: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `pyarrow`.

### Local validation

```bash
python runner.py your_submission.py             # quick: latest period, holdout split
python runner.py your_submission.py --full      # full walk-forward across all periods
python runner.py your_submission.py --gauge-fix # full + city metrics (novelty: see signal_db/NO_CITIES_IN_THIS_RELEASE.txt)
```

This catches **NOT_DEMEANED** and **CANT_RUN** before you submit. It parses your
file and finds your Predictor subclass by the same rules as the entry checks, and it
refuses a byte-order mark as they do, so it also catches **NO_PREDICTOR_CLASS** and
**SYNTAX_ERROR** in the file on your disk. It reads the file in your system's default
encoding, while the entry checks require UTF-8, so save the file as UTF-8; if
`runner.py` then passes and the verdict still says otherwise, the upload delivered
something else.
Give a notebook a lower-case `.ipynb` suffix before running `runner.py` on it:
`runner.py` reads any other name as Python, while the entry checks accept the
suffix in any case.
Leakage and the rest of the code structure are checked server-side; the
overfitting gate runs after submission.

---

## Computation limits

- **No GPU.** `train()` must finish within **4 minutes** on a standard CPU.
- `predict()` must finish within **60 seconds** per call.
- Memory must stay under 8 GB.
- A method that exceeds its limit is normally allowed to finish; its elapsed time is then compared with the limit and the submission is rejected with the matching timing verdict. The sandbox also puts an absolute budget on each validation phase, only minutes above the per-method limits — a run that overshoots its limit by minutes rather than seconds hits that budget first, is cut off before returning, and reports **CANT_RUN** instead of a timing verdict. There is no partial credit for nearly making it.

---

## Timeline

**Season 2, Cycle 1 opens 15 November 2026, 00:00 UTC.**

| Cycle | Submission window | Live scoring |
|---|---|---|
| 1 | 15 Nov 2026 → 1 Dec 2026 | 1 Dec 2026 → 1 Jan 2027 |
| 2 | 1 Dec 2026 → 15 Dec 2026 | 15 Dec 2026 → 15 Jan 2027 |
| 3 | 15 Dec 2026 → 1 Jan 2027 | 1 Jan 2027 → 1 Feb 2027 |
| 4 | 1 Jan 2027 → 15 Jan 2027 | 15 Jan 2027 → 15 Feb 2027 |
| 5 | 15 Jan 2027 → 1 Feb 2027 | 1 Feb 2027 → 1 Mar 2027 |

Every window starts and ends at 00:00 UTC on the date shown; **the contest page is
authoritative**. Prizes settle shortly after each live window closes. After cycle 5 the
season ends; any later season, and its dates, will be announced.

## Important

- Do not modify any provided `.py` files or base-class signatures.
- All code must live inside your Predictor subclass.
- One account per person. Multi-account operations are detected and disqualified.
- Submissions with implausibly high validation Sharpe may be flagged for data
  reverse-engineering and disqualified.
