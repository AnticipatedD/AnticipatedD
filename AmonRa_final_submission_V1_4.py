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
