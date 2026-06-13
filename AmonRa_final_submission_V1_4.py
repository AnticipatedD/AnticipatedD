```
#!/usr/bin/env python3
"""
AmonRa_final_submission_V1_4.py

Compute Prize(U,Q) with half-up rounding and provide CI-friendly JUnit XML output.

Prize(U,Q) = 0                     if Q == 0
            = 2000 + 48000*(U*Q)**0.75   if Q > 0

Features:
- compute_prize_round_half_up: float math, half-up via floor(x+0.5)
- compute_prize_decimal_half_up: Decimal math, exact half-up quantize
- CLI with JSON output (--json), verbosity, examples, and test modes
- Unit tests accessible via --run-tests
- CI integration: --ci-junit <path> runs tests and writes JUnit XML to the path

Usage examples:
    python AmonRa_final_submission_V1_4.py 100 10
    python AmonRa_final_submission_V1_4.py --decimal 100 10 --json
    python AmonRa_final_submission_V1_4.py --run-tests
```
    python AmonRa_final_submission_V1_4.py --ci-junit results.xml
"""
from __future__ import annotations
import sys
import math
import json
import logging
import time
from decimal import Decimal, getcontext, ROUND_HALF_UP
import xml.etree.ElementTree as ET

# Configure logger (module-level)
logger = logging.getLogger("AmonRa_prize")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(_handler)


def compute_prize_round_half_up(U: int | float, Q: int | float) -> int:
    """
    Fast float-based computation. Rounds half-up via int(math.floor(x + 0.5)).

    Returns:
        Prize rounded to nearest dollar, .5 always rounded up.
    """
    U = float(U)
    Q = float(Q)
    if Q == 0:
        logger.debug("Q is zero -> prize 0 (float path)")
        return 0
    product = U * Q
    prize = 2000.0 + 48000.0 * (product ** 0.75)
    result = int(math.floor(prize + 0.5))
    logger.debug("Float prize computed: raw=%s -> rounded=%d", prize, result)
    return result


def compute_prize_decimal_half_up(U: int, Q: int, prec: int = 50) -> int:
    """
    Decimal-based computation with exact half-up rounding.

    Uses (U*Q)**0.75 = ((U*Q)**3)**(1/4) and computes 4th root via two sqrt calls.

    Parameters:
        U, Q: non-negative integers (counts). If Q == 0 returns 0.
        prec: Decimal precision (default 50).
    Returns:
        Prize rounded to nearest dollar (half-up), as int.
    """
    U_i = int(U)
    Q_i = int(Q)

    if Q_i == 0:
        logger.debug("Q is zero -> prize 0 (decimal path)")
        return 0

    getcontext().prec = max(prec, 28)
    U_d = Decimal(U_i)
    Q_d = Decimal(Q_i)
```

    product = U_d * Q_d
    if product == 0:
        prize = Decimal('2000')
    else:
        prod_cubed = product ** 3
        fourth_root = prod_cubed.sqrt().sqrt()
        prize = Decimal('2000') + Decimal('48000') * fourth_root

    prize_q = prize.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    result = int(prize_q)
    logger.debug("Decimal prize computed: raw=%s -> quantized=%s", prize, prize_q)
    return result


def _print_examples():
    examples = [
        (100, 10),
        (10, 1),
        (0, 5),
        (0, 0),
        (1, 1),
    ]
    print("Examples (float half-up):")
    for U, Q in examples:
        print(f"  U={U:>5}, Q={Q:>5} -> ${compute_prize_round_half_up(U, Q):,}")

    print("\nExamples (Decimal half-up):")
    for U, Q in examples:
        print(f"  U={U:>5}, Q={Q:>5} -> ${compute_prize_decimal_half_up(U, Q):,}")


def _cli_main(argv):
    import argparse
    parser = argparse.ArgumentParser(description="Compute contest prize Prize(U,Q).")
    parser.add_argument("U", type=int, nargs="?", help="Number of new users (U).")
    parser.add_argument("Q", type=int, nargs="?", help="Number of quality signals (Q).")
    parser.add_argument("--decimal", action="store_true", help="Use Decimal-based computation (exact half-up).")
    parser.add_argument("--prec", type=int, default=50, help="Decimal precision for Decimal method (default 50).")
    parser.add_argument("--json", action="store_true", help="Output result as JSON {\"U\":.., \"Q\":.., \"prize\":..}.")
    parser.add_argument("--examples", action="store_true", help="Print examples and exit.")
    parser.add_argument("--run-tests", action="store_true", help="Run built-in unit tests and exit (console output).")
    parser.add_argument("--ci-junit", type=str, default=None, help="Run tests and write JUnit XML to provided path, exit nonzero on failure.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging (DEBUG).")
    args = parser.parse_args(argv)

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # CI JUnit mode: run tests, write XML, exit with nonzero on test failures
    if args.ci_junit is not None:
        outpath = args.ci_junit
        logger.info("Running unit tests (CI mode), output JUnit XML to '%s'...", outpath)
        rc = _run_tests_write_junit(outpath)
        if rc == 0:
            logger.info("Tests passed; JUnit XML written to '%s'.", outpath)
        else:
            logger.error("Tests failed; JUnit XML written to '%s'.", outpath)
        return rc
```
    if args.run_tests:
        logger.info("Running unit tests (console mode)...")
        import unittest
        loader = unittest.defaultTestLoader
        tests = loader.loadTestsFromTestCase(TestPrizeCompute)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(tests)
        return 0 if result.wasSuccessful() else 2

    if args.examples or args.U is None or args.Q is None:
        _print_examples()
        return 0

    if args.decimal:
        prize = compute_prize_decimal_half_up(args.U, args.Q, prec=args.prec)
    else:
        prize = compute_prize_round_half_up(args.U, args.Q)

    if args.json:
        out = {"U": args.U, "Q": args.Q, "prize": prize}
        print(json.dumps(out))
    else:
        print(f"Prize(U={args.U}, Q={args.Q}) = ${prize:,}")
    return 0


# ----------------------------
# Unit tests
# ----------------------------
import unittest


class TestPrizeCompute(unittest.TestCase):
    def test_zero_Q_float(self):
        self.assertEqual(compute_prize_round_half_up(100, 0), 0)

    def test_zero_Q_decimal(self):
        self.assertEqual(compute_prize_decimal_half_up(100, 0), 0)

    def test_basic_examples_float(self):
        expected = int(math.floor(2000.0 + 48000.0 * ((100 * 10) ** 0.75) + 0.5))
        self.assertEqual(compute_prize_round_half_up(100, 10), expected)
        self.assertEqual(compute_prize_round_half_up(0, 5), 2000)

    def test_basic_examples_decimal(self):
        a = compute_prize_round_half_up(10, 1)
        b = compute_prize_decimal_half_up(10, 1)
        self.assertEqual(a, b)

    def test_half_up_behavior_float(self):
        raw = 1234.5
        rounded = int(math.floor(raw + 0.5))
        self.assertEqual(rounded, 1235)

    def test_half_up_behavior_decimal(self):
        getcontext().prec = 28
        x = Decimal('1000.5')
        q = x.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
        self.assertEqual(int(q), 1001)
```
```

    def test_large_values_decimal(self):
        res = compute_prize_decimal_half_up(10**6, 10**3, prec=80)
        self.assertIsInstance(res, int)
        self.assertGreaterEqual(res, 2000)

    def test_consistency_small(self):
        for U in [0, 1, 2, 3, 10]:
            for Q in [0, 1, 2, 5]:
                a = compute_prize_round_half_up(U, Q)
                b = compute_prize_decimal_half_up(U, Q)
                self.assertEqual(a, b)


# ----------------------------
# JUnit XML writer for CI
# ----------------------------
def _run_tests_write_junit(output_path: str) -> int:
    """
    Run unittest tests from TestPrizeCompute and write a simple JUnit XML file.

    Returns 0 on success (all tests passed), nonzero otherwise.
    """
    import unittest
    loader = unittest.defaultTestLoader
    suite = loader.loadTestsFromTestCase(TestPrizeCompute)
    runner = unittest.TextTestRunner(verbosity=2)
    # Capture results by running tests via TestResult
    result = runner.run(suite)

    # Construct JUnit XML
    tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(getattr(result, "skipped", []))  # python 3.8+ may include skipped

    testsuite = ET.Element("testsuite", {
        "name": "AmonRa_final_submission_V1_4.Tests",
        "tests": str(tests),
        "failures": str(failures),
        "errors": str(errors),
        "skipped": str(skipped),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "time": "0"
    })

    # For each test record, add testcase entries.
    # result.failures and result.errors are lists of (testcase, traceback)
    failed_cases = {str(t[0]) for t in result.failures}
    error_cases = {str(t[0]) for t in result.errors}

    # We do not have a complete list of passed tests from TextTestRunner result object,
    # so we iterate through the suite to reconstruct test names.
    for test in _iter_tests(suite):
        tc_name = str(test)
        # attempt to parse classname and method name
        classname = test.__class__.__name__
        method = getattr(test, "_testMethodName", None) or ""
        testcase = ET.SubElement(testsuite, "testcase", {
            "classname": classname,
            "name": method or tc_name,
            "time": "0"
```
            "time": "0"
        })
        if tc_name in failed_cases:
            # find the traceback text
            tb = _find_traceback_text(result.failures, tc_name)
            failure = ET.SubElement(testcase, "failure", {"message": "failure"})
            failure.text = tb
        elif tc_name in error_cases:
            tb = _find_traceback_text(result.errors, tc_name)
            error = ET.SubElement(testcase, "error", {"message": "error"})
            error.text = tb
        # skipped not handled specially here (few tests); can extend if needed

    tree = ET.ElementTree(testsuite)
    try:
        tree.write(output_path, encoding="utf-8", xml_declaration=True)
    except Exception as e:
        logger.error("Failed to write JUnit XML to '%s': %s", output_path, e)
        return 2

    # Return 0 if all tests passed
    return 0 if (failures == 0 and errors == 0) else 1


def _iter_tests(suite):
    """
    Yield individual TestCase instances from a TestSuite (recursive).
    """
    try:
        for t in suite:
            if isinstance(t, unittest.TestSuite):
                yield from _iter_tests(t)
            else:
                yield t
    except Exception:
        # If any unexpected structure, just attempt to yield suite itself
        yield suite


def _find_traceback_text(list_of_tuples, test_str):
    """
    Given result.failures or result.errors (list of (testcase, traceback)),
    return traceback text matching test_str, or a short message.
    """
    for tc, tb in list_of_tuples:
        if str(tc) == test_str:
            return tb
    return "Traceback unavailable"


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    try:
        rc = _cli_main(sys.argv[1:]) or 0
        sys.exit(rc)
    except Exception as e:
        logger.error("Unhandled exception: %s", e)
        sys.exit(2)

$ python runner.py AmonRa_final_submission_v1_4 and call when --gauge-fix is requested:
```
# utils/novelty.py
import numpy as np
import pandas as pd

def latlon_to_unit(lat_deg, lon_deg):
    """Convert degrees to 3D unit vector (float32)."""
    lat = np.deg2rad(lat_deg.astype(np.float32))
    lon = np.deg2rad(lon_deg.astype(np.float32))
    cx = np.cos(lat) * np.cos(lon)
    cy = np.cos(lat) * np.sin(lon)
    cz = np.sin(lat)
    return np.stack([cx, cy, cz], axis=1).astype(np.float32)

def load_existing_cities(parquet_path="data/signal_cities.parquet"):
    """
    Load existing city coordinates.
    Accepts either columns ['x','y','z'] (unit vectors) or ['lat','lon'] in degrees.
    Returns ndarray shape (N,3), dtype float32, normalized to unit length.
    """
    df = pd.read_parquet(parquet_path)
    if set(['x','y','z']).issubset(df.columns):
        arr = df[['x','y','z']].to_numpy(dtype=np.float32)
    elif set(['lat','lon']).issubset(df.columns):
        arr = latlon_to_unit(df['lat'].to_numpy(), df['lon'].to_numpy())
    else:
        raise ValueError("Unexpected schema in signal_cities.parquet. Expect ['x','y','z'] or ['lat','lon'].")
    # normalize (safety)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12).astype(np.float32)
    return (arr / norms).astype(np.float32)

def check_novelty(your_city_vec, existing_cities, threshold_dot=0.5, verbose=True):
    """
    your_city_vec: array-like shape (3,) or (1,3) -- can be lat/lon (tuple) or vector.
    existing_cities: ndarray (N,3) unit vectors.
    threshold_dot: default 0.5 corresponds to 60 degrees (cos 60 = 0.5).
    Returns: dict with keys: pass (bool), max_dot, min_angle_deg, nearest_index, n_within_threshold
    """
    u = np.asarray(your_city_vec, dtype=np.float32).reshape(3,)
    # if input is lat/lon pair: detect by magnitude > 1.2 or < -1.2; but prefer explicit call.
    if np.linalg.norm(u) < 1.5:  # assume vector already or lat/lon in degrees would be large magnitudes -> ambiguous
        # normalize vector
        un = u / max(np.linalg.norm(u), 1e-12)
    else:
        # unlikely branch; prefer user to pass a unit vector or lat/lon externally
        un = u / max(np.linalg.norm(u), 1e-12)
    # vectorized dot product (fast)
    dots = existing_cities.astype(np.float32).dot(un.astype(np.float32))
    # numerical safety
    dots = np.clip(dots, -1.0, 1.0)
    max_dot = float(dots.max())
    nearest_idx = int(dots.argmax())
    # Only compute angle for the nearest item (cheap)
    min_angle_deg = float(np.degrees(np.arccos(max_dot)))
    # count how many existing cities are closer than threshold (dot > threshold_dot)
    n_close = int((dots > threshold_dot).sum())

    result = {
        "pass": max_dot <= float(threshold_dot),
        "max_dot": max_dot,
        "min_angle_deg": min_angle_deg,
        "nearest_index": nearest_idx,
        "n_within_threshold": n_close
    }
    if verbose:
        status = "PASS" if result["pass"] else "FAIL"
        print(f"[novelty] {status} | nearest idx={nearest_idx} | angle={min_angle_deg:.3f}° | dot={max_dot:.6f} | n_within_60deg={n_close}")
    return result
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
