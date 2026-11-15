"""Walk-forward backtesting engine for the Bi-weekly competition.

Two modes:

  walkforward_backtest  — Quick validation on the latest period.
                          Holds out min(800, 20%) rows for validation.
                          Returns Sharpe and equity curve.

  walkforward_full      — Full walk-forward across ALL periods. For each period,
                          holds out min(800, 20%) rows for validation.
                          Concatenates PnL across all validation sets.
                          Optionally gauge-fixes predictions to compute city, IC,
                          and other geometric metrics (requires shipped gauge
                          matrices in data/train/gauge/).

                          NOTE: These are validation-set statistics computed on
                          training data. Performance on the held-out test set
                          (which we run internally after submission) will differ.

Each period file already contains all history up to that point — there is
no need to concatenate multiple period files.

Contestants should NOT modify this file.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import list_periods, load_period
from evaluation import (
    backtest,
    check_cross_sectional_demean,
    returns_to_equity,
    sharpe_ratio,
)
from predictor import Predictor

MAX_VAL_ROWS = 800
VAL_FRACTION = 0.2


def ic_dispersion_corr(ic, dispersion) -> float:
    """Pearson correlation between the IC series and the target dispersion
    over the rows where both are finite — the definition in METRICS.md and
    the one the board computes. NaN when fewer than two rows pair up or
    either side is constant over them (a correlation with a constant is
    undefined, not 0). The two series are aligned by position and to the
    shorter one."""
    ic = np.asarray(ic, dtype=float)
    dispersion = np.asarray(dispersion, dtype=float)
    n = min(len(ic), len(dispersion))
    ic, dispersion = ic[:n], dispersion[:n]
    both = np.isfinite(ic) & np.isfinite(dispersion)
    if int(both.sum()) < 2:
        return float("nan")
    ic, dispersion = ic[both], dispersion[both]
    if ic.std() == 0.0 or dispersion.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(ic, dispersion)[0, 1])


def compute_val_rows(n: int) -> int:
    """Validation holdout size: min(MAX_VAL_ROWS, 20% of n)."""
    return min(MAX_VAL_ROWS, int(VAL_FRACTION * n))


def _describe(exc: BaseException) -> str:
    """"<type>: <message>", even for an exception whose __str__ raises — the
    predictor's own class can define that."""
    try:
        text = str(exc)
    except Exception:
        text = "<unprintable message>"
    return f"{type(exc).__name__}: {text}"


def _failure_text(phase: str, exc: Exception) -> tuple[str, str]:
    """(error, error_phase) as _phase_failure words them, without keeping the exception."""
    return f"{phase}: {_describe(exc)}", phase


def _phase_failure(phase: str, exc: Exception) -> dict:
    """walkforward_backtest's result when the predictor raised in `phase`.

    `phase` is the call that raised: init (the constructor),
    generate_features, train or predict. The message names it with the
    exception's type, and `exception` is the exception itself. demean_check
    is True because nothing was wrong with the predictions — there were
    none — and demo.py / demo_engineered.py read the key on every result.
    """
    return {"sharpe": 0.0, "demean_check": True, "error": f"{phase}: {_describe(exc)}",
            "error_phase": phase, "exception": exc}

def _plain_predictions(predictions, n_rows: int):
    """predict()'s output as a plain float64 DataFrame and None, or None and
    what is wrong with it: not a DataFrame, not one row per input row,
    duplicate column labels or a row index of mixed types (which the scoring
    step's parquet refuses), not real numbers (complex, datetime and
    timedelta columns; values that do not convert). Our evaluation then computes on the plain frame only: on the
    returned object, each of these raised inside evaluation.py, where nothing
    said the predictor was at fault — and so did a DataFrame subclass's own
    methods (reviews on #168 and #175). Anything the returned object itself
    raises while being read is the problem too."""
    try:
        if not isinstance(predictions, pd.DataFrame):
            return None, TypeError(f"predict() returned {type(predictions).__name__}, not a pandas DataFrame")
        if len(predictions) != n_rows:
            return None, ValueError(
                f"predict() returned {len(predictions)} rows but features has {n_rows} rows; "
                f"predict() must return exactly one prediction per input row"
            )
        # What the scoring step's parquet write refuses, refused here too, so
        # it is named at the first gate (review on #175).
        if predictions.columns.has_duplicates:
            return None, ValueError("predict() returned duplicate column labels")
        if not isinstance(predictions.index, pd.MultiIndex) and pd.api.types.infer_dtype(
            predictions.index, skipna=True
        ) in ("mixed", "mixed-integer"):
            return None, TypeError("predict() returned a row index of mixed types")
        unreal = sorted({str(dtype) for dtype in predictions.dtypes if getattr(dtype, "kind", "O") in "cmM"})
        if unreal:
            return None, TypeError(f"predict() returned columns that are not real numbers: {', '.join(unreal)}")
        try:
            values = pd.DataFrame.to_numpy(predictions, dtype=np.float64, na_value=np.nan)
        except (TypeError, ValueError) as e:
            return None, TypeError(f"predict() returned values that are not numbers: {e}")
        return pd.DataFrame(values, index=predictions.index, columns=predictions.columns), None
    except Exception as e:  # noqa: BLE001 — a DataFrame subclass's own code
        return None, e


def walkforward_backtest(
    predictor_class: type,
    data_dir: str = "data/train",
    universe: str = "small",
    max_train_time: float = 240.0,
    max_predict_time: float = 60.0,
    verbose: bool = True,
) -> dict:
    """Run walk-forward backtest for a Predictor subclass.

    Uses the latest available period. Holds out min(800, 20%) rows for
    validation. The predictor trains on the rest and predicts on the
    validation portion.

    Each period file already contains all history up to that period's test start,
    so no concatenation of periods is needed.

    Args:
        predictor_class: A class that inherits from Predictor.
        data_dir: Directory containing parquet files.
        universe: Universe type ('small', 'top20vol', 'top50').
        max_train_time: Maximum training time in seconds (default 4 min).
        max_predict_time: Maximum prediction time in seconds (default 60s).
        verbose: Whether to print progress.

    Returns:
        Dictionary with:
            - 'sharpe': Sharpe ratio on the validation portion
            - 'train_rows': Number of training rows
            - 'val_rows': Number of validation rows
            - 'predictions': Validation predictions DataFrame
            - 'returns': Validation returns DataFrame
            - 'pf_returns': Portfolio returns on validation set
            - 'equity': Cumulative equity curve
            - 'demean_check': Whether de-mean check passed
    """
    if not (
        isinstance(predictor_class, type) and issubclass(predictor_class, Predictor)
    ):
        raise TypeError(
            f"{predictor_class} must be a class that inherits from Predictor"
        )

    # Use the latest period (most data)
    periods = list_periods(data_dir, universe)
    if not periods:
        raise FileNotFoundError(
            f"No training data found in '{data_dir}' for universe '{universe}'. "
            f"Check that the data directory exists and contains files matching "
            f"*.{universe}.features.parquet"
        )
    latest_period = periods[-1]

    if verbose:
        print(f"Available periods: {len(periods)} ({periods[0]} to {periods[-1]})")
        print(f"Using latest period: {latest_period}")

    data = load_period(latest_period, data_dir, universe)
    n_total = len(data["features"])
    n_val = compute_val_rows(n_total)
    n_train = n_total - n_val

    if verbose:
        print(f"Total rows: {n_total}")
        print(f"Training rows: {n_train}")
        print(f"Validation rows: {n_val} (min({MAX_VAL_ROWS}, {VAL_FRACTION:.0%}))")
        print("=" * 60)

    # Split
    train_features = data["features"].iloc[:n_train]
    train_target = data["target"].iloc[:n_train]
    train_returns = data["returns"].iloc[:n_train]

    val_features = data["features"].iloc[n_train:]
    val_target = data["target"].iloc[n_train:]
    val_returns = data["returns"].iloc[n_train:]

    # Train. Every call into the predictor is guarded: its own exception
    # comes back as error/error_phase naming the call, the way train() and
    # predict() always did. The constructor and generate_features() used to
    # sit outside, so the scoring sandbox's outermost catch, which cannot
    # tell whose code raised, reported their exceptions as platform failures
    # (DEV-1555).
    try:
        predictor = predictor_class()
    except Exception as e:
        print(f"Constructor FAILED: {_describe(e)}")
        return _phase_failure("init", e)
    # Parity with the internal runner (review of PR #34): predictors that
    # implement generate_features() get it called on the FULL feature frame
    # before training, exactly as the scoring pipeline does after upload.
    # Without this, such submissions validate differently locally than they
    # score remotely — the gap this local kit exists to prevent.
    # The hasattr check runs your code too (a generate_features property or a
    # __getattr__), so it sits inside the same try as the call.
    try:
        if hasattr(predictor, "generate_features"):
            predictor.generate_features(data["features"])
    except Exception as e:
        print(f"generate_features FAILED: {_describe(e)}")
        return _phase_failure("generate_features", e)

    t0 = time.time()
    try:
        predictor.train(train_features, train_target)
    except Exception as e:
        print(f"Training FAILED: {_describe(e)}")
        return _phase_failure("train", e)
    train_time = time.time() - t0

    if train_time > max_train_time:
        print(f"WARNING: Training took {train_time:.1f}s (limit: {max_train_time}s)")
    elif verbose:
        print(f"Trained in {train_time:.1f}s")

    # Predict on validation features
    t0 = time.time()
    try:
        predictions = predictor.predict(val_features)
    except Exception as e:
        print(f"Prediction FAILED: {_describe(e)}")
        return _phase_failure("predict", e)
    # The contract, checked before our own evaluation touches the output, and
    # the plain float frame it then computes on: a None, a numpy array, a
    # frame short of rows, non-numbers or a DataFrame subclass otherwise
    # raised inside evaluation.py, unattributed, for a fault that is the
    # predictor's (reviews on #168 and #175).
    predictions, problem = _plain_predictions(predictions, len(val_features))
    if problem is not None:
        print(f"Prediction FAILED: {_describe(problem)}")
        return _phase_failure("predict", problem)
    predict_time = time.time() - t0

    if predict_time > max_predict_time:
        print(f"WARNING: Prediction took {predict_time:.1f}s (limit: {max_predict_time}s)")
    elif verbose:
        print(f"Predicted in {predict_time:.1f}s")

    dm_ok = check_cross_sectional_demean(predictions)

    # Compute backtest once — `sharpe_ratio` would call backtest again
    # internally, double-charging the alignment / ffill work for no
    # reason. Same shape as the equivalent inline calc in walkforward_full.
    pf_ret = backtest(predictions, val_returns)
    val_sharpe = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
    equity = returns_to_equity(pf_ret)

    if verbose:
        print(f"\nValidation Sharpe: {val_sharpe:.4f}")
        print(f"Equity: start={equity.iloc[0]:.4f}, end={equity.iloc[-1]:.4f}")
        print(f"Max drawdown from peak: {(equity - equity.cummax()).min():.4f}")

    return {
        "sharpe": val_sharpe,
        "train_rows": n_train,
        "val_rows": n_val,
        "predictions": predictions,
        "returns": val_returns,
        "pf_returns": pf_ret,
        "equity": equity,
        "demean_check": dm_ok,
    }


def walkforward_full(
    predictor_class: type,
    data_dir: str = "data/train",
    universe: str = "small",
    gauge_fix: bool = True,
    max_train_time: float = 240.0,
    max_predict_time: float = 60.0,
    verbose: bool = True,
) -> dict:
    """Full walk-forward across all periods with optional gauge-fixing.

    For each period:
        1. Hold out min(800, 20%) rows for validation
        2. Train a fresh predictor on the rest
        3. Predict on the validation portion
        4. Compute PnL on the validation portion
        5. If gauge_fix=True, apply shipped gauge matrices to get gauge-fixed vectors

    After all periods:
        - Concatenate PnL → compute Sharpe
        - If gauge_fix: concatenate gauge-fixed vectors → compute IC, city,
          concentration, compression_loss, and compare against published cities

    NOTE: These are validation-set statistics computed on training data.
    Performance on the held-out test set will differ.

    Args:
        predictor_class: A class that inherits from Predictor.
        data_dir: Directory containing parquet files.
        universe: Universe type ('small', 'top20vol', 'top50').
        gauge_fix: If True, compute gauge-fixed vectors and geometric metrics.
            Requires gauge matrices in {data_dir}/gauge/.
        max_train_time: Maximum training time in seconds (default 4 min).
        max_predict_time: Maximum prediction time in seconds (default 60s).
        verbose: Whether to print progress.

    Returns:
        Dictionary with:
            - 'sharpe': Sharpe ratio across all validation sets
            - 'n_periods': Number of periods processed
            - 'n_observations': Total validation rows across all periods
            - 'pf_returns': Concatenated portfolio returns
            - 'equity': Cumulative equity curve
            - 'per_period': Dict of per-period results (sharpe, n_val)

        If gauge_fix=True, also includes:
            - 'ic': Mean IC (cross-sectional correlation with target)
            - 'ic_std': IC standard deviation
            - 'ic_dispersion_corr': Correlation between the IC series and the
              shipped target dispersion ({data_dir}/{period}.{universe}.dispersion
              .parquet) over the rows where both are finite — the board's
              definition; NaN when a period ships without the file
            - 'concentration': Trajectory tightness on S^18
            - 'compression_loss': IC - city[0]
            - 'city': Unit vector in R^19 (the city)
            - 'gauge_fixed': Concatenated gauge-fixed vectors (n_obs, 19)
            - 'city_comparison': DataFrame comparing against published cities
            - 'city_novelty_deg': Angular distance to nearest city (degrees)
            - 'nearest_city': ID of nearest city in the database
            - 'global_novelty_deg': NaN (computed internally after submission)
            - 'nearest_signal': N/A (computed internally after submission)
    """
    if not (
        isinstance(predictor_class, type) and issubclass(predictor_class, Predictor)
    ):
        raise TypeError(
            f"{predictor_class} must be a class that inherits from Predictor"
        )

    periods = list_periods(data_dir, universe)

    if verbose:
        print(f"Walk-forward full: {len(periods)} periods, gauge_fix={gauge_fix}")
        print(f"Validation: min({MAX_VAL_ROWS}, {VAL_FRACTION:.0%}) per period")
        print("NOTE: These are validation-set statistics. Test-set results will differ.")
        print("=" * 60)

    # Preload gauge matrices if needed
    gauge_dir = Path(data_dir) / "gauge"
    if gauge_fix:
        from city_tools import compare_to_cities, compute_city, compute_ic

        if not gauge_dir.exists():
            raise FileNotFoundError(
                f"Gauge directory {gauge_dir} not found. "
                "Gauge matrices are required for gauge_fix=True."
            )

    all_pnl = []
    # The first period whose train()/predict() failed — its error and phase
    # only, not the exception (whose traceback would pin that period's
    # frames): reported if no period produces a result at all.
    first_failure = None
    all_gf = []
    all_target_dispersion = []
    dispersion_missing = False
    per_period = {}

    for i, period in enumerate(periods):
        t_period = time.time()

        data = load_period(period, data_dir, universe)
        n_total = len(data["features"])
        n_val = compute_val_rows(n_total)
        n_train = n_total - n_val

        if n_val == 0:
            if verbose:
                print(f"  [{period}] SKIP (no validation rows)")
            continue

        train_features = data["features"].iloc[:n_train]
        train_target = data["target"].iloc[:n_train]
        val_features = data["features"].iloc[n_train:]
        val_returns = data["returns"].iloc[n_train:]

        # Train. A raising constructor or generate_features() propagates, as
        # it always did here: skipping those periods would report a Sharpe
        # over the survivors and hide the rest (review on #168). This local
        # tool is not the sandbox's path — validate uses walkforward_backtest.
        predictor = predictor_class()
        # Same parity hook as the single-period path above (PR #34 review).
        if hasattr(predictor, "generate_features"):
            predictor.generate_features(data["features"])
        t0 = time.time()
        try:
            predictor.train(train_features, train_target)
        except Exception as e:
            if verbose:
                print(f"  [{period}] Training FAILED: {_describe(e)}")
            first_failure = first_failure or _failure_text("train", e)
            continue
        train_time = time.time() - t0

        if train_time > max_train_time:
            if verbose:
                print(f"  [{period}] WARNING: Training took {train_time:.1f}s")

        # Predict
        t0 = time.time()
        try:
            predictions = predictor.predict(val_features)
        except Exception as e:
            if verbose:
                print(f"  [{period}] Prediction FAILED: {_describe(e)}")
            first_failure = first_failure or _failure_text("predict", e)
            continue
        predictions, problem = _plain_predictions(predictions, len(val_features))
        if problem is not None:
            if verbose:
                print(f"  [{period}] Prediction FAILED: {_describe(problem)}")
            first_failure = first_failure or _failure_text("predict", problem)
            continue

        # Target dispersion: computed inside gauge block below to stay row-aligned with gf
        predict_time = time.time() - t0

        if predict_time > max_predict_time:
            if verbose:
                print(f"  [{period}] WARNING: Prediction took {predict_time:.1f}s")

        # PnL
        pf_ret = backtest(predictions, val_returns)
        period_sharpe = float(pf_ret.mean() / pf_ret.std()) if pf_ret.std() > 0 else 0.0
        all_pnl.append(pf_ret)

        # Gauge-fix
        if gauge_fix:
            gauge_path = gauge_dir / f"{period}.{universe}.gauge.npy"
            if not gauge_path.exists():
                if verbose:
                    print(f"  [{period}] WARNING: No gauge file, skipping gauge-fix")
            else:
                gauge_matrices = np.load(str(gauge_path))
                pred_vals = predictions.values
                # Ensure ticker ordering: ticker.1, ticker.2, ..., ticker.N
                if hasattr(predictions, 'columns'):
                    cols = predictions.columns.tolist()
                    numeric_order = sorted(cols, key=lambda c: int(c.split('.')[-1]))
                    if cols != numeric_order:
                        pred_vals = predictions[numeric_order].values

                # Cross-sectionally demean (should already be, but ensure)
                pred_vals = pred_vals - pred_vals.mean(axis=1, keepdims=True)

                n_gauge = gauge_matrices.shape[0]
                if n_gauge != n_val:
                    # Align to shorter
                    n_use = min(n_gauge, pred_vals.shape[0])
                    gauge_matrices = gauge_matrices[:n_use]
                    pred_vals = pred_vals[:n_use]

                from city_tools import compute_gauge_fixed
                gf = compute_gauge_fixed(pred_vals, gauge_matrices)
                all_gf.append(gf)

                # Target dispersion: the shipped per-period series, trimmed to
                # the gauge rows. Not the row std of the target — the target is
                # cross-sectionally z-scored, so that std is a constant and says
                # nothing; the shipped series is the z-score's denominator, the
                # quantity METRICS.md defines and the board correlates against.
                if data.get("dispersion") is None:
                    dispersion_missing = True
                else:
                    disp_val = data["dispersion"].iloc[n_train:].to_numpy(dtype=float)
                    all_target_dispersion.append(disp_val[:gf.shape[0]])

        per_period[period] = {
            "sharpe": period_sharpe,
            "n_val": n_val,
        }

        elapsed = time.time() - t_period
        if verbose:
            print(
                f"  [{period}] {n_val} val rows, "
                f"Sharpe={period_sharpe:+.4f}, "
                f"{elapsed:.1f}s"
            )

        del data, predictor

    # Aggregate
    if not all_pnl:
        if verbose:
            print("No periods produced results.")
        result = {"sharpe": 0.0, "n_periods": 0, "n_observations": 0}
        # Every period's predictor raised: say so. Without an error key a
        # crashing model read as a valid zero-Sharpe run, and runner.py's
        # `results.get("error")` check wrote it to results.csv (review on
        # #168). A period that merely had no validation rows is not a failure.
        if first_failure is not None:
            result["error"], result["error_phase"] = first_failure
        return result

    concat_pnl = pd.concat(all_pnl, ignore_index=True)
    overall_sharpe = float(concat_pnl.mean() / concat_pnl.std()) if concat_pnl.std() > 0 else 0.0
    equity = returns_to_equity(concat_pnl)

    result = {
        "sharpe": overall_sharpe,
        "n_periods": len(per_period),
        "n_observations": len(concat_pnl),
        "pf_returns": concat_pnl,
        "equity": equity,
        "per_period": per_period,
    }

    if gauge_fix and all_gf:
        concat_gf = np.concatenate(all_gf, axis=0)
        ic_stats = compute_ic(concat_gf)
        city = compute_city(concat_gf)

        # IC-dispersion correlation, over the rows where both sides are finite
        # (ic_dispersion_corr above). NaN — and said so — when a period was
        # shipped without its dispersion file: a number from fewer periods than
        # the IC covers would be a different metric, not this one.
        if dispersion_missing:
            ic_disp = float("nan")
            if verbose:
                print("  NOTE: a period has no dispersion parquet — ic_dispersion_corr is NaN "
                      "(re-download the data package; the file ships beside each target)")
        else:
            ic_disp = ic_dispersion_corr(concat_gf[:, 0], np.concatenate(all_target_dispersion))

        result.update({
            "ic": ic_stats["ic_mean"],
            "ic_std": ic_stats["ic_std"],
            "ic_dispersion_corr": ic_disp,
            "concentration": ic_stats["concentration"],
            "compression_loss": ic_stats["ic_mean"] - city[0],
            "city": city,
            "gauge_fixed": concat_gf,
        })

        # Compare against published cities → city novelty
        try:
            city_comparison = compare_to_cities(city)
            result["city_comparison"] = city_comparison
            result["city_novelty_deg"] = float(city_comparison.iloc[0]["angular_distance_deg"])
            result["nearest_city"] = str(city_comparison.iloc[0]["signal_id"])
        except Exception:
            result["city_novelty_deg"] = float('nan')
            result["nearest_city"] = "N/A"

        # Global novelty is not computed here — it requires loading all signals'
        # full gauge-fixed trajectories and aligning timesteps. It is computed
        # internally after submission via compute_all_novelty().
        result["global_novelty_deg"] = float('nan')
        result["nearest_signal"] = "N/A (computed after submission)"

    if verbose:
        print("=" * 60)
        print(f"Overall Sharpe: {overall_sharpe:.4f}")
        print(f"Periods: {len(per_period)}, Observations: {len(concat_pnl)}")
        if gauge_fix and all_gf:
            print(f"IC: {result['ic']:.4f} (std={result['ic_std']:.4f})")
            print(f"IC-dispersion corr: {result['ic_dispersion_corr']:.4f}")
            print(f"Concentration: {result['concentration']:.4f}")
            print(f"Compression loss: {result['compression_loss']:.4f}")
            print(f"City novelty: {result['city_novelty_deg']:.1f} deg (nearest: {result['nearest_city']})")
            print(f"Global novelty: computed after submission")
        print()
        print("NOTE: These are validation-set statistics. Test-set results will differ.")

    return result
