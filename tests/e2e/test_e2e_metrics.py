"""End-to-end tests for the D7 statistical metrics."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import FitFailedWarning

from purgedcv import (
    CombinatorialPurgedCV,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_full,
    effective_n_trials,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from purgedcv.optuna_integration import TrialSharpeRecorder


@pytest.mark.e2e
def test_user_story_cpcv_to_psr_pipeline() -> None:
    """End-to-end: generate backtest paths from CPCV, derive per-path
    return series, and compute PSR and DSR per path. This is the canonical
    workflow the library is built around."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=120, freq="D"))
    evalu = pred + pd.Timedelta(days=2)
    cv = CombinatorialPurgedCV(
        n_splits=6,
        n_test_groups=2,
        prediction_times=pred,
        evaluation_times=evalu,
    )
    rng = np.random.default_rng(99)
    X = rng.standard_normal((120, 3))  # noqa: N806
    y = X @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(120) * 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FitFailedWarning)
        paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
    # Per-path "returns" = prediction error (toy definition for the test).
    per_path_returns = paths - y[np.newaxis, :]
    per_path_psr = [
        probabilistic_sharpe_ratio(row[np.isfinite(row)], benchmark_skill=0.0)
        for row in per_path_returns
    ]
    assert len(per_path_psr) == 5  # n_paths for N=6, K=2
    for psr in per_path_psr:
        assert 0.0 <= psr <= 1.0
    # Also verify DSR on first path (multiple-comparison correction with 5 paths).
    first_path_finite = per_path_returns[0][np.isfinite(per_path_returns[0])]
    dsr = deflated_sharpe_ratio(first_path_finite, n_trials=5, var_sharpe=0.01**2)
    assert 0.0 <= dsr <= 1.0


@pytest.mark.e2e
def test_user_story_min_track_record_length() -> None:
    """User Story: a quant researcher wants to know how many observations
    are needed to prove their observed Sharpe of 0.7 beats a benchmark of
    0.5 at the 95% confidence level, given normal return moments."""
    required_n = min_track_record_length(
        observed_sharpe=0.7,
        target_sharpe=0.5,
        alpha=0.05,
        skew=0.0,
        kurtosis=3.0,
    )
    assert required_n > 0  # a positive sample size is required
    # A smaller gap (0.21 vs 0.2) requires far more data than a larger gap (0.7 vs 0.5)
    required_n_small_gap = min_track_record_length(
        observed_sharpe=0.21,
        target_sharpe=0.2,
        alpha=0.05,
        skew=0.0,
        kurtosis=3.0,
    )
    assert required_n_small_gap > required_n


@pytest.mark.e2e
def test_user_story_optuna_recorder_feeds_deflated_sharpe() -> None:
    """User Story: a researcher tunes a strategy with Optuna, records each
    trial's Sharpe through TrialSharpeRecorder, then deflates the champion's
    Sharpe by the variance and count of all trials searched. The recorder
    needs no Optuna import; a SimpleNamespace stands in for the frozen trial.
    """
    from types import SimpleNamespace

    rng = np.random.default_rng(31)
    recorder = TrialSharpeRecorder()
    for s in rng.normal(0.8, 0.4, 120):
        recorder(study=None, trial=SimpleNamespace(value=float(s), user_attrs={"sharpe": float(s)}))

    assert recorder.n_trials() == 120
    var_sharpe = recorder.var_sharpe()
    assert var_sharpe > 0

    champion_returns = rng.normal(0.001, 0.01, 504)
    diag = deflated_sharpe_ratio_full(
        champion_returns, n_trials=recorder.n_trials(), var_sharpe=var_sharpe
    )
    # The diagnostics expose why the deflation landed where it did.
    assert diag.n_trials == 120
    assert diag.n_obs == 504
    assert diag.var_sharpe == pytest.approx(var_sharpe)
    assert diag.sr_star == pytest.approx(np.sqrt(var_sharpe) * diag.expected_max_z)
    assert diag.dsr == pytest.approx(deflated_sharpe_ratio(champion_returns, 120, var_sharpe))


@pytest.mark.e2e
def test_user_story_effective_n_trials_rescues_dsr() -> None:
    """User Story: a 6000-trial TPE search is heavily autocorrelated. Feeding
    the raw count to DSR over-deflates it toward zero; the effective count
    collapses to a few hundred and DSR becomes informative again."""
    rng = np.random.default_rng(123)
    # Correlated trial-Sharpe path (random walk stands in for a TPE trajectory).
    trial_sharpes = np.cumsum(rng.standard_normal(6000)) * 0.01
    n_eff = effective_n_trials(trial_sharpes)
    assert 1 <= n_eff < 6000

    returns = rng.normal(0.0015, 0.01, 504)
    dsr_raw = deflated_sharpe_ratio(returns, n_trials=6000, var_sharpe=0.02**2)
    dsr_eff = deflated_sharpe_ratio(returns, n_trials=n_eff, var_sharpe=0.02**2)
    # Fewer effective trials -> lower deflated benchmark -> larger DSR.
    assert dsr_eff >= dsr_raw


@pytest.mark.e2e
def test_user_story_min_track_record_length_unreachable() -> None:
    """User Story: a researcher checks how long a track record must run to
    prove a Sharpe they have not yet beaten. The honest answer is 'no finite
    length suffices', returned as infinity rather than an exception."""
    import math

    unreachable = min_track_record_length(
        observed_sharpe=0.3, target_sharpe=0.5, alpha=0.05, skew=0.0, kurtosis=3.0
    )
    assert math.isinf(unreachable)
    reachable = min_track_record_length(
        observed_sharpe=0.7, target_sharpe=0.5, alpha=0.05, skew=0.0, kurtosis=3.0
    )
    assert math.isfinite(reachable)
    assert int(reachable) > 0


@pytest.mark.e2e
def test_subprocess_metrics_smoke() -> None:
    snippet = textwrap.dedent("""\
        import numpy as np
        from purgedcv import (
            probabilistic_sharpe_ratio,
            deflated_sharpe_ratio,
            deflated_sharpe_ratio_full,
            effective_n_trials,
            min_track_record_length,
        )
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, 200)
        psr = probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)
        dsr = deflated_sharpe_ratio(returns, n_trials=20, var_sharpe=0.005**2)
        diag = deflated_sharpe_ratio_full(returns, n_trials=20, var_sharpe=0.005**2)
        n_min = min_track_record_length(0.5, 0.2, 0.05, 0.0, 3.0)
        n_eff = effective_n_trials(np.cumsum(rng.standard_normal(1000)))
        assert 0 <= psr <= 1
        assert 0 <= dsr <= 1
        assert diag.dsr == dsr
        assert diag.n_obs == 200
        assert n_min > 0
        assert 1 <= n_eff <= 1000
        print("OK")
        """)
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "OK"
    assert result.stderr == ""
