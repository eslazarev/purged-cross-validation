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
    min_track_record_length,
    probabilistic_sharpe_ratio,
)


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
def test_subprocess_metrics_smoke() -> None:
    snippet = textwrap.dedent("""\
        import numpy as np
        from purgedcv import (
            probabilistic_sharpe_ratio,
            deflated_sharpe_ratio,
            min_track_record_length,
        )
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, 200)
        psr = probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)
        dsr = deflated_sharpe_ratio(returns, n_trials=20, var_sharpe=0.005**2)
        n_min = min_track_record_length(0.5, 0.2, 0.05, 0.0, 3.0)
        assert 0 <= psr <= 1
        assert 0 <= dsr <= 1
        assert n_min > 0
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
