"""End-to-end tests for CombinatorialPurgedCV.backtest_paths and the
top-level reconstruct_paths function."""

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

from purgedcv import CombinatorialPurgedCV, reconstruct_paths


@pytest.mark.e2e
def test_user_story_quant_researcher_gets_distribution_of_predictions() -> None:
    """A quant researcher wants 5 backtest paths (N=6, K=2) on 120 daily
    samples. Each path is a complete time-ordered sequence; the user
    can compute per-path metrics on the (5, 120) matrix."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=120, freq="D"))
    evalu = pred + pd.Timedelta(days=2)
    cv = CombinatorialPurgedCV(
        n_splits=6,
        n_test_groups=2,
        purge_horizon="2D",
        embargo="1D",
        prediction_times=pred,
        evaluation_times=evalu,
    )
    X = np.arange(120).reshape(-1, 1).astype(float)  # noqa: N806
    y = np.arange(120).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FitFailedWarning)
        paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
    assert paths.shape == (5, 120)
    # User computes a per-path metric: mean prediction per path.
    per_path_mean = np.nanmean(paths, axis=1)
    assert per_path_mean.shape == (5,)


@pytest.mark.e2e
def test_user_story_low_level_reconstruct_paths_with_precomputed_predictions() -> None:
    """A power user who has cached pre-computed per-fold predictions
    (e.g. from a parallel training run) can call reconstruct_paths
    directly without touching the splitter."""
    n_samples = 16
    # Simulate the C(4, 2) = 6 folds in itertools.combinations order.
    fold_test = [
        np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
        np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=np.int64),
        np.array([0, 1, 2, 3, 12, 13, 14, 15], dtype=np.int64),
        np.array([4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64),
        np.array([4, 5, 6, 7, 12, 13, 14, 15], dtype=np.int64),
        np.array([8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int64),
    ]
    fold_preds = [
        np.full(len(test_idx), float(fold_idx), dtype=float)
        for fold_idx, test_idx in enumerate(fold_test)
    ]
    paths = reconstruct_paths(fold_preds, fold_test, 4, 2, n_samples)
    assert paths.shape == (3, n_samples)
    assert np.all(np.isfinite(paths))


@pytest.mark.e2e
def test_subprocess_backtest_paths_smoke() -> None:
    snippet = textwrap.dedent(
        """\
        import warnings
        import numpy as np
        import pandas as pd
        from sklearn.dummy import DummyRegressor
        from sklearn.exceptions import FitFailedWarning
        from purgedcv import CombinatorialPurgedCV, reconstruct_paths
        pred = pd.Series(pd.date_range("2024-01-01", periods=24, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        cv = CombinatorialPurgedCV(
            n_splits=6, n_test_groups=2,
            prediction_times=pred, evaluation_times=evalu,
        )
        X = np.arange(24).reshape(-1, 1).astype(float)
        y = np.arange(24).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FitFailedWarning)
            paths = cv.backtest_paths(DummyRegressor(strategy='mean'), X, y)
        assert paths.shape == (5, 24)
        # Verify reconstruct_paths is also importable as the lower-level function:
        assert reconstruct_paths is not None
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "OK"
    assert result.stderr == ""
