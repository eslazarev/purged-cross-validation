"""End-to-end tests for path_metrics, cv.reconstruct_paths, and CSCV."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from math import comb

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import FitFailedWarning

from purgedcv import (
    CombinatoriallySymmetricCV,
    CombinatorialPurgedCV,
    default_backtest_metrics,
    path_metrics,
)


@pytest.mark.e2e
def test_user_story_cpcv_paths_to_metrics_table() -> None:
    """User Story: a quant builds CPCV backtest paths, turns them into a
    per-period PnL series, and wants one table summarising every path's
    Sharpe, Calmar, drawdown, and total return so the path distribution can
    be reported at a glance."""
    pred = pd.Series(pd.date_range("2023-01-01", periods=120, freq="D"))
    evalu = pred + pd.Timedelta(days=2)
    cv = CombinatorialPurgedCV(
        n_splits=6,
        n_test_groups=2,
        prediction_times=pred,
        evaluation_times=evalu,
    )
    rng = np.random.default_rng(5)
    X = rng.standard_normal((120, 3))  # noqa: N806
    y = X @ np.array([0.4, -0.2, 0.1]) + rng.standard_normal(120) * 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FitFailedWarning)
        paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)

    pnl = paths - y[np.newaxis, :]  # toy per-period PnL
    table = path_metrics(pnl, bars_per_year=252)
    assert table.shape == (comb(5, 1), 4)  # 5 paths, 4 metrics
    assert list(table.columns) == ["sharpe", "calmar", "max_drawdown", "total_return"]
    assert table.index.name == "path"
    assert (table["max_drawdown"].dropna() >= 0).all()


@pytest.mark.e2e
def test_user_story_manual_fold_loop_then_reconstruct() -> None:
    """User Story: a researcher runs a bespoke per-fold backtest loop (not the
    built-in backtest_paths) and then assembles the fold outputs into paths
    with the splitter's own reconstruct_paths method, no bookkeeping of
    n_splits / n_test_groups / n_samples required."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=24, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    cv = CombinatorialPurgedCV(
        n_splits=4,
        n_test_groups=2,
        prediction_times=pred,
        evaluation_times=evalu,
    )
    X = np.arange(24).reshape(-1, 1).astype(float)  # noqa: N806
    y = np.arange(24).astype(float)

    fold_preds = []
    for train_idx, test_idx in cv.split(X):
        model = DummyRegressor(strategy="mean").fit(X[train_idx], y[train_idx])
        fold_preds.append(np.asarray(model.predict(X[test_idx]), dtype=float))

    paths = cv.reconstruct_paths(fold_preds)
    assert paths.shape == (comb(3, 1), 24)
    assert np.all(np.isfinite(paths))

    metrics = default_backtest_metrics(paths[0])
    assert set(metrics) == {"sharpe", "calmar", "max_drawdown", "total_return"}


@pytest.mark.e2e
def test_user_story_cscv_symmetric_folds() -> None:
    """User Story: a user wants the symmetric IS/OOS folds that PBO is built
    on, exposed directly for their own analysis."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=48, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    cv = CombinatoriallySymmetricCV(
        n_splits=8,
        prediction_times=pred,
        evaluation_times=evalu,
    )
    assert cv.get_n_splits() == comb(8, 4)
    X = np.zeros((48, 1))  # noqa: N806
    for _, test_idx in cv.split(X):
        assert len(test_idx) == 24  # half the timeline is out-of-sample each fold


@pytest.mark.e2e
def test_subprocess_path_metrics_smoke() -> None:
    snippet = textwrap.dedent("""\
        import numpy as np
        from purgedcv import path_metrics, default_backtest_metrics
        rng = np.random.default_rng(0)
        paths = rng.standard_normal((5, 60)) * 0.01
        df = path_metrics(paths, bars_per_year=252)
        assert df.shape == (5, 4)
        assert list(df.columns) == ["sharpe", "calmar", "max_drawdown", "total_return"]
        m = default_backtest_metrics(paths[0])
        assert "sharpe" in m
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
