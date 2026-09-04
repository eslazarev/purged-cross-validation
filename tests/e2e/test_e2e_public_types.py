"""End-to-end tests for the public result types and callback aliases (A7)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from purgedcv import (
    DSRDiagnostics,
    PBOResult,
    deflated_sharpe_ratio_full,
    probability_of_backtest_overfitting,
)


@pytest.mark.e2e
def test_user_story_public_result_types_support_typed_downstream_code() -> None:
    """User Story: an application imports documented result classes from the
    package root, validates returned objects, and converts their frozen
    dataclass fields for a report without importing private modules.
    """
    rng = np.random.default_rng(17)
    returns = rng.normal(0.001, 0.01, 96)
    diag = deflated_sharpe_ratio_full(returns, n_trials=20, var_sharpe=0.005**2)

    assert isinstance(diag, DSRDiagnostics)
    assert asdict(diag)["n_obs"] == 96
    with pytest.raises(FrozenInstanceError):
        diag.n_obs = 97  # type: ignore[misc]

    strategy_returns = rng.normal(0.0, 0.01, (8, 96))
    result = probability_of_backtest_overfitting(strategy_returns, n_splits=8)
    assert isinstance(result, PBOResult)
    assert asdict(result)["n_combos"] == 70
    with pytest.raises(FrozenInstanceError):
        result.pbo = 0.0  # type: ignore[misc]


@pytest.mark.e2e
def test_subprocess_can_use_all_public_a7_types() -> None:
    """A fresh user environment can import and use all five A7 exports."""
    snippet = textwrap.dedent("""\
        import numpy as np
        from purgedcv import (
            DSRDiagnostics,
            HorizonLike,
            PBOResult,
            PathMetricFn,
            PerformanceMetric,
            deflated_sharpe_ratio_full,
            parse_horizon,
            path_metrics,
            probability_of_backtest_overfitting,
        )

        horizon: HorizonLike = "2D"
        assert parse_horizon(horizon).days == 2

        path_metric: PathMetricFn = lambda path: {"mean": float(np.mean(path))}
        table = path_metrics(np.array([[0.01, 0.02], [-0.01, 0.01]]), path_metric)
        assert list(table.columns) == ["mean"]

        performance_metric: PerformanceMetric = lambda values: float(np.mean(values))
        rng = np.random.default_rng(4)
        matrix = rng.normal(0.0, 0.01, (8, 96))
        pbo = probability_of_backtest_overfitting(
            matrix, n_splits=8, metric=performance_metric
        )
        assert isinstance(pbo, PBOResult)

        diag = deflated_sharpe_ratio_full(
            matrix[0], n_trials=8, var_sharpe=0.005**2
        )
        assert isinstance(diag, DSRDiagnostics)
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
