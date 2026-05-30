"""E2E: the backtest-overfitting audit workflow must keep holding.

This mirrors ``examples/backtest_overfitting_audit.ipynb`` as a deterministic
library-level test (notebooks are docs here, not run under pytest, and the
notebook needs Optuna and a download). The audit chains the new tools: PBO on a
matrix of candidate strategies, the effective-trial correction for a correlated
search, the deflated Sharpe under raw vs effective counts, and the per-path
metric spread. Deterministic via fixed seeds.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import FitFailedWarning

from purgedcv import (
    CombinatorialPurgedCV,
    deflated_sharpe_ratio,
    effective_n_trials,
    path_metrics,
    probability_of_backtest_overfitting,
)

SEED = 0


@pytest.mark.e2e
def test_pbo_flags_a_search_over_noise() -> None:
    """Searching over pure-noise configurations cannot find a generalising
    winner, so PBO sits near the 0.5 no-skill line. Injecting one genuinely
    profitable configuration pulls PBO down: the selection now locks on."""
    rng = np.random.default_rng(SEED)
    noise = rng.standard_normal((80, 960)) * 0.01
    pbo_noise = probability_of_backtest_overfitting(noise, n_splits=12).pbo
    assert 0.3 <= pbo_noise <= 0.7

    winner = 0.03 + rng.standard_normal((1, 960)) * 0.01
    with_winner = np.vstack([winner, noise])
    pbo_with_winner = probability_of_backtest_overfitting(with_winner, n_splits=12).pbo
    assert pbo_with_winner < pbo_noise


@pytest.mark.e2e
def test_effective_trial_correction_rescues_dsr() -> None:
    """A correlated search (a converging trial trajectory) is far fewer
    independent bets than its raw count; deflating by the effective count
    leaves the champion's deflated Sharpe higher than the raw count does."""
    rng = np.random.default_rng(SEED)
    trial_sharpes = np.cumsum(rng.standard_normal(400)) * 0.01  # correlated path
    n_eff = effective_n_trials(trial_sharpes)
    assert 1 <= n_eff < 400

    returns = rng.normal(0.0015, 0.01, 282)
    dsr_raw = deflated_sharpe_ratio(returns, n_trials=400, var_sharpe=0.02**2)
    dsr_eff = deflated_sharpe_ratio(returns, n_trials=n_eff, var_sharpe=0.02**2)
    assert dsr_eff >= dsr_raw


@pytest.mark.e2e
def test_champion_path_spread_is_summarised() -> None:
    """The champion is run across combinatorial paths and reduced to a tidy
    per-path metric table, the honest alternative to one backtest number."""
    rng = np.random.default_rng(SEED)
    n = 240
    pred = pd.Series(pd.date_range("2021-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    cv = CombinatorialPurgedCV(
        n_splits=8,
        n_test_groups=2,
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon="1D",
    )
    X = rng.standard_normal((n, 3))  # noqa: N806
    y = X @ np.array([0.3, -0.2, 0.1]) + rng.standard_normal(n) * 0.1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FitFailedWarning)
        pred_paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
    strat_paths = np.sign(pred_paths) * y[np.newaxis, :]

    table = path_metrics(strat_paths, bars_per_year=252)
    from math import comb

    assert table.shape == (comb(7, 1), 4)
    assert list(table.columns) == ["sharpe", "calmar", "max_drawdown", "total_return"]
    assert table["max_drawdown"].dropna().ge(0).all()
