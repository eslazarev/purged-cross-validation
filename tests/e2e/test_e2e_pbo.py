"""End-to-end tests for Probability of Backtest Overfitting (PBO)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import probability_of_backtest_overfitting


@pytest.mark.e2e
def test_user_story_grid_search_over_noise_is_flagged() -> None:
    """User Story: a researcher grid-searches many strategy configurations on
    a market with no real edge, keeps each configuration's per-period
    returns, and asks PBO whether selecting the in-sample best is overfitting.

    With pure noise the in-sample winner is luck, so PBO sits near the 0.5
    no-skill line. Adding one genuinely profitable configuration pulls PBO
    down: the selector now locks onto real, repeatable performance.
    """
    rng = np.random.default_rng(2024)
    n_obs = 16 * 15  # 16 CSCV blocks, 15 rows each

    noise_grid = rng.standard_normal((40, n_obs)) * 0.01
    pbo_noise = probability_of_backtest_overfitting(noise_grid, n_splits=16)["pbo"]
    assert 0.3 <= pbo_noise <= 0.7

    winner = 0.04 + rng.standard_normal((1, n_obs)) * 0.01
    with_winner = np.vstack([winner, noise_grid])
    pbo_with_winner = probability_of_backtest_overfitting(with_winner, n_splits=16)["pbo"]
    assert pbo_with_winner < pbo_noise


@pytest.mark.e2e
def test_user_story_pbo_with_purge_on_overlapping_labels() -> None:
    """User Story: the return series come from a model whose labels span two
    days, so adjacent CSCV blocks share information. The researcher supplies
    prediction/evaluation times and a purge horizon; PBO cleans every IS/OOS
    boundary and still returns a coherent estimate over all combinations."""
    rng = np.random.default_rng(7)
    n_obs = 8 * 24
    returns = rng.standard_normal((12, n_obs)) * 0.01
    pred = pd.Series(pd.date_range("2022-01-01", periods=n_obs, freq="D"))
    evalu = pred + pd.Timedelta(days=2)

    result = probability_of_backtest_overfitting(
        returns,
        n_splits=8,
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon="2D",
        embargo="1D",
    )
    from math import comb

    assert result["n_combos"] == comb(8, 4)
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["logits"].shape == (result["n_combos"],)


@pytest.mark.e2e
def test_subprocess_pbo_smoke() -> None:
    snippet = textwrap.dedent("""\
        import numpy as np
        from purgedcv import probability_of_backtest_overfitting
        rng = np.random.default_rng(0)
        returns = rng.standard_normal((8, 200)) * 0.01
        result = probability_of_backtest_overfitting(returns, n_splits=8)
        assert 0.0 <= result["pbo"] <= 1.0
        assert result["n_combos"] == 70
        assert result["is_oos_performance"].shape == (70, 2)
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
