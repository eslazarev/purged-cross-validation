"""End-to-end tests for CombinatorialPurgedCV."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from math import comb

import numpy as np
import pandas as pd
import pytest

from purgedcv import CombinatorialPurgedCV
from purgedcv.diagnostics import (
    assert_embargo_respected,
    assert_no_temporal_leakage,
)


@pytest.mark.e2e
def test_user_story_cpcv_yields_distribution_of_folds() -> None:
    """A quantitative researcher wants a distribution of backtest folds
    instead of a single train/test split. CPCV with N=6, K=2 yields 15
    folds; each fold passes purge and embargo diagnostics."""
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
    X = np.zeros((120, 1))  # noqa: N806
    folds = list(cv.split(X))
    assert len(folds) == comb(6, 2) == 15
    for train_idx, test_idx in folds:
        assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="2D")
        assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo="1D")


@pytest.mark.e2e
def test_subprocess_cpcv_smoke() -> None:
    snippet = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd
        from math import comb
        from purgedcv import CombinatorialPurgedCV
        pred = pd.Series(pd.date_range("2024-01-01", periods=24, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        cv = CombinatorialPurgedCV(
            n_splits=6,
            n_test_groups=2,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        folds = list(cv.split(np.zeros((24, 1))))
        assert len(folds) == comb(6, 2) == 15
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
