"""End-to-end tests for PurgedKFold."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import PurgedKFold
from purgedcv.diagnostics import (
    assert_embargo_respected,
    assert_no_temporal_leakage,
)


@pytest.mark.e2e
def test_user_story_5fold_with_overlapping_labels() -> None:
    """Clinical-style dataset: 100 daily samples with 3-day overlapping
    labels. 5-fold CV with a 3-day purge horizon must produce clean folds."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
    evalu = pred + pd.Timedelta(days=3)
    cv = PurgedKFold(
        n_splits=5,
        purge_horizon="3D",
        embargo="1D",
        prediction_times=pred,
        evaluation_times=evalu,
    )
    X = np.zeros((100, 1))  # noqa: N806
    folds = list(cv.split(X))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="3D")
        assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo="1D")


@pytest.mark.e2e
def test_subprocess_purged_kfold_smoke() -> None:
    snippet = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd
        from purgedcv import PurgedKFold
        pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        cv = PurgedKFold(
            n_splits=5,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        folds = list(cv.split(np.zeros((20, 1))))
        assert len(folds) == 5
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
