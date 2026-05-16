"""End-to-end tests for WalkForwardSplit."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import WalkForwardSplit
from purgedcv.diagnostics import (
    assert_embargo_respected,
    assert_no_temporal_leakage,
)


@pytest.mark.e2e
def test_user_story_walk_forward_with_purge_and_embargo() -> None:
    """A researcher with 5-day labels and a 1-day embargo runs walk-forward
    CV. Every fold must satisfy the diagnostics."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
    evalu = pred + pd.Timedelta(days=5)
    cv = WalkForwardSplit(
        n_splits=5,
        test_size=10,
        purge_horizon="2D",
        embargo="1D",
        prediction_times=pred,
        evaluation_times=evalu,
    )
    X = np.zeros((100, 1))  # noqa: N806
    folds = list(cv.split(X))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="2D")
        assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo="1D")


@pytest.mark.e2e
def test_user_story_with_times_adapter() -> None:
    """A code path that builds the splitter and rebinds times via
    .with_times() should produce splits with the new times applied."""
    pred1 = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    evalu1 = pred1 + pd.Timedelta(days=1)
    cv = WalkForwardSplit(
        n_splits=5,
        test_size=5,
        prediction_times=pred1,
        evaluation_times=evalu1,
    )
    # Rebind to a different (but same-length) date range:
    pred2 = pd.Series(pd.date_range("2030-06-01", periods=50, freq="D"))
    evalu2 = pred2 + pd.Timedelta(days=1)
    cv2 = cv.with_times(pred2, evalu2)
    X = np.zeros((50, 1))  # noqa: N806
    # Verify the rebind actually happened: a bug where with_times silently
    # returned self would leave cv2._prediction_times pointing at pred1.
    assert cv2._prediction_times.iloc[0] == pred2.iloc[0]
    assert cv2._evaluation_times.iloc[0] == evalu2.iloc[0]
    # And the diagnostic against the new times must still succeed:
    for train_idx, test_idx in cv2.split(X):
        assert_no_temporal_leakage(train_idx, test_idx, pred2, evalu2)


@pytest.mark.e2e
def test_subprocess_walk_forward_smoke() -> None:
    snippet = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd
        from purgedcv import WalkForwardSplit
        pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        cv = WalkForwardSplit(
            n_splits=4,
            test_size=2,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        folds = list(cv.split(np.zeros((20, 1))))
        assert len(folds) == 4
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
