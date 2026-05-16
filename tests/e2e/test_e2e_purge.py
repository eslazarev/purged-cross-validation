"""End-to-end tests for ``purge``.

User stories:
- A researcher builds a custom split by hand and wants to remove training
  rows that would leak into the test horizon, then verify the result with
  the diagnostic.
- The composition `purge → assert_no_temporal_leakage` must always be
  clean: this contract is the foundation that every splitter in domain
  D5 will rely on.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import purge
from purgedcv.diagnostics import assert_no_temporal_leakage


@pytest.mark.e2e
def test_user_story_purge_then_diagnostic_clean() -> None:
    """A researcher's mortality-prediction split has a 24-hour horizon; rows
    14..19 leak under a chronological train/test split. After purge, the
    diagnostic must pass."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=7)
    train_idx = np.arange(0, 20)
    test_idx = np.arange(20, 30)
    purged = purge(train_idx, test_idx, pred, evalu)
    # At least one row was dropped (the boundary rows leak).
    assert len(purged) < len(train_idx)
    # The result is leak-free.
    assert_no_temporal_leakage(purged, test_idx, pred, evalu)


@pytest.mark.e2e
def test_user_story_already_clean_split_unchanged() -> None:
    """No-op case: a properly-separated split should not lose any rows."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 50)
    test_idx = np.arange(60, 100)
    result = purge(train_idx, test_idx, pred, evalu)
    np.testing.assert_array_equal(result, train_idx)


@pytest.mark.e2e
def test_user_story_purge_returns_subset_of_input() -> None:
    """purge must always return a subset (in index-set sense) of its input,
    never a superset or a re-ordering of new rows."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    evalu = pred + pd.Timedelta(days=5)
    rng = np.random.default_rng(seed=42)
    for _ in range(20):
        train_idx = rng.choice(50, size=30, replace=False)
        test_idx = rng.choice(50, size=10, replace=False)
        # Strip any overlap so train and test are disjoint.
        train_idx = np.array([i for i in train_idx if i not in set(test_idx.tolist())])
        if len(train_idx) == 0:
            continue
        result = purge(train_idx, test_idx, pred, evalu)
        assert set(result.tolist()).issubset(set(train_idx.tolist()))


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_purge() -> None:
    snippet = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd
        from purgedcv import purge
        from purgedcv.diagnostics import assert_no_temporal_leakage

        pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        evalu = pred + pd.Timedelta(days=3)
        train = np.arange(0, 12)
        test = np.arange(10, 15)
        purged = purge(train, test, pred, evalu)
        assert_no_temporal_leakage(purged, test, pred, evalu)
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
