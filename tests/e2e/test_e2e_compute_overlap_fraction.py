"""End-to-end tests for ``compute_overlap_fraction``.

User stories:
- A data engineer is debugging a custom splitter and wants to monitor
  the per-fold overlap fraction over many random splits without their
  monitoring loop blowing up on the first leaky split.
- A researcher wants a single number to log alongside model metrics so
  reviewers can verify split health at a glance.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv.diagnostics import compute_overlap_fraction


@pytest.mark.e2e
def test_user_story_clean_split_returns_zero() -> None:
    """A naive walk-forward split with non-overlapping labels: 0% overlap."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 70)
    test_idx = np.arange(80, 100)
    assert compute_overlap_fraction(train_idx, test_idx, pred, evalu) == 0.0


@pytest.mark.e2e
def test_user_story_partial_leakage_quantified() -> None:
    """7-day overlapping labels: 6 of 20 training rows leak into test.

    Test starts Jan 21 (pred[20]). Train rows 0..13 have evalu ranging
    Jan 8..Jan 21 — all evalu <= Jan 21, so under half-open no overlap.
    Train rows 14..19 have evalu Jan 22..Jan 27, all overlapping
    [Jan 21, Feb 6). That is exactly 6 of 20 → 0.30.
    """
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=7)
    train_idx = np.arange(0, 20)
    test_idx = np.arange(20, 30)
    fraction = compute_overlap_fraction(train_idx, test_idx, pred, evalu)
    assert fraction == pytest.approx(0.30)


@pytest.mark.e2e
def test_user_story_non_raising_in_monitoring_loop() -> None:
    """A monitoring loop that wraps every iteration in assert_no_temporal_leakage
    would blow up on the first leak. compute_overlap_fraction lets the loop
    continue collecting numbers."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    evalu = pred + pd.Timedelta(days=3)
    samples = []
    for shift in range(5):
        train_idx = np.arange(0, 30 - shift)
        test_idx = np.arange(30, 40)
        # Never raises:
        samples.append(compute_overlap_fraction(train_idx, test_idx, pred, evalu))
    assert all(isinstance(s, float) for s in samples)
    assert all(0.0 <= s <= 1.0 for s in samples)


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_fraction() -> None:
    snippet = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd
        from purgedcv import diagnostics

        pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        clean = diagnostics.compute_overlap_fraction(
            np.arange(0, 5), np.arange(10, 15), pred, evalu
        )
        assert clean == 0.0, f"expected 0.0, got {clean}"
        leak = diagnostics.compute_overlap_fraction(
            np.arange(10, 15), np.arange(10, 15), pred, evalu
        )
        assert leak == 1.0, f"expected 1.0, got {leak}"
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
