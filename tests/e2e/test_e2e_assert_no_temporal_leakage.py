"""End-to-end tests for ``assert_no_temporal_leakage``.

User stories:
- A researcher builds a custom train/test split by hand and wants a
  one-liner to verify it has no temporal leakage before training a model.
- A reviewer auditing someone else's split wants to compute, with the
  same diagnostic, exactly which row leaked so they can investigate.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import TemporalLeakageError
from purgedcv.diagnostics import assert_no_temporal_leakage


@pytest.mark.e2e
def test_user_story_clinical_split_clean() -> None:
    """A clinical researcher splits a 100-patient dataset chronologically:
    first 70 days train, last 30 days test, with a 24h prediction horizon."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 70)
    test_idx = np.arange(70, 100)
    assert_no_temporal_leakage(train_idx, test_idx, pred, evalu)


@pytest.mark.e2e
def test_user_story_overlapping_labels_caught() -> None:
    """7-day overlapping labels: any naive split has leakage at the boundary.

    Test starts at Jan 21 (pred[20]); the diagnostic's argmax flags the first
    leaking training row in order. Row 14 has horizon [Jan 15, Jan 22) which
    overlaps [Jan 21, Feb 6), so the error names row 14.
    """
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=7)
    train_idx = np.arange(0, 20)
    test_idx = np.arange(20, 30)
    with pytest.raises(TemporalLeakageError) as exc_info:
        assert_no_temporal_leakage(train_idx, test_idx, pred, evalu)
    assert "row 14" in str(exc_info.value)


@pytest.mark.e2e
def test_user_story_purge_horizon_audits_safety_buffer() -> None:
    """A risk analyst wants to verify the split survives a 2-day safety buffer.

    Train spans Jan 1 through Jan 19 (rows 0..18), evaluation horizon = 1 day,
    so train evaluation times end at Jan 20. Test starts at Jan 21. Without
    buffer this is clean. With a 2-day purge buffer, the test window expands
    to [Jan 19, Feb 2): train row 18's horizon [Jan 19, Jan 20) overlaps that.
    """
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 19)
    test_idx = np.arange(20, 30)
    assert_no_temporal_leakage(train_idx, test_idx, pred, evalu)
    with pytest.raises(TemporalLeakageError):
        assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="2D")


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_diagnostic() -> None:
    snippet = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd
        from purgedcv import diagnostics, TemporalLeakageError

        pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        diagnostics.assert_no_temporal_leakage(
            np.arange(10), np.arange(15, 20), pred, evalu
        )
        # Now deliberately broken:
        try:
            diagnostics.assert_no_temporal_leakage(
                np.array([15]), np.arange(15, 20), pred, evalu
            )
            raise AssertionError("should have raised")
        except TemporalLeakageError:
            pass
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
