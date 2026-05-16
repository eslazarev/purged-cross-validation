"""End-to-end tests for ``assert_embargo_respected``.

User stories:
- A risk analyst wants to verify a financial-strategy split respects a
  multi-day embargo gap that prevents serial-correlation leakage.
- The asymmetry of the embargo (post-test only) must be visible at the
  public API so users don't double-apply it on the pre-test side.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import EmbargoViolationError
from purgedcv.diagnostics import assert_embargo_respected


@pytest.mark.e2e
def test_user_story_split_respects_3day_embargo() -> None:
    """Train ends Jan 19, test runs Jan 23-30, embargo=3D pushes cutoff to
    Feb 2. Train row at Jan 19 is strictly before test_eval_max=Jan 31, so
    it is not flagged."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=40, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 19)
    test_idx = np.arange(22, 30)
    assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=3))


@pytest.mark.e2e
def test_user_story_post_test_train_inside_embargo_caught() -> None:
    """Train row immediately following the test horizon must be inside the
    embargo window. Test runs rows 10-14 → eval_max = Jan 16. Embargo=2D
    → cutoff Jan 18. Train row 15 has pred=Jan 16 → inside [Jan 16, Jan 18]."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.array([15])
    test_idx = np.arange(10, 15)
    with pytest.raises(EmbargoViolationError) as exc_info:
        assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=2))
    assert "row 15" in str(exc_info.value)


@pytest.mark.e2e
def test_user_story_pre_test_history_never_flagged() -> None:
    """Users learning the API must see asymmetry: an oversized embargo
    does NOT drop pre-test training rows. This guards against the common
    misconception that embargo == purge."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 5)
    test_idx = np.arange(10, 15)
    assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=1000))


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_embargo() -> None:
    snippet = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd
        from purgedcv import diagnostics, EmbargoViolationError

        pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        diagnostics.assert_embargo_respected(
            np.array([18]), np.arange(10, 15), pred, evalu, embargo="2D"
        )
        try:
            diagnostics.assert_embargo_respected(
                np.array([15]), np.arange(10, 15), pred, evalu, embargo="2D"
            )
            raise AssertionError("should have raised")
        except EmbargoViolationError:
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
