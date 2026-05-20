"""End-to-end tests for ``apply_embargo``.

User stories:
- A quantitative researcher applies a 2-day embargo after every test fold
  to prevent serial-correlation leakage in their financial-strategy
  backtest.
- The composition `apply_embargo -> assert_embargo_respected` must always
  be silent — this contract is the foundation of D5 splitters.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import apply_embargo
from purgedcv.diagnostics import assert_embargo_respected


@pytest.mark.e2e
def test_user_story_embargo_then_diagnostic_clean() -> None:
    """A researcher applies a 2-day embargo; the result must pass the
    diagnostic with the same embargo value."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 30)
    test_idx = np.arange(10, 15)
    emb = pd.Timedelta(days=2)
    embargoed = apply_embargo(train_idx, test_idx, pred, evalu, embargo=emb)
    # At least the first post-test row dropped.
    assert len(embargoed) < len(train_idx)
    # And the diagnostic agrees.
    assert_embargo_respected(embargoed, test_idx, pred, evalu, embargo=emb)


@pytest.mark.e2e
def test_user_story_zero_embargo_is_identity() -> None:
    """Users who pass embargo=0 (deliberately or by default) should see
    their train_idx returned unchanged."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 40)
    test_idx = np.arange(40, 50)
    result = apply_embargo(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(0))
    np.testing.assert_array_equal(result, train_idx)


@pytest.mark.e2e
def test_user_story_pre_test_history_preserved() -> None:
    """Asymmetry visible at API level: embargo never drops pre-test data."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    train_idx = np.arange(0, 5)  # all pre-test
    test_idx = np.arange(20, 25)
    result = apply_embargo(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=100))
    np.testing.assert_array_equal(result, train_idx)


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_embargo() -> None:
    snippet = textwrap.dedent("""\
        import numpy as np
        import pandas as pd
        from purgedcv import apply_embargo
        from purgedcv.diagnostics import assert_embargo_respected

        pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        train = np.arange(0, 25)
        test = np.arange(10, 15)
        emb = pd.Timedelta(days=2)
        embargoed = apply_embargo(train, test, pred, evalu, embargo=emb)
        assert_embargo_respected(embargoed, test, pred, evalu, embargo=emb)
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
