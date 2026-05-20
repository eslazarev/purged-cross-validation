"""End-to-end tests for ``validate_times``.

User stories:
- A researcher prepares prediction/evaluation time arrays for their splitter
  and wants a single up-front check that catches the common mistakes.
- The error message must point at the offending row, not just say
  "invalid input", so the researcher can fix their data quickly.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from purgedcv import validate_times


@pytest.mark.e2e
def test_user_story_well_formed_data_passes_silently() -> None:
    """A standard daily-cadence dataset with 24h horizons validates clean."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    validate_times(pred, evalu)


@pytest.mark.e2e
def test_user_story_length_mismatch_reported() -> None:
    """The researcher's labels Series got truncated by accident."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
    evalu = pd.Series(pd.date_range("2024-01-02", periods=9, freq="D"))
    with pytest.raises(ValueError, match="length"):
        validate_times(pred, evalu)


@pytest.mark.e2e
def test_user_story_inverted_horizon_at_specific_row_reported() -> None:
    """A row where the evaluation time landed BEFORE the prediction time —
    common when a researcher subtracts instead of adding a horizon."""
    pred = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-10"]))
    evalu = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-09"]))
    with pytest.raises(ValueError) as exc_info:
        validate_times(pred, evalu)
    message = str(exc_info.value)
    assert "index 2" in message
    assert "2024-01-09" in message
    assert "2024-01-10" in message


@pytest.mark.e2e
def test_user_story_nat_values_rejected() -> None:
    pred = pd.Series([pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-03")])
    evalu = pd.Series(
        [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    )
    with pytest.raises(ValueError, match="NaT"):
        validate_times(pred, evalu)


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_validate_clean_path() -> None:
    snippet = textwrap.dedent("""\
        import pandas as pd
        from purgedcv import validate_times
        pred = pd.Series(pd.date_range("2024-01-01", periods=5, freq="D"))
        evalu = pred + pd.Timedelta(hours=12)
        validate_times(pred, evalu)
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
