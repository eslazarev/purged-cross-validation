"""End-to-end tests for the public ``parse_horizon`` entry point.

These tests treat ``purgedcv`` as an installed third-party library and
exercise its public surface the way a data scientist would write it. Two
flavors:

1. **User-story** — pytest functions using ``from purgedcv import parse_horizon``
   that walk through a realistic happy path plus a failure mode with a
   useful error message.
2. **Subprocess smoke** — spawns a fresh Python interpreter so we catch
   any import-time side effects or missed re-exports that the in-process
   tests would not see.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from purgedcv import parse_horizon


@pytest.mark.e2e
def test_user_story_typical_offset_strings() -> None:
    """User story: a researcher converts engineering-shorthand horizons to Timedeltas."""
    assert parse_horizon("2D") == pd.Timedelta(days=2)
    assert parse_horizon("6h") == pd.Timedelta(hours=6)
    assert parse_horizon("30min") == pd.Timedelta(minutes=30)
    assert parse_horizon("1W") == pd.Timedelta(weeks=1)


@pytest.mark.e2e
def test_user_story_existing_timedelta_passes_through() -> None:
    """User story: code that already has ``pd.Timedelta`` objects shouldn't need to convert."""
    horizon = pd.Timedelta(hours=12)
    assert parse_horizon(horizon) is horizon or parse_horizon(horizon) == horizon


@pytest.mark.e2e
def test_user_story_calendar_ambiguous_input_is_rejected_with_useful_message() -> None:
    """User story: a researcher accidentally passes 'M' for 'month'.

    They should get a clear error explaining the issue, not a silent
    miscoercion to '1 minute' or similar.
    """
    with pytest.raises(ValueError) as exc_info:
        parse_horizon("M")
    message = str(exc_info.value).lower()
    assert "ambiguous" in message, f"error should mention ambiguity, got: {message}"
    assert "'m'" in message or "m" in message


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_can_use_parse_horizon() -> None:
    """A fresh Python interpreter that just imports the library can call
    ``parse_horizon`` and get the expected result, with no stderr noise."""
    snippet = textwrap.dedent(
        """\
        import pandas as pd
        from purgedcv import parse_horizon
        result = parse_horizon("3D")
        assert result == pd.Timedelta(days=3), f"got {result}"
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
