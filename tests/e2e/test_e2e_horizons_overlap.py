"""End-to-end tests for ``horizons_overlap``.

User stories:
- A researcher wants to check whether two label horizons of a time-series
  dataset overlap, so they can decide whether one sample's label leaks into
  another sample's training context.
- The half-open convention must be visible at the public API: a user who
  expects "touching" intervals to count as overlap should get a clear no.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from purgedcv import horizons_overlap


@pytest.mark.e2e
def test_user_story_label_horizons_clearly_disjoint() -> None:
    """Two patient encounters on different days have non-overlapping 24h horizons."""
    a_start = pd.Timestamp("2024-03-15 08:00")
    a_end = pd.Timestamp("2024-03-16 08:00")
    b_start = pd.Timestamp("2024-03-20 08:00")
    b_end = pd.Timestamp("2024-03-21 08:00")
    assert horizons_overlap(a_start, a_end, b_start, b_end) is False


@pytest.mark.e2e
def test_user_story_overlapping_label_horizons_detected() -> None:
    """A 24h horizon starting at hour 0 overlaps a 24h horizon starting at hour 12."""
    a_start = pd.Timestamp("2024-03-15 00:00")
    a_end = pd.Timestamp("2024-03-16 00:00")
    b_start = pd.Timestamp("2024-03-15 12:00")
    b_end = pd.Timestamp("2024-03-16 12:00")
    assert horizons_overlap(a_start, a_end, b_start, b_end) is True


@pytest.mark.e2e
def test_user_story_touching_intervals_do_not_overlap() -> None:
    """Half-open convention: a 1-hour horizon ending at 09:00 and one starting
    at 09:00 share zero duration, so they do not overlap."""
    a_start = pd.Timestamp("2024-03-15 08:00")
    a_end = pd.Timestamp("2024-03-15 09:00")
    b_start = pd.Timestamp("2024-03-15 09:00")
    b_end = pd.Timestamp("2024-03-15 10:00")
    assert horizons_overlap(a_start, a_end, b_start, b_end) is False


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_overlap_check() -> None:
    snippet = textwrap.dedent("""\
        import pandas as pd
        from purgedcv import horizons_overlap
        a = (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"))
        b = (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"))
        assert horizons_overlap(*a, *b) is True
        c = (pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-11"))
        assert horizons_overlap(*a, *c) is False
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
