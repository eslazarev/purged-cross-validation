"""Hypothesis property tests for purgedcv._time.

These tests assert algebraic invariants that should hold for ANY valid
input, complementing the hand-crafted unit tests with broad fuzz coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from purgedcv._time import horizons_overlap, parse_horizon

# pd.Timedelta has a hard range limit of about +/- 292 years (~106751 days).
# Hypothesis bounds stay well inside this to avoid OverflowError fuzz noise.
_MAX_DAYS = 100_000


@st.composite
def timestamps(draw: st.DrawFn) -> pd.Timestamp:
    """Generate timestamps in a constrained range to avoid pandas overflow."""
    dt = draw(
        st.datetimes(
            min_value=datetime(2000, 1, 1),
            max_value=datetime(2050, 12, 31),
        )
    )
    ts: pd.Timestamp = pd.Timestamp(dt)
    return ts


@st.composite
def interval(draw: st.DrawFn) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = draw(timestamps())
    delta_seconds = draw(st.integers(min_value=0, max_value=10**8))
    end = start + pd.Timedelta(seconds=delta_seconds)
    return start, end


class TestHorizonsOverlapProperties:
    @given(interval(), interval())
    def test_symmetry(
        self,
        a: tuple[pd.Timestamp, pd.Timestamp],
        b: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        forward = horizons_overlap(a[0], a[1], b[0], b[1])
        reverse = horizons_overlap(b[0], b[1], a[0], a[1])
        assert forward == reverse

    @given(interval())
    def test_non_degenerate_self_overlap(
        self,
        a: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        if a[1] > a[0]:
            assert horizons_overlap(a[0], a[1], a[0], a[1]) is True

    @given(timestamps(), st.integers(min_value=1, max_value=10**6))
    def test_touching_endpoints_do_not_overlap(
        self,
        anchor: pd.Timestamp,
        seconds: int,
    ) -> None:
        """Half-open invariant: [a, b) and [b, c) never overlap, regardless
        of which timestamp b lands on."""
        b = anchor + pd.Timedelta(seconds=seconds)
        c = b + pd.Timedelta(seconds=seconds)
        assert horizons_overlap(anchor, b, b, c) is False


class TestParseHorizonProperties:
    @given(st.integers(min_value=0, max_value=10**6))
    def test_seconds_roundtrip(self, n: int) -> None:
        result = parse_horizon(pd.Timedelta(seconds=n))
        assert result.total_seconds() == n

    @given(st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=10000)))
    def test_python_timedelta_roundtrip(self, td: timedelta) -> None:
        result = parse_horizon(td)
        assert result == pd.Timedelta(td)

    @given(st.integers(min_value=1, max_value=_MAX_DAYS))
    def test_day_string_roundtrip(self, days: int) -> None:
        """parse_horizon(f'{n}D') == Timedelta(days=n) for every positive n
        within pd.Timedelta's representable range."""
        result = parse_horizon(f"{days}D")
        assert result == pd.Timedelta(days=days)

    @given(st.integers(min_value=-_MAX_DAYS, max_value=-1))
    def test_negative_durations_always_rejected(self, days: int) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            parse_horizon(pd.Timedelta(days=days))
