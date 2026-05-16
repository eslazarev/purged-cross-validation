"""Shared pytest fixtures for purged-cross-validation tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def daily_index_20() -> pd.DatetimeIndex:
    """Twenty consecutive daily timestamps starting 2024-01-01."""
    return pd.date_range("2024-01-01", periods=20, freq="D")
