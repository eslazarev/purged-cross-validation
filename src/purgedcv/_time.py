"""Internal: time and horizon utilities (Domain D1)."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_timedelta64_dtype

HorizonLike = str | pd.Timedelta | timedelta | np.timedelta64

_AMBIGUOUS_OFFSETS = frozenset(
    {"M", "Y", "MS", "YS", "BM", "BMS", "BY", "BYS", "ME", "YE", "Q", "QS", "QE"}
)


def _is_temporal_series(values: pd.Series) -> bool:
    return bool(is_datetime64_any_dtype(values.dtype) or is_timedelta64_dtype(values.dtype))


def parse_horizon(value: HorizonLike) -> pd.Timedelta:
    """Coerce a horizon-like input to a non-negative ``pd.Timedelta``.

    Accepts pandas offset strings (``"2D"``, ``"6h"``, ``"30min"``),
    ``pd.Timedelta``, ``datetime.timedelta``, and ``numpy.timedelta64``.
    Rejects missing/``NaT`` values, negative durations, and
    calendar-ambiguous offsets such as ``"M"`` (month) or ``"Y"`` (year),
    which do not represent a fixed duration in seconds.

    Args:
        value: The horizon to parse.

    Returns:
        A non-negative ``pd.Timedelta``.

    Raises:
        ValueError: if the input is missing/``NaT``, negative, or a
            calendar-ambiguous string.
        TypeError: if the input is not one of the supported types.

    Examples:
        >>> from purgedcv import parse_horizon
        >>> parse_horizon("2D")
        Timedelta('2 days 00:00:00')
        >>> parse_horizon("6h")
        Timedelta('0 days 06:00:00')
    """
    if isinstance(value, str):
        if value in _AMBIGUOUS_OFFSETS:
            raise ValueError(
                f"Horizon string {value!r} is calendar-ambiguous; "
                "use a fixed duration like '30D' instead."
            )
        try:
            td = pd.Timedelta(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Could not parse horizon {value!r}: {exc}") from exc
    elif isinstance(value, pd.Timedelta):
        td = value
    elif isinstance(value, timedelta | np.timedelta64):
        td = pd.Timedelta(value)
    else:
        raise TypeError(
            f"Unsupported horizon type {type(value).__name__}; "
            "expected str, pd.Timedelta, datetime.timedelta, or np.timedelta64."
        )

    if pd.isna(td):
        raise ValueError("Horizon must be non-missing, got NaT.")
    if td < pd.Timedelta(0):
        raise ValueError(f"Horizon must be non-negative, got {td}.")

    return td


def horizons_overlap(
    a_start: pd.Timestamp,
    a_end: pd.Timestamp,
    b_start: pd.Timestamp,
    b_end: pd.Timestamp,
) -> bool:
    """Return ``True`` iff half-open intervals ``[a_start, a_end)`` and
    ``[b_start, b_end)`` overlap.

    Touching intervals (``a_end == b_start``) do NOT overlap. The function
    is symmetric in its arguments.

    Examples:
        >>> import pandas as pd
        >>> from purgedcv import horizons_overlap
        >>> horizons_overlap(
        ...     pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03"),
        ...     pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04"),
        ... )
        True
        >>> horizons_overlap(
        ...     pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"),
        ...     pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"),
        ... )
        False
    """
    return not (a_end <= b_start or b_end <= a_start)


def validate_times(
    prediction_times: pd.Series,
    evaluation_times: pd.Series,
    *,
    require_monotonic: bool = True,
) -> None:
    """Validate that ``prediction_times`` and ``evaluation_times`` are well-formed.

    Raises:
        ValueError: on length mismatch, non-temporal dtype, NaT values,
            ``evaluation_times < prediction_times`` at any row, or (when
            ``require_monotonic``) non-monotonic prediction times. The
            error message names the offending row index when applicable.

    Examples:
        >>> import pandas as pd
        >>> from purgedcv import validate_times
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=5, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> validate_times(pred, evalu)
    """
    if len(prediction_times) != len(evaluation_times):
        raise ValueError(
            f"length mismatch: prediction_times has {len(prediction_times)} rows, "
            f"evaluation_times has {len(evaluation_times)} rows."
        )
    if not _is_temporal_series(prediction_times):
        raise ValueError("prediction_times must have a datetime-like or timedelta-like dtype.")
    if not _is_temporal_series(evaluation_times):
        raise ValueError("evaluation_times must have a datetime-like or timedelta-like dtype.")
    if prediction_times.isna().any():
        raise ValueError("prediction_times contains NaT values.")
    if evaluation_times.isna().any():
        raise ValueError("evaluation_times contains NaT values.")
    bad = evaluation_times.to_numpy() < prediction_times.to_numpy()
    if bad.any():
        first = int(bad.argmax())
        raise ValueError(
            f"evaluation_times falls before prediction_times at index {first}: "
            f"{evaluation_times.iloc[first]} < {prediction_times.iloc[first]}."
        )
    if require_monotonic and not prediction_times.is_monotonic_increasing:
        raise ValueError("prediction_times must be monotonic non-decreasing.")
