"""Internal: time and horizon utilities (Domain D1)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from ._typing import NDArrayAny, SupportsToNumpy, TimesLike

HorizonLike = str | pd.Timedelta | timedelta | np.timedelta64

_AMBIGUOUS_OFFSETS = frozenset(
    {"M", "Y", "MS", "YS", "BM", "BMS", "BY", "BYS", "ME", "YE", "Q", "QS", "QE"}
)


def _temporal_kind(arr: NDArrayAny) -> str | None:
    if np.issubdtype(arr.dtype, np.datetime64):
        return "datetime"
    if np.issubdtype(arr.dtype, np.timedelta64):
        return "timedelta"
    return None


def _coerce_1d(x: TimesLike, *, name: str) -> NDArrayAny:
    """Coerce any accepted time-like input to a 1-D numpy array.

    Handles pandas ``Series`` / ``Index`` (including tz-aware), numpy
    ``datetime64`` / ``timedelta64`` arrays, polars ``Series`` (duck-typed
    via ``.to_numpy()``, never imported), and Python sequences of
    ``datetime`` / ``Timestamp`` / ``datetime64`` / ``timedelta``.

    tz-aware pandas input is converted to UTC and made tz-naive
    ``datetime64``. Sequences that land as ``object`` dtype (or as numpy's
    fixed-width string dtype, which is what a plain Python list of strings
    becomes under ``np.asarray``) are routed through pandas so lists of
    datetime/Timedelta objects resolve to ``datetime64`` / ``timedelta64``.
    Inputs that stay ``object`` after that (for example a list of strings)
    are returned as-is; ``validate_times`` then rejects them with a clear
    dtype message. ``name`` is unused today but kept so future messages can
    name the offending input.
    """
    dtype = getattr(x, "dtype", None)
    if isinstance(dtype, pd.DatetimeTZDtype):
        # x is a pandas Series/Index here (only those expose a real .dtype
        # attribute equal to a DatetimeTZDtype instance); narrow explicitly
        # since TimesLike's Sequence[Any] member confuses the DatetimeIndex
        # constructor's overloads under mypy --strict.
        x_pandas: Any = x
        idx = pd.DatetimeIndex(x_pandas)  # accepts both Series and Index
        out: NDArrayAny = idx.tz_convert("UTC").tz_localize(None).to_numpy()
        return out
    if isinstance(x, np.ndarray):
        arr: NDArrayAny = x
    elif isinstance(x, SupportsToNumpy):  # pandas Series/Index, polars Series
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)
    arr = np.asarray(arr)
    # Check dimensionality before the object/string coercion below: a 0-D
    # scalar routed through ``pd.Series(...).to_numpy()`` would be reshaped to a
    # length-1 1-D array and slip past this guard.
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array-like; got a {arr.ndim}-D input of shape "
            f"{arr.shape}. Pass a 1-D sequence, not a scalar or a 2-D array/"
            f"DataFrame (select a single column first)."
        )
    if arr.dtype == object or arr.dtype.kind in "US":
        arr = pd.Series(arr).to_numpy()
    return arr


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

    horizon: pd.Timedelta = td
    return horizon


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
    endpoints = {
        "a_start": a_start,
        "a_end": a_end,
        "b_start": b_start,
        "b_end": b_end,
    }
    for name, value in endpoints.items():
        if pd.isna(value):
            raise ValueError(f"{name} must be non-missing, got NaT.")
    if a_end < a_start:
        raise ValueError(f"a_end ({a_end}) must be greater than or equal to a_start ({a_start}).")
    if b_end < b_start:
        raise ValueError(f"b_end ({b_end}) must be greater than or equal to b_start ({b_start}).")
    return not (a_end <= b_start or b_end <= a_start)


def validate_times(
    prediction_times: TimesLike,
    evaluation_times: TimesLike,
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
    pred = _coerce_1d(prediction_times, name="prediction_times")
    evalu = _coerce_1d(evaluation_times, name="evaluation_times")
    if len(pred) != len(evalu):
        raise ValueError(
            f"length mismatch: prediction_times has {len(pred)} rows, "
            f"evaluation_times has {len(evalu)} rows."
        )
    prediction_kind = _temporal_kind(pred)
    evaluation_kind = _temporal_kind(evalu)
    if prediction_kind is None:
        raise ValueError("prediction_times must have a datetime-like or timedelta-like dtype.")
    if evaluation_kind is None:
        raise ValueError("evaluation_times must have a datetime-like or timedelta-like dtype.")
    if prediction_kind != evaluation_kind:
        raise ValueError(
            "prediction_times and evaluation_times must use the same temporal dtype family."
        )
    if np.isnat(pred).any():
        raise ValueError("prediction_times contains NaT values.")
    if np.isnat(evalu).any():
        raise ValueError("evaluation_times contains NaT values.")
    bad = evalu < pred
    if bad.any():
        first = int(bad.argmax())
        raise ValueError(
            f"evaluation_times falls before prediction_times at index {first}: "
            f"{evalu[first]} < {pred[first]}."
        )
    if require_monotonic and not bool(np.all(pred[1:] >= pred[:-1])):
        raise ValueError("prediction_times must be monotonic non-decreasing.")
