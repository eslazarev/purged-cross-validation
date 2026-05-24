"""Internal: shared input-validation helpers."""

from __future__ import annotations

import numpy as np

from ._typing import NDArrayAny


def _validate_integer(name: str, value: object, *, minimum: int) -> int:
    """Coerce ``value`` to an ``int`` of at least ``minimum``, or raise.

    ``bool`` is rejected explicitly: ``True`` and ``False`` are ``int``
    subclasses but are never meaningful split counts.

    Raises:
        TypeError: if ``value`` is not an integer.
        ValueError: if ``value`` is below ``minimum``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    value_int = int(value)
    if value_int < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value_int}.")
    return value_int


def _validate_positional_indices(
    name: str,
    indices: NDArrayAny,
    *,
    n_samples: int,
) -> NDArrayAny:
    """Validate a 1-D integer positional-index array.

    Negative positions are rejected even though ``pandas.iloc`` accepts
    them, because train/test index arrays are dataset positions, not Python
    slice syntax.

    Raises:
        TypeError: if ``indices`` is not integer-typed.
        ValueError: if ``indices`` is not 1-D or contains out-of-bounds
            positions.
    """
    arr = np.asarray(indices)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D positional index array.")
    if arr.size == 0:
        return arr.astype(np.int64, copy=False)
    if arr.dtype == np.dtype(bool) or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"{name} must contain integer positional indices.")
    negative = arr < 0
    if bool(np.any(negative)):
        first = int(np.flatnonzero(negative)[0])
        raise ValueError(f"{name} contains negative index {arr[first]} at position {first}.")
    too_large = arr >= n_samples
    if bool(np.any(too_large)):
        first = int(np.flatnonzero(too_large)[0])
        raise ValueError(
            f"{name} contains out-of-bounds index {arr[first]} at position {first}; "
            f"n_samples={n_samples}."
        )
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"{name} must not contain duplicate indices.")
    return arr
