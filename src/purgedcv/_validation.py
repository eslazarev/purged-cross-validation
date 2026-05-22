"""Internal: shared input-validation helpers."""

from __future__ import annotations

import numpy as np


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
