"""Internal typing helpers.

`numpy.ndarray` is generic. Under ``mypy --strict`` (``disallow_any_generics``)
a bare ``np.ndarray`` annotation is rejected on Python 3.10 environments, where
numpy's PEP 696 TypeVar defaults are not applied by the type checker. Spelling
the array type explicitly is valid in every Python / numpy / mypy combination,
so the whole package annotates arrays with this alias instead of bare
``np.ndarray``.
"""

from collections.abc import Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy.typing as npt

#: Any numpy array. Equivalent in intent to the previously-used bare
#: ``np.ndarray``; the explicit ``Any`` satisfies ``disallow_any_generics``.
NDArrayAny: TypeAlias = npt.NDArray[Any]


@runtime_checkable
class SupportsToNumpy(Protocol):
    """Structural type for a sized container exposing a zero-argument
    ``.to_numpy()``.

    pandas ``Series``/``Index`` and polars ``Series`` all satisfy this, so the
    library accepts them without importing pandas as a typing requirement or
    importing polars at all. numpy arrays do NOT define ``.to_numpy()``; they
    are matched by the ``NDArrayAny`` arm of the array-like aliases below.
    ``__len__`` is part of the contract because the library length-checks these
    inputs before coercing them.
    """

    def __len__(self) -> int: ...

    def to_numpy(self) -> NDArrayAny: ...


ArrayLike1D: TypeAlias = NDArrayAny | Sequence[Any] | SupportsToNumpy
"""A 1-D array-like of arbitrary elements: a NumPy array, a Python sequence, or
a sized object exposing ``.to_numpy()`` (including pandas and polars containers).

This is the container contract shared by time inputs and group labels; only
the element semantics differ. Public functions coerce inputs to 1-D NumPy
arrays and validate the shape at runtime.

Unlike a bare ``Any``, this alias rejects unrelated types (an ``int``, a
mapping) statically. It is a concrete union rather than a string, so
``typing.get_type_hints`` can resolve annotations on public functions.
"""

TimesLike: TypeAlias = ArrayLike1D
"""Time inputs for ``prediction_times`` and ``evaluation_times``.

Accepts the same containers as [`ArrayLike1D`][purgedcv.ArrayLike1D].
After coercion, inputs must have a ``datetime64`` or ``timedelta64`` dtype,
not strings or numeric timestamps. Examples include pandas ``DatetimeIndex``
or ``Series``, NumPy temporal arrays, Python lists of datetime/timedelta
objects, and polars temporal ``Series``. Timezone-aware pandas inputs are
normalized to UTC.

[`validate_times`][purgedcv.validate_times] checks lengths, temporal dtype
families, missing values, and label ordering. Splitters also require
non-decreasing prediction times.
"""
