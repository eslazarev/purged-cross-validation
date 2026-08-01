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


#: A 1-D array-like of arbitrary elements: a numpy array, a Python sequence, or
#: any object exposing ``.to_numpy()`` (pandas/polars). This is the container
#: contract shared by time inputs and group labels; only the element semantics
#: differ. Coerced to a 1-D numpy array by ``purgedcv._time._coerce_1d``.
#:
#: Unlike a bare ``Any``, this alias still rejects unrelated types (an ``int``,
#: a mapping) statically, and it is a concrete ``Union`` object rather than a
#: string, so ``typing.get_type_hints`` on the annotated public functions
#: resolves without a ``NameError``.
ArrayLike1D: TypeAlias = NDArrayAny | Sequence[Any] | SupportsToNumpy

#: Time inputs (``prediction_times`` / ``evaluation_times``). The same
#: containers as :data:`ArrayLike1D`; ``validate_times`` additionally requires
#: the coerced array to hold a ``datetime64`` or ``timedelta64`` dtype.
TimesLike: TypeAlias = ArrayLike1D
