"""Internal typing helpers.

`numpy.ndarray` is generic. Under ``mypy --strict`` (``disallow_any_generics``)
a bare ``np.ndarray`` annotation is rejected on Python 3.10 environments, where
numpy's PEP 696 TypeVar defaults are not applied by the type checker. Spelling
the array type explicitly is valid in every Python / numpy / mypy combination,
so the whole package annotates arrays with this alias instead of bare
``np.ndarray``.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

#: Any numpy array. Equivalent in intent to the previously-used bare
#: ``np.ndarray``; the explicit ``Any`` satisfies ``disallow_any_generics``.
NDArrayAny: TypeAlias = npt.NDArray[Any]

#: Any accepted time-like input at a public boundary. pandas ``Series`` /
#: ``Index``, numpy ``datetime64`` / ``timedelta64`` arrays, Python sequences
#: of datetime/Timestamp/datetime64/timedelta, and polars ``Series`` (matched
#: by the trailing ``Any``, since polars is duck-typed and not importable at
#: type-check time). Coerced to numpy by ``purgedcv._time._coerce_1d``.
TimesLike: TypeAlias = "pd.Series | pd.Index | NDArrayAny | Sequence[Any] | Any"
