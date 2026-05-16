"""Internal typing helpers.

`numpy.ndarray` is generic. Under ``mypy --strict`` (``disallow_any_generics``)
a bare ``np.ndarray`` annotation is rejected on Python 3.10 environments, where
numpy's PEP 696 TypeVar defaults are not applied by the type checker. Spelling
the array type explicitly is valid in every Python / numpy / mypy combination,
so the whole package annotates arrays with this alias instead of bare
``np.ndarray``.
"""

from typing import Any, TypeAlias

import numpy.typing as npt

#: Any numpy array. Equivalent in intent to the previously-used bare
#: ``np.ndarray``; the explicit ``Any`` satisfies ``disallow_any_generics``.
NDArrayAny: TypeAlias = npt.NDArray[Any]
