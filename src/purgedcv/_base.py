"""Internal: abstract base class for purged-cross-validation splitters (Domain D4)."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd

from purgedcv._embargo import apply_embargo
from purgedcv._purge import purge
from purgedcv._time import HorizonLike, parse_horizon, validate_times
from purgedcv.diagnostics import assert_groups_disjoint


class BaseTemporalSplitter(ABC):
    """Duck-typed sklearn CV splitter with purge + embargo orchestration.

    Concrete subclasses implement :meth:`_iter_test_indices` to yield the
    raw test-index arrays for each fold. The base class handles purge,
    embargo, optional group-disjointness, and the sklearn-compatible
    :meth:`split` / :meth:`get_n_splits` protocol.

    Times are bound to the splitter at construction. This couples the
    splitter to a specific dataset's timestamps, which is intentional:
    a splitter for one dataset is rarely meaningful for another.

    .. note::
       The subclassing interface (``_iter_test_indices``,
       ``_candidate_train_idx``) is not yet covered by the v0.2 stability
       contract. Subclasses may need adjustments through v1.0.
    """

    def __init__(
        self,
        *,
        prediction_times: pd.Series,
        evaluation_times: pd.Series,
        purge_horizon: HorizonLike | None = None,
        embargo: HorizonLike | None = None,
        groups: pd.Series | None = None,
    ) -> None:
        validate_times(prediction_times, evaluation_times, require_monotonic=False)
        self._prediction_times = prediction_times.reset_index(drop=True)
        self._evaluation_times = evaluation_times.reset_index(drop=True)
        self.purge_horizon = (
            parse_horizon(purge_horizon) if purge_horizon is not None else pd.Timedelta(0)
        )
        self.embargo = parse_horizon(embargo) if embargo is not None else pd.Timedelta(0)
        self._groups: pd.Series | None
        if groups is not None:
            if len(groups) != len(self._prediction_times):
                raise ValueError(
                    f"groups length {len(groups)} does not match "
                    f"prediction_times length {len(self._prediction_times)}."
                )
            self._groups = groups.reset_index(drop=True)
        else:
            self._groups = None

    @abstractmethod
    def _iter_test_indices(self, n_samples: int) -> Sequence[np.ndarray]:
        """Return the test index arrays for each fold, in order."""
        ...

    @abstractmethod
    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:  # noqa: N803
        """Return the total number of splits the iterator will yield."""
        ...

    def split(
        self,
        X: np.ndarray | pd.DataFrame,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` pairs for each fold.

        The ``y`` and ``groups`` parameters of this method are accepted for
        sklearn protocol compatibility but ignored — group information must
        be bound at construction via the ``groups`` argument of ``__init__``.

        When groups were bound at construction,
        :func:`~purgedcv.diagnostics.assert_groups_disjoint` is called on
        every fold after purge and embargo; a
        :class:`~purgedcv.exceptions.GroupLeakageError` is raised if any
        group identifier appears in both train and test of the same fold.
        """
        n_samples = self._n_samples_or_check(X)

        for test_idx in self._iter_test_indices(n_samples):
            train_idx = self._candidate_train_idx(n_samples, test_idx)
            train_idx = purge(
                train_idx,
                test_idx,
                self._prediction_times,
                self._evaluation_times,
                purge_horizon=self.purge_horizon,
            )
            train_idx = apply_embargo(
                train_idx,
                test_idx,
                self._prediction_times,
                self._evaluation_times,
                embargo=self.embargo,
            )
            if self._groups is not None:
                assert_groups_disjoint(train_idx, test_idx, self._groups)
            yield train_idx, test_idx

    def _candidate_train_idx(self, n_samples: int, test_idx: np.ndarray) -> np.ndarray:
        """Return the candidate training indices BEFORE purge and embargo
        are applied.

        Default: every index that is not in ``test_idx``. Walk-forward
        splitters override this to restrict the candidate set to indices
        strictly before the test fold.

        The return must be a 1-D ``np.ndarray`` of integer indices (dtype int64
        is the convention), because downstream :func:`purge` and
        :func:`apply_embargo` use boolean fancy indexing on it.
        """
        mask = np.ones(n_samples, dtype=bool)
        mask[test_idx] = False
        return np.where(mask)[0].astype(np.int64)

    def with_times(
        self,
        prediction_times: pd.Series,
        evaluation_times: pd.Series,
    ) -> BaseTemporalSplitter:
        """Return a copy of this splitter with new times bound. All other
        parameters (``n_splits``, ``purge_horizon``, ``embargo``, ``groups``,
        and any subclass-specific state such as a cached unique-group list)
        are preserved unchanged.

        To change ``groups`` or any other construction parameter, build a
        fresh splitter via the constructor — this avoids surprising
        interactions between cached state and rebound inputs.
        """
        validate_times(prediction_times, evaluation_times, require_monotonic=False)
        if len(prediction_times) != len(self._prediction_times):
            raise ValueError(
                f"with_times got prediction_times of length {len(prediction_times)}; "
                f"this splitter was constructed for length "
                f"{len(self._prediction_times)}."
            )
        new = copy.copy(self)
        new._prediction_times = prediction_times.reset_index(drop=True)
        new._evaluation_times = evaluation_times.reset_index(drop=True)
        return new

    def _n_samples_or_check(self, X: np.ndarray | pd.DataFrame) -> int:  # noqa: N803
        """Return ``len(X)``, raising ``ValueError`` if it disagrees with
        the bound ``prediction_times`` length.
        """
        n = len(X)
        if n != len(self._prediction_times):
            raise ValueError(
                f"X length {n} does not match bound prediction_times length "
                f"{len(self._prediction_times)}."
            )
        return n
