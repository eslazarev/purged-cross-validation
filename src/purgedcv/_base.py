"""Internal: abstract base class for purged-cross-validation splitters (Domain D4)."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from purgedcv._embargo import _validate_embargo_modes, apply_embargo
from purgedcv._purge import purge
from purgedcv._time import HorizonLike, _coerce_1d, parse_horizon, validate_times
from purgedcv._validation import _validate_positional_indices

from ._typing import ArrayLike1D, NDArrayAny, TimesLike


@dataclass(frozen=True)
class _SplitStages:
    """Internal snapshots of the train-index filtering pipeline."""

    candidate_train_idx: NDArrayAny
    purged_train_idx: NDArrayAny
    embargoed_train_idx: NDArrayAny
    final_train_idx: NDArrayAny
    test_idx: NDArrayAny


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
       ``_candidate_train_idx``) is an implementation detail, not part of the
       maintained ``0.1.x`` public contract. Subclasses may need adjustments
       before v1.0.
    """

    def __init__(
        self,
        *,
        prediction_times: TimesLike,
        evaluation_times: TimesLike,
        purge_horizon: HorizonLike | None = None,
        embargo: HorizonLike | None = None,
        embargo_observations: int | None = None,
        embargo_fraction: float | None = None,
        groups: ArrayLike1D | None = None,
    ) -> None:
        pred = _coerce_1d(prediction_times, name="prediction_times")
        evalu = _coerce_1d(evaluation_times, name="evaluation_times")
        validate_times(pred, evalu, require_monotonic=True)
        self._prediction_times = pred
        self._evaluation_times = evalu
        self.purge_horizon = (
            parse_horizon(purge_horizon) if purge_horizon is not None else pd.Timedelta(0)
        )
        duration, observations, fraction = _validate_embargo_modes(
            embargo, embargo_observations, embargo_fraction
        )
        self.embargo = duration if duration is not None else pd.Timedelta(0)
        self.embargo_observations = observations
        self.embargo_fraction = fraction
        self._groups: NDArrayAny | None
        if groups is not None:
            groups_arr = _coerce_1d(groups, name="groups")
            if len(groups_arr) != len(self._prediction_times):
                raise ValueError(
                    f"groups length {len(groups_arr)} does not match "
                    f"prediction_times length {len(self._prediction_times)}."
                )
            if pd.isna(groups_arr).any():
                raise ValueError("groups contains missing values.")
            self._groups = groups_arr
        else:
            self._groups = None

    @abstractmethod
    def _iter_test_indices(self, n_samples: int) -> Sequence[NDArrayAny]:
        """Return the test index arrays for each fold, in order."""
        ...

    @abstractmethod
    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        """Return the total number of splits the iterator will yield."""
        ...

    def split(
        self,
        X: NDArrayAny | pd.DataFrame,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> Iterator[tuple[NDArrayAny, NDArrayAny]]:
        """Yield ``(train_idx, test_idx)`` pairs for each fold.

        The ``y`` and ``groups`` parameters of this method are accepted for
        sklearn protocol compatibility but ignored — group information must
        be bound at construction via the ``groups`` argument of ``__init__``.

        When groups were bound at construction,
        :func:`~purgedcv.diagnostics.assert_groups_disjoint` is called on
        every fold after purge, embargo, and splitter-specific finalization; a
        :class:`~purgedcv.exceptions.GroupLeakageError` is raised if any
        group identifier appears in both train and test of the same fold.
        """
        for stages in self._iter_split_stages(X):
            train_idx = stages.final_train_idx
            test_idx = stages.test_idx
            if self._groups is not None:
                # Local import keeps diagnostics free to depend on the base
                # splitter for the public audit_splitter type and pipeline.
                from purgedcv.diagnostics import assert_groups_disjoint

                assert_groups_disjoint(train_idx, test_idx, self._groups)
            yield train_idx, test_idx

    def _iter_split_stages(
        self,
        X: NDArrayAny | pd.DataFrame,  # noqa: N803
    ) -> Iterator[_SplitStages]:
        """Yield each real candidate → purge → embargo → final pipeline.

        This is the single orchestration path used by :meth:`split` and the
        public :func:`purgedcv.audit_splitter` report. Keeping the intermediate
        arrays here prevents diagnostics from guessing purge and embargo
        effects by comparing only the final split indices.
        """
        n_samples = self._n_samples_or_check(X)

        for test_idx in self._iter_test_indices(n_samples):
            test_idx = _validate_positional_indices("test_idx", test_idx, n_samples=n_samples)
            candidate_train_idx = self._candidate_train_idx(n_samples, test_idx)
            purged_train_idx = purge(
                candidate_train_idx,
                test_idx,
                self._prediction_times,
                self._evaluation_times,
                purge_horizon=self.purge_horizon,
            )
            embargoed_train_idx = apply_embargo(
                purged_train_idx,
                test_idx,
                self._prediction_times,
                self._evaluation_times,
                embargo=(
                    self.embargo
                    if self.embargo_observations is None and self.embargo_fraction is None
                    else None
                ),
                embargo_observations=self.embargo_observations,
                embargo_fraction=self.embargo_fraction,
            )
            final_train_idx = self._finalize_train_idx(embargoed_train_idx, test_idx)
            yield _SplitStages(
                candidate_train_idx=candidate_train_idx,
                purged_train_idx=purged_train_idx,
                embargoed_train_idx=embargoed_train_idx,
                final_train_idx=final_train_idx,
                test_idx=test_idx,
            )

    def _candidate_train_idx(self, n_samples: int, test_idx: NDArrayAny) -> NDArrayAny:
        """Return the candidate training indices BEFORE purge and embargo
        are applied.

        Default: every index that is not in ``test_idx``. Walk-forward
        splitters override this to restrict the candidate set to indices
        strictly before the test fold.

        The return must be a 1-D ``NDArrayAny`` of integer indices (dtype int64
        is the convention), because downstream :func:`purge` and
        :func:`apply_embargo` use boolean fancy indexing on it.
        """
        mask = np.ones(n_samples, dtype=bool)
        mask[test_idx] = False
        return np.where(mask)[0].astype(np.int64)

    def _finalize_train_idx(
        self,
        train_idx: NDArrayAny,
        test_idx: NDArrayAny,
    ) -> NDArrayAny:
        """Apply splitter-specific filtering after purge and embargo."""
        return train_idx

    def with_times(
        self,
        prediction_times: TimesLike,
        evaluation_times: TimesLike,
    ) -> BaseTemporalSplitter:
        """Return a copy of this splitter with new times bound. All other
        parameters (``n_splits``, ``purge_horizon``, every embargo mode,
        ``groups``, and any subclass-specific state such as a cached unique-group list)
        are preserved unchanged.

        To change ``groups`` or any other construction parameter, build a
        fresh splitter via the constructor — this avoids surprising
        interactions between cached state and rebound inputs.
        """
        pred = _coerce_1d(prediction_times, name="prediction_times")
        evalu = _coerce_1d(evaluation_times, name="evaluation_times")
        validate_times(pred, evalu, require_monotonic=True)
        if len(pred) != len(self._prediction_times):
            raise ValueError(
                f"with_times got prediction_times of length {len(pred)}; "
                f"this splitter was constructed for length "
                f"{len(self._prediction_times)}."
            )
        new = copy.copy(self)
        new._prediction_times = pred
        new._evaluation_times = evalu
        return new

    def _n_samples_or_check(self, X: NDArrayAny | pd.DataFrame) -> int:  # noqa: N803
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
