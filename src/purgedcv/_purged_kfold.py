"""Internal: PurgedKFold and PurgedGroupKFold (Domains D5.2 + D5.3).

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 7 section 7.4, for the figure that defines the k-fold layout this
class implements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from purgedcv._base import BaseTemporalSplitter
from purgedcv._time import HorizonLike
from purgedcv._validation import _validate_integer

from ._typing import NDArrayAny


class PurgedKFold(BaseTemporalSplitter):
    """K-fold CV with contiguous test folds and purge + embargo applied.

    Test folds tile the index space contiguously: fold k holds rows
    ``[start_k, start_k + size_k)`` where ``size_k`` differs across folds
    by at most one due to integer division. Train = complement of the
    test fold, with D2 purge and D3 embargo applied by the base class.

    For zero ``purge_horizon`` and ``embargo`` the test folds are
    identical to :class:`sklearn.model_selection.KFold(shuffle=False)`.
    Purge still drops any training row whose own label horizon
    ``[prediction_time, evaluation_time)`` overlaps the test labels, so
    the training set equals the full complement only when no two label
    horizons overlap (for instance when ``evaluation_times`` equals
    ``prediction_times``). The splitter then degrades exactly to
    ``KFold(shuffle=False)``.

    The first ``n_samples % n_splits`` folds receive
    ``n_samples // n_splits + 1`` rows; the remaining folds receive
    ``n_samples // n_splits``. When ``n_splits > n_samples``, the
    excess folds are empty — the base class purge and embargo handle
    empty arrays gracefully.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv import PurgedKFold
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> cv = PurgedKFold(
        ...     n_splits=5,
        ...     prediction_times=pred, evaluation_times=evalu,
        ... )
        >>> cv.get_n_splits()
        5
    """

    def __init__(
        self,
        n_splits: int,
        *,
        prediction_times: pd.Series,
        evaluation_times: pd.Series,
        purge_horizon: HorizonLike | None = None,
        embargo: HorizonLike | None = None,
    ) -> None:
        """Configure a purged k-fold splitter.

        Args:
            n_splits: Number of folds. Must be at least 2.
            prediction_times: Per-sample prediction times. Bound at
                construction so :meth:`split` matches the sklearn signature.
            evaluation_times: Per-sample evaluation times.
            purge_horizon: Symmetric padding around the test fold's label
                window; training rows whose label horizon overlaps the
                padded test horizon are dropped. ``None`` means no purge.
            embargo: Post-test embargo duration; training rows whose
                prediction time falls in any closed window
                ``[test_evaluation_time, test_evaluation_time + embargo]``
                are dropped.
                ``None`` means no embargo.

        Raises:
            ValueError: if ``n_splits < 2``.
        """
        n_splits = _validate_integer("n_splits", n_splits, minimum=2)
        super().__init__(
            prediction_times=prediction_times,
            evaluation_times=evaluation_times,
            purge_horizon=purge_horizon,
            embargo=embargo,
        )
        self.n_splits = n_splits

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        return self.n_splits

    def _iter_test_indices(self, n_samples: int) -> list[NDArrayAny]:
        fold_size, remainder = divmod(n_samples, self.n_splits)
        # Distribute the remainder across the first `remainder` folds so
        # all samples are used and fold sizes differ by at most 1.
        cursor = 0
        out: list[NDArrayAny] = []
        for k in range(self.n_splits):
            sz = fold_size + (1 if k < remainder else 0)
            out.append(np.arange(cursor, cursor + sz, dtype=np.int64))
            cursor += sz
        return out


class PurgedGroupKFold(BaseTemporalSplitter):
    """Group-aware k-fold variant of :class:`PurgedKFold`.

    Each test fold consists of all rows from a contiguous block of unique
    group identifiers, so no ``group_id`` ever appears in both train and
    test of the same fold. Purge and embargo still apply ACROSS group
    boundaries — training rows from other groups whose horizons overlap
    the test window are dropped by D2/D3 in the base class.

    Groups are assigned to fold blocks in first-appearance order within
    the ``groups`` Series (the order returned by ``pd.Series.unique()``).
    For temporal coherence — patients enrolled in chronological order,
    assets in time-of-IPO order — ensure the ``groups`` Series is sorted
    by first occurrence time.

    The base class's :func:`assert_groups_disjoint` enforcement runs on
    every fold automatically because ``groups`` is bound at construction.

    Useful in clinical ML (no patient appears in both sides of a fold),
    asset CV (no symbol appears in both sides), and any setting where
    grouped observations must not leak across the train/test boundary.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv import PurgedGroupKFold
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=12, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> groups = pd.Series(np.repeat([0, 1, 2, 3], 3))
        >>> cv = PurgedGroupKFold(
        ...     n_splits=2,
        ...     prediction_times=pred, evaluation_times=evalu,
        ...     groups=groups,
        ... )
        >>> cv.get_n_splits()
        2
    """

    def __init__(
        self,
        n_splits: int,
        *,
        prediction_times: pd.Series,
        evaluation_times: pd.Series,
        groups: pd.Series,
        purge_horizon: HorizonLike | None = None,
        embargo: HorizonLike | None = None,
    ) -> None:
        """Configure a group-aware purged k-fold splitter.

        Args:
            n_splits: Number of folds. Must be at least 2 and must not
                exceed the number of unique group identifiers.
            prediction_times: Per-sample prediction times.
            evaluation_times: Per-sample evaluation times.
            groups: Group identifier per sample. Must have the same
                length as ``prediction_times``. Unique groups are
                partitioned into ``n_splits`` contiguous blocks in the
                order returned by ``pd.Series.unique()``.
            purge_horizon: Symmetric padding around the test fold's
                label window; cross-group training rows whose label
                horizon overlaps the padded test horizon are dropped.
                ``None`` means no purge.
            embargo: Post-test embargo duration. ``None`` means no
                embargo.

        Raises:
            ValueError: if ``n_splits < 2`` or
                ``n_splits > len(groups.unique())``.
        """
        n_splits = _validate_integer("n_splits", n_splits, minimum=2)
        super().__init__(
            prediction_times=prediction_times,
            evaluation_times=evaluation_times,
            purge_horizon=purge_horizon,
            embargo=embargo,
            groups=groups,
        )
        unique = list(groups.unique())
        if n_splits > len(unique):
            raise ValueError(
                f"n_splits={n_splits} exceeds number of unique groups ({len(unique)})."
            )
        self.n_splits = n_splits
        self._unique_groups = unique

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        return self.n_splits

    def _iter_test_indices(self, n_samples: int) -> list[NDArrayAny]:
        if self._groups is None:  # bound at construction; should be unreachable
            raise RuntimeError(
                "PurgedGroupKFold._groups was unbound at split time. "
                "This is an internal invariant violation."
            )
        groups_array = self._groups.to_numpy()
        n_groups = len(self._unique_groups)
        block_size, remainder = divmod(n_groups, self.n_splits)
        cursor = 0
        out: list[NDArrayAny] = []
        for k in range(self.n_splits):
            sz = block_size + (1 if k < remainder else 0)
            block = self._unique_groups[cursor : cursor + sz]
            cursor += sz
            mask = np.isin(groups_array, block)
            out.append(np.where(mask)[0].astype(np.int64))
        return out
