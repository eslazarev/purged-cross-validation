"""Internal: WalkForwardSplit splitter (Domain D5.1)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import numpy as np
import pandas as pd

from purgedcv._base import BaseTemporalSplitter
from purgedcv._time import HorizonLike
from purgedcv._validation import _validate_integer

from ._typing import NDArrayAny

WindowMode = Literal["sliding", "expanding"]


class WalkForwardSplit(BaseTemporalSplitter):
    """Walk-forward CV with sliding or expanding training window.

    For ``window="expanding"`` and zero purge/embargo, this matches
    :class:`sklearn.model_selection.TimeSeriesSplit`. For
    ``window="sliding"``, the training window has a fixed maximum length
    (the most recent ``train_size`` samples before the test fold).

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv import WalkForwardSplit
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> cv = WalkForwardSplit(
        ...     n_splits=3, test_size=2,
        ...     prediction_times=pred, evaluation_times=evalu,
        ... )
        >>> cv.get_n_splits()
        3
    """

    def __init__(
        self,
        n_splits: int,
        test_size: int,
        *,
        train_size: int | None = None,
        window: WindowMode = "expanding",
        prediction_times: pd.Series,
        evaluation_times: pd.Series,
        purge_horizon: HorizonLike | None = None,
        embargo: HorizonLike | None = None,
    ) -> None:
        """Configure a walk-forward CV splitter.

        Args:
            n_splits: Number of folds to yield. Test folds tile the END of
                the dataset; each fold trains only on data strictly before
                its test indices.
            test_size: Number of consecutive rows in each test fold.
                ``n_splits * test_size`` must be strictly less than
                ``n_samples``.
            train_size: Maximum number of training rows per fold AFTER
                purge and embargo. Counts kept rows, not raw indices, so
                with a 2-day embargo and ``train_size=100`` the effective
                training set may use indices spanning more than 100 rows.
                Must be ``None`` when ``window='expanding'`` and a positive
                integer when ``window='sliding'``.
            window: ``"expanding"`` (default) uses all pre-test data as
                training; matches :class:`sklearn.model_selection.TimeSeriesSplit`
                when purge and embargo are zero. ``"sliding"`` caps each
                training set at the most recent ``train_size`` rows after
                purge and embargo.
            prediction_times: Per-sample prediction times for the dataset.
                Bound at construction so :meth:`split` matches the sklearn
                signature.
            evaluation_times: Per-sample evaluation times. Required to
                apply purge and embargo correctly.
            purge_horizon: Symmetric padding around the test fold's label
                window; training rows whose label horizon overlaps the
                padded test horizon are dropped. ``None`` means no purge.
            embargo: Post-test embargo duration; training rows whose
                prediction time falls in the closed window
                ``[test_eval_max, test_eval_max + embargo]`` are dropped.
                ``None`` means no embargo.

        Raises:
            ValueError: if ``window`` is not one of the literal values, if
                ``n_splits < 1`` or ``test_size < 1``, if
                ``train_size`` is supplied with ``window='expanding'``, or
                if ``train_size`` is ``None`` with ``window='sliding'``.
        """
        super().__init__(
            prediction_times=prediction_times,
            evaluation_times=evaluation_times,
            purge_horizon=purge_horizon,
            embargo=embargo,
        )
        if window not in ("sliding", "expanding"):
            raise ValueError(f"window must be 'sliding' or 'expanding', got {window!r}.")
        if window == "expanding" and train_size is not None:
            raise ValueError(
                "train_size has no effect when window='expanding'; "
                "pass train_size=None or use window='sliding'."
            )
        if window == "sliding" and train_size is None:
            raise ValueError(
                "train_size is required when window='sliding'; "
                "pass a positive integer or use window='expanding'."
            )
        n_splits = _validate_integer("n_splits", n_splits, minimum=1)
        test_size = _validate_integer("test_size", test_size, minimum=1)
        if window == "sliding" and train_size is not None:
            train_size = _validate_integer("train_size", train_size, minimum=1)
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.window = window

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        return self.n_splits

    def _iter_test_indices(self, n_samples: int) -> list[NDArrayAny]:
        total_test = self.test_size * self.n_splits
        if total_test >= n_samples:
            raise ValueError(
                f"too many folds: n_splits * test_size = {total_test} >= n_samples = {n_samples}."
            )
        first_test_start = n_samples - total_test
        return [
            np.arange(
                first_test_start + k * self.test_size,
                first_test_start + (k + 1) * self.test_size,
            )
            for k in range(self.n_splits)
        ]

    def _candidate_train_idx(self, n_samples: int, test_idx: NDArrayAny) -> NDArrayAny:
        """Walk-forward semantics: training set is everything strictly
        before the test fold (regardless of window mode). This overrides
        the base class default of 'everything except test'."""
        test_start = int(test_idx.min())
        return np.arange(test_start, dtype=np.int64)

    def split(
        self,
        X: NDArrayAny | pd.DataFrame,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> Iterator[tuple[NDArrayAny, NDArrayAny]]:
        """Yield ``(train_idx, test_idx)`` pairs for each walk-forward fold.

        The base class handles X length validation, purge, and embargo;
        this override additionally trims each training set to the most
        recent ``train_size`` rows when ``window="sliding"``. Training
        indices are always strictly less than the minimum index of the
        test fold (the defining walk-forward property).

        The number of pairs yielded equals ``self.n_splits``. The ``y``
        and ``groups`` arguments are accepted for sklearn protocol
        compatibility but ignored.
        """
        for train_idx, test_idx in super().split(X, y=y, groups=groups):
            if self.window == "sliding" and self.train_size is not None:
                train_idx = train_idx[-self.train_size :]
            yield train_idx, test_idx
