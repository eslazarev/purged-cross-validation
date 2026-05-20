"""Leakage diagnostics (Domain D8).

These functions exist for two purposes:

1. Tests of internal splitters assert, across random fuzz inputs, that
   their output is leakage-free.
2. Users audit custom splits they have built by hand.

Each ``assert_*`` raises a specific :class:`TemporalCVError` subclass with
the offending row index in the message; the non-raising
:func:`compute_overlap_fraction` returns a scalar summary suitable for
logging or sanity-checking.
"""

from __future__ import annotations

import pandas as pd

from purgedcv._intervals import overlaps_any_half_open_interval, points_in_any_closed_interval
from purgedcv._time import HorizonLike, parse_horizon
from purgedcv.exceptions import (
    EmbargoViolationError,
    GroupLeakageError,
    TemporalLeakageError,
)

from ._typing import NDArrayAny

__all__ = [
    "assert_embargo_respected",
    "assert_groups_disjoint",
    "assert_no_temporal_leakage",
    "compute_overlap_fraction",
]


def assert_no_temporal_leakage(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: pd.Series,
    evaluation_times: pd.Series,
    *,
    purge_horizon: HorizonLike | None = None,
) -> None:
    """Raise :class:`TemporalLeakageError` if any training row's label horizon
    overlaps any test label horizon, optionally padded on both sides by
    ``purge_horizon``.

    Test horizons are checked as a union of half-open intervals, not as one
    convex hull. This matters for CPCV folds whose test groups are
    intentionally non-contiguous.

    Args:
        train_idx: positional indices of training rows.
        test_idx: positional indices of test rows.
        prediction_times: prediction times for all rows.
        evaluation_times: evaluation times for all rows.
        purge_horizon: optional padding (default: ``None`` ≡ zero padding).

    Raises:
        TemporalLeakageError: with the offending training row index and the
            two overlapping intervals in the message.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv.diagnostics import assert_no_temporal_leakage
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> assert_no_temporal_leakage(np.arange(5), np.arange(10, 15), pred, evalu)
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        return

    horizon = parse_horizon(purge_horizon) if purge_horizon is not None else pd.Timedelta(0)

    train_pred = prediction_times.iloc[train_idx].to_numpy()
    train_eval = evaluation_times.iloc[train_idx].to_numpy()
    test_starts = (prediction_times.iloc[test_idx] - horizon).to_numpy()
    test_ends = (evaluation_times.iloc[test_idx] + horizon).to_numpy()

    overlaps = overlaps_any_half_open_interval(train_pred, train_eval, test_starts, test_ends)
    if overlaps.any():
        first_local = int(overlaps.argmax())
        first_global = int(train_idx[first_local])
        raise TemporalLeakageError(
            f"Temporal leakage at row {first_global}: training horizon "
            f"[{train_pred[first_local]}, {train_eval[first_local]}) overlaps "
            "at least one test horizon."
        )


def assert_embargo_respected(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: pd.Series,
    evaluation_times: pd.Series,
    embargo: HorizonLike,
) -> None:
    """Raise :class:`EmbargoViolationError` if any training row's
    ``prediction_time`` falls inside any closed embargo window
    ``[test_evaluation_time, test_evaluation_time + embargo]``.

    Embargo is asymmetric: rows whose ``prediction_time`` is strictly before
    all test evaluation times are never flagged. ``embargo == 0`` is the
    identity (no rows flagged) — the embargo window is logically empty at
    zero width.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv.diagnostics import assert_embargo_respected
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> assert_embargo_respected(
        ...     np.array([18]), np.arange(5, 10), pred, evalu, embargo="2D"
        ... )
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        return

    emb = parse_horizon(embargo)
    if emb == pd.Timedelta(0):
        return

    train_pred = prediction_times.iloc[train_idx].to_numpy()
    embargo_starts = evaluation_times.iloc[test_idx].to_numpy()
    embargo_ends = (evaluation_times.iloc[test_idx] + emb).to_numpy()
    in_embargo = points_in_any_closed_interval(train_pred, embargo_starts, embargo_ends)
    if in_embargo.any():
        first_local = int(in_embargo.argmax())
        first_global = int(train_idx[first_local])
        raise EmbargoViolationError(
            f"Embargo violation at row {first_global}: prediction_time "
            f"{train_pred[first_local]} falls inside at least one post-test "
            "embargo window."
        )


def assert_groups_disjoint(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    groups: pd.Series,
) -> None:
    """Raise :class:`GroupLeakageError` if any group identifier appears in
    both ``train_idx`` and ``test_idx``.

    Used by group-aware splitters to verify that no entity (patient, asset,
    user, etc.) is represented in both training and test of the same fold.
    The error message names a representative overlapping group plus the
    total count of overlapping groups, so the caller can scope follow-up.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv.diagnostics import assert_groups_disjoint
        >>> groups = pd.Series([0, 0, 1, 1, 2, 2])
        >>> assert_groups_disjoint(np.array([0, 1]), np.array([2, 3]), groups)
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    train_groups = set(groups.iloc[train_idx].unique())
    test_groups = set(groups.iloc[test_idx].unique())
    overlap = train_groups & test_groups
    if overlap:
        offender = next(iter(sorted(overlap, key=str)))
        raise GroupLeakageError(
            f"Group leakage: group {offender} appears in both train and test "
            f"(total overlapping groups: {len(overlap)})."
        )


def compute_overlap_fraction(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: pd.Series,
    evaluation_times: pd.Series,
) -> float:
    """Return the fraction of training rows whose half-open label horizon
    overlaps any test horizon.

    Unlike :func:`assert_no_temporal_leakage`, this never raises — it
    returns ``0.0`` for clean splits and ``1.0`` when every training row
    leaks. Useful for logging splitter health metrics or for debugging
    a splitter that produces unexpected behavior.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv.diagnostics import compute_overlap_fraction
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> compute_overlap_fraction(np.arange(5), np.arange(10, 15), pred, evalu)
        0.0
        >>> compute_overlap_fraction(
        ...     np.arange(10, 15), np.arange(10, 15), pred, evalu
        ... )
        1.0
    """
    if len(train_idx) == 0 or len(test_idx) == 0:
        return 0.0
    train_pred = prediction_times.iloc[train_idx].to_numpy()
    train_eval = evaluation_times.iloc[train_idx].to_numpy()
    test_starts = prediction_times.iloc[test_idx].to_numpy()
    test_ends = evaluation_times.iloc[test_idx].to_numpy()
    overlaps = overlaps_any_half_open_interval(train_pred, train_eval, test_starts, test_ends)
    return float(overlaps.mean())
