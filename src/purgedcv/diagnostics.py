"""Leakage diagnostics (Domain D8).

These functions exist for two purposes:

1. Tests of internal splitters assert, across random fuzz inputs, that
   their output is leakage-free.
2. Users audit custom splits they have built by hand.

Each ``assert_*`` raises a specific :class:`TemporalCVError` subclass with
the offending row index in the message. The non-raising helpers return either
a scalar overlap summary or a per-fold splitter audit suitable for logging,
review, and regression checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from purgedcv._base import BaseTemporalSplitter
from purgedcv._embargo import _positional_embargo_mask, _validate_embargo_modes
from purgedcv._intervals import overlaps_any_half_open_interval, points_in_any_closed_interval
from purgedcv._time import HorizonLike, _coerce_1d, parse_horizon, validate_times
from purgedcv._validation import _validate_positional_indices
from purgedcv.exceptions import (
    EmbargoViolationError,
    GroupLeakageError,
    TemporalLeakageError,
)

from ._typing import ArrayLike1D, NDArrayAny, TimesLike

__all__ = [
    "assert_embargo_respected",
    "assert_groups_disjoint",
    "assert_no_temporal_leakage",
    "audit_splitter",
    "compute_overlap_fraction",
]

_AUDIT_COLUMNS = (
    "fold",
    "candidate_train_size",
    "final_train_size",
    "test_size",
    "train_nonempty",
    "rows_removed_by_purge",
    "rows_removed_by_embargo",
    "rows_removed_by_finalization",
    "rows_added_by_finalization",
    "candidate_overlap_fraction",
    "final_overlap_fraction",
    "temporal_leakage_free",
    "train_block_count",
    "test_block_count",
    "train_time_envelope_start",
    "train_time_envelope_end",
    "test_time_envelope_start",
    "test_time_envelope_end",
    "groups_disjoint",
)


def _overlap_fraction_from_arrays(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: NDArrayAny,
    evaluation_times: NDArrayAny,
    purge_horizon: pd.Timedelta,
) -> float:
    """Compute overlap after public-boundary validation has already run."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        return 0.0
    train_pred = prediction_times[train_idx]
    train_eval = evaluation_times[train_idx]
    test_starts = prediction_times[test_idx] - purge_horizon
    test_ends = evaluation_times[test_idx] + purge_horizon
    overlaps = overlaps_any_half_open_interval(train_pred, train_eval, test_starts, test_ends)
    return float(overlaps.mean())


def _nat_like(values: NDArrayAny) -> object:
    """Return a NaT scalar preserving a temporal array's dtype family/unit."""
    if not (
        np.issubdtype(values.dtype, np.datetime64) or np.issubdtype(values.dtype, np.timedelta64)
    ):
        raise TypeError(f"expected a temporal dtype, got {values.dtype}.")
    nat: object = np.asarray("NaT", dtype=values.dtype)[()]
    return nat


def _time_envelope(
    indices: NDArrayAny,
    prediction_times: NDArrayAny,
    evaluation_times: NDArrayAny,
) -> tuple[object, object]:
    """Return the outer label-horizon envelope for positional indices."""
    if len(indices) == 0:
        return _nat_like(prediction_times), _nat_like(evaluation_times)
    return prediction_times[indices].min(), evaluation_times[indices].max()


def _contiguous_block_count(indices: NDArrayAny) -> int:
    """Count contiguous positional-index runs, independent of input order."""
    if len(indices) == 0:
        return 0
    ordered = np.sort(indices)
    return int(1 + np.count_nonzero(np.diff(ordered) > 1))


def _group_overlap_from_values(
    train_group_values: NDArrayAny,
    test_group_values: NDArrayAny,
) -> set[Any]:
    """Return group identifiers shared by train and test arrays."""
    train_groups = set(train_group_values.tolist())
    test_groups = set(test_group_values.tolist())
    return train_groups & test_groups


def _index_membership_changes(
    before: NDArrayAny,
    after: NDArrayAny,
) -> tuple[int, int]:
    """Count indices removed from and added to a finalization stage."""
    before_set = set(before.tolist())
    after_set = set(after.tolist())
    return len(before_set - after_set), len(after_set - before_set)


def audit_splitter(
    cv: BaseTemporalSplitter,
    X: NDArrayAny | pd.DataFrame,  # noqa: N803
) -> pd.DataFrame:
    """Return a non-raising, per-fold report for a temporal splitter.

    The report consumes the same candidate → purge → embargo → final pipeline
    as :meth:`BaseTemporalSplitter.split`; purge and embargo counts therefore
    come from the actual intermediate index arrays rather than being inferred
    from the final split. Sliding walk-forward truncation is implemented by
    the finalization stage and therefore appears in
    ``rows_removed_by_finalization``. If a custom finalization hook introduces
    indices, ``rows_added_by_finalization`` reports them separately.

    ``candidate_overlap_fraction`` is the fraction of candidate training rows
    whose label horizons overlap the test horizons after applying the
    splitter's ``purge_horizon`` padding. For non-empty candidates it equals
    ``rows_removed_by_purge / candidate_train_size``; empty candidates report
    zero. :func:`compute_overlap_fraction` reproduces it only when given the
    same pre-purge candidate indices and purge horizon — indices returned by
    ``cv.split()`` are already final. ``final_overlap_fraction`` repeats the
    measure on final training rows and should be zero for a clean splitter.

    For the built-in splitters, ``temporal_leakage_free`` is structurally
    expected to be ``True``: purge removes overlaps and later stages only
    remove rows. The column is primarily a regression guard for custom
    :class:`BaseTemporalSplitter` subclasses whose finalization hook could
    accidentally reintroduce indices; it is not an independent validation
    algorithm. For an empty final train set it is vacuously ``True``; inspect
    ``train_nonempty`` before treating a fold as usable. Leakage and group
    overlap are reported as values instead of raising; malformed inputs and
    splitter configuration errors still raise.

    Time columns are outer envelopes, not continuous windows. CPCV and
    interleaved group folds can contain several disjoint positional blocks, so
    train and test envelopes may overlap even when
    ``final_overlap_fraction == 0``. ``train_block_count`` and
    ``test_block_count`` expose that layout; the counts refer to contiguous
    runs of positional indices.

    The audit requires the inherited :meth:`BaseTemporalSplitter.split`
    implementation. A subclass overriding ``split()`` is rejected because its
    returned folds may diverge from the auditable candidate → purge → embargo
    → finalization pipeline. Custom subclasses should use
    ``_iter_test_indices``, ``_candidate_train_idx``, and
    ``_finalize_train_idx`` instead.

    Args:
        cv: A :class:`BaseTemporalSplitter` instance with times already bound.
        X: Feature matrix or other sized sample container. Its length must
            match the times bound to ``cv``; feature values are not inspected.

    Returns:
        A DataFrame with one row per fold. Columns contain the zero-based fold
        number; candidate, final, and test sizes; train non-emptiness; rows
        removed at purge, embargo, and finalization; indices added by custom
        finalization; candidate and final temporal-overlap fractions;
        contiguous block counts; outer train/test time envelopes; and
        ``groups_disjoint`` (``None`` when no groups are bound).

    Raises:
        TypeError: if ``cv`` is not a :class:`BaseTemporalSplitter` or its
            class overrides :meth:`BaseTemporalSplitter.split`.
        ValueError: if ``X`` has the wrong length or the splitter cannot form
            its configured folds.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv import PurgedKFold, audit_splitter
        >>> pred = pd.date_range("2024-01-01", periods=20, freq="D")
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> cv = PurgedKFold(
        ...     4, prediction_times=pred, evaluation_times=evalu,
        ...     purge_horizon="1D", embargo_observations=2,
        ... )
        >>> report = audit_splitter(cv, np.zeros((20, 1)))
        >>> report.shape[0]
        4
        >>> bool(report["final_overlap_fraction"].eq(0.0).all())
        True
    """
    if not isinstance(cv, BaseTemporalSplitter):
        raise TypeError(
            f"cv must be a purgedcv BaseTemporalSplitter instance, got {type(cv).__name__}."
        )
    if type(cv).split is not BaseTemporalSplitter.split:
        raise TypeError(
            "audit_splitter cannot audit a subclass that overrides split(); "
            "customize _iter_test_indices, _candidate_train_idx, or "
            "_finalize_train_idx while inheriting BaseTemporalSplitter.split."
        )

    rows: list[dict[str, object]] = []
    prediction_times = cv._prediction_times
    evaluation_times = cv._evaluation_times
    groups = cv._groups
    for fold, stages in enumerate(cv._iter_split_stages(X)):
        candidate_overlap = _overlap_fraction_from_arrays(
            stages.candidate_train_idx,
            stages.test_idx,
            prediction_times,
            evaluation_times,
            cv.purge_horizon,
        )
        final_overlap = _overlap_fraction_from_arrays(
            stages.final_train_idx,
            stages.test_idx,
            prediction_times,
            evaluation_times,
            cv.purge_horizon,
        )
        train_envelope_start, train_envelope_end = _time_envelope(
            stages.final_train_idx, prediction_times, evaluation_times
        )
        test_envelope_start, test_envelope_end = _time_envelope(
            stages.test_idx, prediction_times, evaluation_times
        )
        groups_disjoint: bool | None = None
        if groups is not None:
            train_group_values = groups[stages.final_train_idx]
            test_group_values = groups[stages.test_idx]
            groups_disjoint = not _group_overlap_from_values(train_group_values, test_group_values)
        removed_by_finalization, added_by_finalization = _index_membership_changes(
            stages.embargoed_train_idx, stages.final_train_idx
        )

        rows.append(
            {
                "fold": fold,
                "candidate_train_size": len(stages.candidate_train_idx),
                "final_train_size": len(stages.final_train_idx),
                "test_size": len(stages.test_idx),
                "train_nonempty": len(stages.final_train_idx) > 0,
                "rows_removed_by_purge": len(stages.candidate_train_idx)
                - len(stages.purged_train_idx),
                "rows_removed_by_embargo": len(stages.purged_train_idx)
                - len(stages.embargoed_train_idx),
                "rows_removed_by_finalization": removed_by_finalization,
                "rows_added_by_finalization": added_by_finalization,
                "candidate_overlap_fraction": candidate_overlap,
                "final_overlap_fraction": final_overlap,
                "temporal_leakage_free": final_overlap == 0.0,
                "train_block_count": _contiguous_block_count(stages.final_train_idx),
                "test_block_count": _contiguous_block_count(stages.test_idx),
                "train_time_envelope_start": train_envelope_start,
                "train_time_envelope_end": train_envelope_end,
                "test_time_envelope_start": test_envelope_start,
                "test_time_envelope_end": test_envelope_end,
                "groups_disjoint": groups_disjoint,
            }
        )
    return pd.DataFrame.from_records(rows, columns=_AUDIT_COLUMNS)


def assert_no_temporal_leakage(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: TimesLike,
    evaluation_times: TimesLike,
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
    horizon = parse_horizon(purge_horizon) if purge_horizon is not None else pd.Timedelta(0)
    prediction_times = _coerce_1d(prediction_times, name="prediction_times")
    evaluation_times = _coerce_1d(evaluation_times, name="evaluation_times")
    validate_times(prediction_times, evaluation_times, require_monotonic=False)
    n_samples = len(prediction_times)
    train_idx = _validate_positional_indices("train_idx", train_idx, n_samples=n_samples)
    test_idx = _validate_positional_indices("test_idx", test_idx, n_samples=n_samples)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return

    train_pred = prediction_times[train_idx]
    train_eval = evaluation_times[train_idx]
    test_starts = prediction_times[test_idx] - horizon
    test_ends = evaluation_times[test_idx] + horizon

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
    prediction_times: TimesLike,
    evaluation_times: TimesLike,
    embargo: HorizonLike | None = None,
    *,
    embargo_observations: int | None = None,
    embargo_fraction: float | None = None,
) -> None:
    """Raise :class:`EmbargoViolationError` when an embargo is violated.

    The duration, observation-count, and fractional modes have the same
    mutually-exclusive semantics as :func:`purgedcv.apply_embargo`. Exactly
    one mode must be supplied; omitting all three is treated as a configuration
    error rather than a successful audit.

    Embargo is asymmetric: rows before test boundaries are never flagged.
    Zero-width modes are identities.

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
    duration, observations, fraction = _validate_embargo_modes(
        embargo, embargo_observations, embargo_fraction
    )
    if duration is None and observations is None and fraction is None:
        raise ValueError(
            "assert_embargo_respected requires one of embargo, "
            "embargo_observations, or embargo_fraction."
        )
    prediction_times = _coerce_1d(prediction_times, name="prediction_times")
    evaluation_times = _coerce_1d(evaluation_times, name="evaluation_times")
    validate_times(prediction_times, evaluation_times, require_monotonic=False)
    n_samples = len(prediction_times)
    train_idx = _validate_positional_indices("train_idx", train_idx, n_samples=n_samples)
    test_idx = _validate_positional_indices("test_idx", test_idx, n_samples=n_samples)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    if fraction is not None:
        observations = int(n_samples * fraction)
    if observations is not None:
        in_embargo = _positional_embargo_mask(train_idx, test_idx, observations)
        if in_embargo.any():
            first_global = int(train_idx[int(in_embargo.argmax())])
            raise EmbargoViolationError(
                f"Embargo violation at row {first_global}: row position falls "
                "inside a post-test positional embargo window."
            )
        return

    if duration == pd.Timedelta(0):
        return
    if duration is None:  # pragma: no cover - exhaustive mode validation above
        raise RuntimeError("embargo mode was not resolved")

    train_pred = prediction_times[train_idx]
    embargo_starts = evaluation_times[test_idx]
    embargo_ends = evaluation_times[test_idx] + duration
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
    groups: ArrayLike1D,
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
    groups = _coerce_1d(groups, name="groups")
    train_idx = _validate_positional_indices("train_idx", train_idx, n_samples=len(groups))
    test_idx = _validate_positional_indices("test_idx", test_idx, n_samples=len(groups))
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    train_group_values = groups[train_idx]
    test_group_values = groups[test_idx]
    if pd.isna(train_group_values).any() or pd.isna(test_group_values).any():
        raise ValueError("groups contains missing values in train or test indices.")
    overlap = _group_overlap_from_values(train_group_values, test_group_values)
    if overlap:
        offender = next(iter(sorted(overlap, key=str)))
        raise GroupLeakageError(
            f"Group leakage: group {offender} appears in both train and test "
            f"(total overlapping groups: {len(overlap)})."
        )


def compute_overlap_fraction(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: TimesLike,
    evaluation_times: TimesLike,
    *,
    purge_horizon: HorizonLike | None = None,
) -> float:
    """Return the fraction of training rows whose half-open label horizon
    overlaps any test horizon, optionally padded by ``purge_horizon``.

    Unlike :func:`assert_no_temporal_leakage`, this does not raise when it
    finds leakage: it returns ``0.0`` for clean splits and ``1.0`` when every
    training row leaks. Malformed indices, times, and horizons still raise
    validation errors. Useful for logging splitter health metrics or for
    debugging a splitter that produces unexpected behavior.

    ``purge_horizon`` defaults to no padding. Reproducing an
    ``audit_splitter`` report's ``candidate_overlap_fraction`` also requires
    its pre-purge candidate indices; the indices yielded by ``cv.split()`` are
    final and instead reproduce ``final_overlap_fraction``.

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
    horizon = parse_horizon(purge_horizon) if purge_horizon is not None else pd.Timedelta(0)
    prediction_times = _coerce_1d(prediction_times, name="prediction_times")
    evaluation_times = _coerce_1d(evaluation_times, name="evaluation_times")
    validate_times(prediction_times, evaluation_times, require_monotonic=False)
    n_samples = len(prediction_times)
    train_idx = _validate_positional_indices("train_idx", train_idx, n_samples=n_samples)
    test_idx = _validate_positional_indices("test_idx", test_idx, n_samples=n_samples)
    return _overlap_fraction_from_arrays(
        train_idx,
        test_idx,
        prediction_times,
        evaluation_times,
        horizon,
    )
