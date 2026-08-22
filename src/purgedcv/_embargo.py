"""Internal: embargo primitive (Domain D3).

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 7 section 7.4.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from purgedcv._intervals import points_in_any_closed_interval
from purgedcv._time import HorizonLike, _coerce_1d, parse_horizon, validate_times
from purgedcv._validation import _validate_integer, _validate_positional_indices

from ._typing import NDArrayAny, TimesLike


def _validate_embargo_modes(
    embargo: HorizonLike | None,
    embargo_observations: int | None,
    embargo_fraction: float | None,
) -> tuple[pd.Timedelta | None, int | None, float | None]:
    """Validate the three mutually-exclusive embargo modes."""
    supplied = sum(value is not None for value in (embargo, embargo_observations, embargo_fraction))
    if supplied > 1:
        raise ValueError(
            "embargo, embargo_observations, and embargo_fraction are mutually exclusive; "
            "supply at most one."
        )

    if embargo is not None:
        missing = pd.isna(embargo)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            raise ValueError("embargo must be non-missing, got NaT.")
    duration = parse_horizon(embargo) if embargo is not None else None
    observations = (
        _validate_integer("embargo_observations", embargo_observations, minimum=0)
        if embargo_observations is not None
        else None
    )
    fraction: float | None = None
    if embargo_fraction is not None:
        if isinstance(embargo_fraction, bool) or not isinstance(
            embargo_fraction, (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                "embargo_fraction must be a real number in [0, 1], got "
                f"{type(embargo_fraction).__name__}."
            )
        fraction = float(embargo_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(f"embargo_fraction must be in [0, 1], got {embargo_fraction}.")
    return duration, observations, fraction


def _positional_embargo_mask(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    embargo_observations: int,
) -> NDArrayAny:
    """Return a mask for train rows in the post-test positional windows."""
    if embargo_observations == 0 or len(train_idx) == 0 or len(test_idx) == 0:
        return np.zeros(len(train_idx), dtype=bool)

    sorted_test = np.sort(test_idx)
    run_ends = sorted_test[np.r_[np.diff(sorted_test) > 1, True]]
    in_embargo = np.zeros(len(train_idx), dtype=bool)
    for run_end in run_ends:
        in_embargo |= (train_idx > run_end) & (train_idx <= int(run_end) + embargo_observations)
    return in_embargo


def apply_embargo(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: TimesLike,
    evaluation_times: TimesLike,
    embargo: HorizonLike | None = None,
    *,
    embargo_observations: int | None = None,
    embargo_fraction: float | None = None,
) -> NDArrayAny:
    """Drop training rows inside a post-test embargo window.

    Choose at most one embargo mode:

    - ``embargo`` uses a wall-clock duration and drops rows whose
      ``prediction_time`` falls inside any closed window
      ``[test_evaluation_time, test_evaluation_time + embargo]``;
    - ``embargo_observations`` drops that many row positions immediately
      after each contiguous test block;
    - ``embargo_fraction`` drops ``floor(n_samples * fraction)`` row positions
      after each contiguous test block.

    Embargo is asymmetric: only rows after a test boundary are dropped.
    ``embargo == 0``, ``embargo_observations == 0``, and a fractional mode
    that rounds to zero are identities.

    Args:
        train_idx: positional indices of training rows.
        test_idx: positional indices of test rows.
        prediction_times: prediction times for all rows.
        evaluation_times: evaluation times for all rows.
        embargo: post-test embargo duration.
        embargo_observations: number of row positions to drop after each
            contiguous test block.
        embargo_fraction: fraction of the full dataset to drop after each
            contiguous test block. Must be between 0 and 1 inclusive; the
            observation count is rounded down.

    Returns:
        The subset of ``train_idx`` outside the post-test embargo windows.
        Input ordering and dtype are preserved.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv import apply_embargo
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> train_idx = np.array([11, 12, 13, 14])
        >>> test_idx = np.arange(5, 10)
        >>> apply_embargo(train_idx, test_idx, pred, evalu, pd.Timedelta(days=1))
        array([12, 13, 14])
    """
    duration, observations, fraction = _validate_embargo_modes(
        embargo, embargo_observations, embargo_fraction
    )
    prediction_times = _coerce_1d(prediction_times, name="prediction_times")
    evaluation_times = _coerce_1d(evaluation_times, name="evaluation_times")
    validate_times(prediction_times, evaluation_times, require_monotonic=False)
    n_samples = len(prediction_times)
    train_idx = _validate_positional_indices("train_idx", train_idx, n_samples=n_samples)
    test_idx = _validate_positional_indices("test_idx", test_idx, n_samples=n_samples)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.asarray(train_idx)
    if duration is None and observations is None and fraction is None:
        return np.asarray(train_idx)

    if fraction is not None:
        observations = int(n_samples * fraction)
    if observations is not None:
        in_embargo = _positional_embargo_mask(train_idx, test_idx, observations)
        kept: NDArrayAny = train_idx[~in_embargo]
        return kept

    if duration == pd.Timedelta(0):
        return np.asarray(train_idx)
    if duration is None:  # pragma: no cover - exhaustive mode validation above
        raise RuntimeError("embargo mode was not resolved")

    train_pred = prediction_times[train_idx]
    embargo_starts = evaluation_times[test_idx]
    embargo_ends = evaluation_times[test_idx] + duration
    in_embargo = points_in_any_closed_interval(train_pred, embargo_starts, embargo_ends)
    kept_by_time: NDArrayAny = train_idx[~in_embargo]
    return kept_by_time
