"""Internal: embargo primitive (Domain D3).

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 7 section 7.4.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from purgedcv._intervals import points_in_any_closed_interval
from purgedcv._time import _coerce_1d, validate_times
from purgedcv._validation import _validate_positional_indices

from ._typing import NDArrayAny, TimesLike


def apply_embargo(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: TimesLike,
    evaluation_times: TimesLike,
    embargo: pd.Timedelta,
) -> NDArrayAny:
    """Drop training rows whose ``prediction_time`` falls inside any closed
    embargo window ``[test_evaluation_time, test_evaluation_time + embargo]``.

    Embargo is asymmetric: rows whose ``prediction_time`` is strictly before
    all test evaluation times are never dropped. ``embargo == 0`` is the
    identity (the embargo window is logically empty at zero width), avoiding
    degenerate single-point windows.

    Args:
        train_idx: positional indices of training rows.
        test_idx: positional indices of test rows.
        prediction_times: prediction times for all rows.
        evaluation_times: evaluation times for all rows.
        embargo: post-test embargo duration.

    Returns:
        The subset of ``train_idx`` whose prediction_times fall outside the
        post-test embargo window. Input ordering and dtype are preserved.

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
    if pd.isna(embargo):
        raise ValueError("embargo must be non-missing, got NaT.")
    if embargo < pd.Timedelta(0):
        raise ValueError(f"embargo must be non-negative, got {embargo}.")
    prediction_times = _coerce_1d(prediction_times, name="prediction_times")
    evaluation_times = _coerce_1d(evaluation_times, name="evaluation_times")
    validate_times(prediction_times, evaluation_times, require_monotonic=False)
    n_samples = len(prediction_times)
    train_idx = _validate_positional_indices("train_idx", train_idx, n_samples=n_samples)
    test_idx = _validate_positional_indices("test_idx", test_idx, n_samples=n_samples)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.asarray(train_idx)
    if embargo == pd.Timedelta(0):
        return np.asarray(train_idx)

    train_pred = prediction_times[train_idx]
    embargo_starts = evaluation_times[test_idx]
    embargo_ends = evaluation_times[test_idx] + embargo
    in_embargo = points_in_any_closed_interval(train_pred, embargo_starts, embargo_ends)
    kept: NDArrayAny = train_idx[~in_embargo]
    return kept
