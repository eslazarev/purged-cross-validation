"""Internal: purge primitive (Domain D2).

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 7 section 7.4.1 Snippet 7.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from purgedcv._intervals import overlaps_any_half_open_interval
from purgedcv._time import _coerce_1d, validate_times
from purgedcv._validation import _validate_positional_indices

from ._typing import NDArrayAny, TimesLike


def purge(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: TimesLike,
    evaluation_times: TimesLike,
    purge_horizon: pd.Timedelta | None = None,
) -> NDArrayAny:
    """Drop training rows whose half-open label horizon overlaps any test horizon.

    Each test row contributes a half-open horizon
    ``[prediction_time - purge_horizon, evaluation_time + purge_horizon)``.
    The intervals are merged before filtering, so disjoint test blocks purge
    only their local overlap zones rather than the full convex hull between
    them.

    A training row at position ``i`` is kept iff
    its label horizon overlaps none of those merged test horizons.

    Args:
        train_idx: positional indices of training rows.
        test_idx: positional indices of test rows.
        prediction_times: prediction times for all rows.
        evaluation_times: evaluation times for all rows.
        purge_horizon: optional symmetric padding (``None`` ≡ zero).

    Returns:
        The subset of ``train_idx`` whose horizons do not overlap the test
        window. Input ordering and dtype are preserved.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv import purge
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=3)
        >>> train_idx = np.array([0, 1, 2, 3, 4, 8, 9])
        >>> test_idx = np.array([5, 6, 7])
        >>> purge(train_idx, test_idx, pred, evalu)
        array([0, 1, 2])
    """
    horizon = purge_horizon if purge_horizon is not None else pd.Timedelta(0)
    if pd.isna(horizon):
        raise ValueError("purge_horizon must be non-missing, got NaT.")
    if horizon < pd.Timedelta(0):
        raise ValueError(f"purge_horizon must be non-negative, got {horizon}.")
    prediction_times = _coerce_1d(prediction_times, name="prediction_times")
    evaluation_times = _coerce_1d(evaluation_times, name="evaluation_times")
    validate_times(prediction_times, evaluation_times, require_monotonic=False)
    n_samples = len(prediction_times)
    train_idx = _validate_positional_indices("train_idx", train_idx, n_samples=n_samples)
    test_idx = _validate_positional_indices("test_idx", test_idx, n_samples=n_samples)
    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.asarray(train_idx)

    train_pred = prediction_times[train_idx]
    train_eval = evaluation_times[train_idx]
    test_starts = prediction_times[test_idx] - horizon
    test_ends = evaluation_times[test_idx] + horizon

    keep = ~overlaps_any_half_open_interval(train_pred, train_eval, test_starts, test_ends)
    kept: NDArrayAny = train_idx[keep]
    return kept
