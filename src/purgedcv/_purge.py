"""Internal: purge primitive (Domain D2).

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 7 section 7.4.1 Snippet 7.1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from purgedcv._intervals import overlaps_any_half_open_interval

from ._typing import NDArrayAny


def purge(
    train_idx: NDArrayAny,
    test_idx: NDArrayAny,
    prediction_times: pd.Series,
    evaluation_times: pd.Series,
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
    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.asarray(train_idx)

    train_pred = prediction_times.iloc[train_idx].to_numpy()
    train_eval = evaluation_times.iloc[train_idx].to_numpy()
    test_starts = (prediction_times.iloc[test_idx] - horizon).to_numpy()
    test_ends = (evaluation_times.iloc[test_idx] + horizon).to_numpy()

    keep = ~overlaps_any_half_open_interval(train_pred, train_eval, test_starts, test_ends)
    kept: NDArrayAny = train_idx[keep]
    return kept
