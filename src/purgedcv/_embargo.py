"""Internal: embargo primitive (Domain D3).

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 7 section 7.4.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_embargo(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    prediction_times: pd.Series,
    evaluation_times: pd.Series,
    embargo: pd.Timedelta,
) -> np.ndarray:
    """Drop training rows whose ``prediction_time`` falls inside the closed
    embargo window ``[test_eval_max, test_eval_max + embargo]``.

    Embargo is asymmetric: rows whose ``prediction_time`` is strictly before
    ``test_eval_max`` are never dropped. ``embargo == 0`` is the identity
    (the embargo window is logically empty at zero width), avoiding the
    degenerate single-point case.

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
    if len(train_idx) == 0 or len(test_idx) == 0:
        return np.asarray(train_idx)
    if embargo <= pd.Timedelta(0):
        return np.asarray(train_idx)

    test_eval_max = evaluation_times.iloc[test_idx].max()
    cutoff = test_eval_max + embargo

    train_pred = prediction_times.iloc[train_idx].to_numpy()
    in_embargo = (train_pred >= test_eval_max.to_numpy()) & (train_pred <= cutoff.to_numpy())
    kept: np.ndarray = train_idx[~in_embargo]
    return kept
