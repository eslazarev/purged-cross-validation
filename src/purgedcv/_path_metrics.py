"""Internal: per-path summary statistics for CPCV backtest paths (Domain D6).

:func:`~purgedcv.reconstruct_paths` (and
:meth:`CombinatorialPurgedCV.backtest_paths`) return an
``(n_paths, n_samples)`` matrix. The next step is almost always the same:
reduce each path to a handful of summary numbers (Sharpe, Calmar, drawdown,
total return). :func:`path_metrics` does that in one call and hands back a
tidy DataFrame so the whole path distribution can be described at once.

Each row of the input is treated as a per-period **return** (or PnL) series.
The default metric assumes additive returns; pass your own ``metric_fn`` for
a different convention (for example log vs simple compounding).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial

import numpy as np
import pandas as pd

from ._typing import NDArrayAny

#: A path metric: maps one path's 1-D return series to a name -> value mapping.
PathMetricFn = Callable[[NDArrayAny], Mapping[str, float]]


def default_backtest_metrics(
    path: NDArrayAny,
    *,
    bars_per_year: int | None = None,
) -> dict[str, float]:
    """Summarise one path of per-period returns.

    Non-finite entries (for example NaN from a fold that could not be fit)
    are dropped before the statistics are computed. A path with fewer than
    two finite returns yields all-NaN metrics.

    Returns are treated as **additive** (equity curve = cumulative sum), the
    convention for log returns. For simple returns supply a custom metric.

    Args:
        path: 1-D per-period return series.
        bars_per_year: If given, the Sharpe ratio is annualised by
            ``sqrt(bars_per_year)`` and Calmar uses the annualised mean
            return. If ``None``, Sharpe is per-bar and Calmar uses the
            total return.

    Returns:
        Dict with ``sharpe``, ``calmar``, ``max_drawdown`` (a non-negative
        magnitude), and ``total_return``.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import default_backtest_metrics
        >>> m = default_backtest_metrics(np.array([0.01, -0.02, 0.03, 0.0]))
        >>> round(m["total_return"], 4)
        0.02
        >>> round(m["max_drawdown"], 4)
        0.02
    """
    if bars_per_year is not None and bars_per_year <= 0:
        raise ValueError(f"bars_per_year must be positive, got {bars_per_year}.")
    arr = np.asarray(path, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "sharpe": float("nan"),
            "calmar": float("nan"),
            "max_drawdown": float("nan"),
            "total_return": float("nan"),
        }
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    sharpe = mean / std if std > 0 else float("nan")
    if bars_per_year is not None and np.isfinite(sharpe):
        sharpe *= float(np.sqrt(bars_per_year))

    total_return = float(arr.sum())
    # Seed the equity curve with a 0 starting point so the running peak
    # begins at the pre-trade level. Without it a path that opens with a
    # loss measures drawdown from its own first (already negative) point
    # and reports no drawdown at all.
    equity = np.concatenate([[0.0], np.cumsum(arr)])
    peak = np.maximum.accumulate(equity)
    max_drawdown = float(-(equity - peak).min())  # >= 0

    if max_drawdown > 0:
        numerator = mean * bars_per_year if bars_per_year is not None else total_return
        calmar = float(numerator) / max_drawdown
    else:
        calmar = float("nan")

    return {
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "max_drawdown": max_drawdown,
        "total_return": total_return,
    }


def path_metrics(
    paths: NDArrayAny,
    metric_fn: PathMetricFn | None = None,
    *,
    bars_per_year: int | None = None,
) -> pd.DataFrame:
    """Reduce an ``(n_paths, n_samples)`` matrix to per-path metrics.

    Applies ``metric_fn`` to each row and stacks the results into a
    DataFrame with one row per path and one column per metric.

    Args:
        paths: ``(n_paths, n_samples)`` array, as returned by
            :func:`~purgedcv.reconstruct_paths` or
            :meth:`CombinatorialPurgedCV.backtest_paths`. Each row is one
            path's per-period return series.
        metric_fn: Maps a 1-D path to a name -> value mapping. Defaults to
            :func:`default_backtest_metrics`. A custom function lets you
            choose your own statistics and return convention.
        bars_per_year: Forwarded to :func:`default_backtest_metrics` for
            annualisation. Ignored when a custom ``metric_fn`` is given
            (bind any such option into the function yourself).

    Returns:
        DataFrame indexed ``0 .. n_paths - 1`` (index name ``"path"``) with
        one column per metric.

    Raises:
        ValueError: if ``paths`` is not 2-D.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import path_metrics
        >>> paths = np.array([[0.01, -0.02, 0.03, 0.0], [0.0, 0.01, 0.01, 0.01]])
        >>> df = path_metrics(paths)
        >>> list(df.columns)
        ['sharpe', 'calmar', 'max_drawdown', 'total_return']
        >>> df.shape
        (2, 4)
    """
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"paths must be a 2-D (n_paths, n_samples) array, got {arr.ndim}-D.")
    fn: PathMetricFn = (
        partial(default_backtest_metrics, bars_per_year=bars_per_year)
        if metric_fn is None
        else metric_fn
    )
    rows = [dict(fn(arr[i])) for i in range(arr.shape[0])]
    index = pd.RangeIndex(arr.shape[0], name="path")
    return pd.DataFrame(rows, index=index)
