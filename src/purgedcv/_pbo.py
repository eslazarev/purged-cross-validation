"""Internal: Probability of Backtest Overfitting (PBO) via CSCV.

PBO answers a single question: when you pick the configuration that looks
best in-sample, how often does it land below the median out-of-sample? A
high PBO means your selection procedure is mostly fitting noise.

The estimator is Combinatorially Symmetric Cross-Validation (CSCV): the
time axis is cut into ``n_splits`` contiguous blocks, every way of choosing
half the blocks as the in-sample (IS) set is enumerated, and the
complementary half is the out-of-sample (OOS) set. CSCV is exactly
:class:`~purgedcv.CombinatorialPurgedCV` with
``n_test_groups = n_splits // 2``, so when prediction/evaluation times are
supplied the same purge and embargo machinery cleans every IS/OOS boundary.

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 11, and Bailey, Borwein, Lopez de Prado & Salehipour (2017), "The
Probability of Backtest Overfitting", Journal of Computational Finance.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from purgedcv._cpcv import CombinatorialPurgedCV
from purgedcv._time import HorizonLike
from purgedcv._validation import _validate_integer

from ._typing import NDArrayAny

#: A configuration-performance metric: maps a 1-D return slice to a scalar
#: where larger is better. The default is :func:`sharpe`.
PerformanceMetric = Callable[[NDArrayAny], float]


def sharpe(returns: NDArrayAny) -> float:
    """Plain Sharpe ratio (mean / sample standard deviation).

    The default selection metric for :func:`probability_of_backtest_overfitting`.
    A degenerate slice (zero or non-finite standard deviation) scores 0.0 so
    it never wins the in-sample selection.

    Examples:
        >>> import numpy as np
        >>> from purgedcv._pbo import sharpe
        >>> round(sharpe(np.array([0.01, 0.02, -0.01, 0.03])), 4)
        0.7319
    """
    arr = np.asarray(returns, dtype=float)
    # A constant slice has an undefined Sharpe. Test the value range rather
    # than the computed std: for bit-identical values the mean carries a
    # rounding error that leaves std at ~1e-18 instead of exactly 0, which
    # would otherwise explode the ratio.
    if arr.size < 2 or float(np.ptp(arr)) == 0.0:
        return 0.0
    std = float(arr.std(ddof=1))
    if std == 0.0 or not np.isfinite(std):
        return 0.0
    return float(arr.mean() / std)


@dataclass(frozen=True, eq=False)
class PBOResult:
    """Return type of :func:`probability_of_backtest_overfitting`.

    A frozen dataclass: read fields by attribute (``result.pbo``), and call
    :func:`dataclasses.asdict` if you need a plain dict to serialise.

    Attributes:
        pbo: Probability of backtest overfitting in [0, 1]. The fraction of
            CSCV combinations whose in-sample-best configuration ranked
            below the OOS median (equivalently, logit < 0). Read it as
            ``0.0`` = selection never overfits, ``0.5`` = selection is no
            better than chance, ``1.0`` = the in-sample best is always an
            out-of-sample loser. Anything materially above 0.5 means the
            search is mostly fitting noise.
        logits: Per-combination logit of the OOS relative rank of the
            in-sample-best configuration. Negative values are the overfit
            cases; the distribution's mass below 0 is ``pbo``.
        slope: OLS slope of OOS performance on IS performance across the
            combinations (for the IS-best configuration). Direction is the
            signal: ``> 0`` means in-sample strength carries over (no
            overfit), ``~ 0`` means the link is random, and ``< 0`` means
            in-sample strength predicts out-of-sample weakness (severe
            overfit). A positive slope below 1 still indicates some decay.
        is_oos_performance: ``(n_combos, 2)`` array; column 0 is the IS
            performance of the IS-best configuration, column 1 its OOS
            performance, one row per combination.
        n_combos: Number of CSCV combinations scored, ``C(n_splits,
            n_splits // 2)`` minus any folds dropped for an empty IS set.
    """

    pbo: float
    logits: NDArrayAny
    slope: float
    is_oos_performance: NDArrayAny
    n_combos: int


def _contiguous_blocks(n_obs: int, n_splits: int) -> list[NDArrayAny]:
    """Partition ``range(n_obs)`` into ``n_splits`` contiguous blocks.

    Matches :meth:`CombinatorialPurgedCV._iter_test_indices` exactly so the
    no-purge fast path and the purged splitter path agree on block layout.
    """
    block_size, remainder = divmod(n_obs, n_splits)
    cursor = 0
    blocks: list[NDArrayAny] = []
    for k in range(n_splits):
        size = block_size + (1 if k < remainder else 0)
        blocks.append(np.arange(cursor, cursor + size, dtype=np.int64))
        cursor += size
    return blocks


def _iter_is_oos(
    n_obs: int,
    n_splits: int,
    prediction_times: pd.Series | None,
    evaluation_times: pd.Series | None,
    purge_horizon: HorizonLike | None,
    embargo: HorizonLike | None,
) -> Iterator[tuple[NDArrayAny, NDArrayAny]]:
    """Yield ``(is_idx, oos_idx)`` for every CSCV combination.

    Without times (and without purge/embargo) the in-sample set is the plain
    complement of the chosen OOS blocks. With times, the work is delegated to
    :class:`CombinatorialPurgedCV` so purge and embargo trim each IS set at
    the IS/OOS boundaries.
    """
    n_test_groups = n_splits // 2
    wants_purge = purge_horizon is not None or embargo is not None
    if prediction_times is None and evaluation_times is None and not wants_purge:
        blocks = _contiguous_blocks(n_obs, n_splits)
        for oos_combo in combinations(range(n_splits), n_test_groups):
            oos_mask = np.zeros(n_obs, dtype=bool)
            for g in oos_combo:
                oos_mask[blocks[g]] = True
            oos_idx = np.flatnonzero(oos_mask)
            is_idx = np.flatnonzero(~oos_mask)
            yield is_idx, oos_idx
        return

    if prediction_times is None or evaluation_times is None:
        raise ValueError(
            "prediction_times and evaluation_times must be supplied together "
            "(or both omitted); purge_horizon and embargo require both."
        )
    cv = CombinatorialPurgedCV(
        n_splits=n_splits,
        n_test_groups=n_test_groups,
        prediction_times=prediction_times,
        evaluation_times=evaluation_times,
        purge_horizon=purge_horizon,
        embargo=embargo,
    )
    placeholder = np.empty((n_obs, 1), dtype=float)
    yield from cv.split(placeholder)


def probability_of_backtest_overfitting(
    returns: NDArrayAny,
    n_splits: int = 16,
    *,
    metric: PerformanceMetric = sharpe,
    prediction_times: pd.Series | None = None,
    evaluation_times: pd.Series | None = None,
    purge_horizon: HorizonLike | None = None,
    embargo: HorizonLike | None = None,
) -> PBOResult:
    """Estimate the Probability of Backtest Overfitting (PBO) via CSCV.

    Given the per-period returns of ``n_configs`` candidate configurations,
    PBO is the probability that the configuration selected as best
    in-sample performs below the median out-of-sample. The estimate is the
    relative frequency of that event across all CSCV combinations.

    Args:
        returns: ``(n_configs, n_obs)`` array; ``returns[j]`` is the per-
            period return series of configuration ``j`` over a shared time
            axis. At least 2 configurations and ``n_obs >= n_splits`` are
            required; all values must be finite.
        n_splits: Number of contiguous blocks the time axis is cut into
            (``S`` in CSCV). Must be even and >= 2; the number of
            combinations is ``C(n_splits, n_splits // 2)``. The standard
            choice is 16.
        metric: Performance metric mapping a 1-D return slice to a scalar
            (larger is better). Defaults to :func:`sharpe`.
        prediction_times: Optional per-period prediction times. Supply with
            ``evaluation_times`` to enable purge/embargo at IS/OOS
            boundaries. Length must equal ``n_obs``.
        evaluation_times: Optional per-period evaluation times; see
            ``prediction_times``.
        purge_horizon: Optional purge horizon (requires the time series).
        embargo: Optional embargo horizon (requires the time series).

    Returns:
        A :class:`PBOResult` (frozen dataclass; read fields by attribute,
        e.g. ``result.pbo``, ``result.slope``).

    Raises:
        ValueError: on a malformed ``returns`` matrix, an odd or too-small
            ``n_splits``, a time series whose length disagrees with
            ``n_obs``, or purge/embargo requested without times.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import probability_of_backtest_overfitting
        >>> rng = np.random.default_rng(0)
        >>> returns = rng.standard_normal((10, 240))  # 10 configs, pure noise
        >>> result = probability_of_backtest_overfitting(returns, n_splits=8)
        >>> 0.0 <= result.pbo <= 1.0
        True
        >>> result.n_combos
        70
    """
    n_splits = _validate_integer("n_splits", n_splits, minimum=2)
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even for CSCV, got {n_splits}.")

    matrix = np.asarray(returns, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"returns must be a 2-D (n_configs, n_obs) array, got {matrix.ndim}-D.")
    n_configs, n_obs = matrix.shape
    if n_configs < 2:
        raise ValueError(f"returns needs at least 2 configurations, got {n_configs}.")
    if n_obs < n_splits:
        raise ValueError(
            f"returns has n_obs={n_obs} columns, fewer than n_splits={n_splits}; "
            "CSCV requires non-empty blocks."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("returns contains NaN or infinite values.")
    if prediction_times is not None and len(prediction_times) != n_obs:
        raise ValueError(
            f"prediction_times length {len(prediction_times)} does not match " f"n_obs={n_obs}."
        )
    if evaluation_times is not None and len(evaluation_times) != n_obs:
        raise ValueError(
            f"evaluation_times length {len(evaluation_times)} does not match " f"n_obs={n_obs}."
        )

    logits: list[float] = []
    is_perf_best: list[float] = []
    oos_perf_best: list[float] = []
    n_dropped = 0
    for is_idx, oos_idx in _iter_is_oos(
        n_obs, n_splits, prediction_times, evaluation_times, purge_horizon, embargo
    ):
        if len(is_idx) == 0 or len(oos_idx) == 0:
            n_dropped += 1
            continue
        try:
            is_perf = np.asarray([metric(matrix[j, is_idx]) for j in range(n_configs)], dtype=float)
            oos_perf = np.asarray(
                [metric(matrix[j, oos_idx]) for j in range(n_configs)], dtype=float
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "metric must return a single finite number per configuration; "
                "it returned a non-numeric or ragged value."
            ) from exc
        if is_perf.ndim != 1 or oos_perf.ndim != 1:
            raise ValueError(
                "metric must return a scalar per configuration, got a "
                f"{max(is_perf.ndim, oos_perf.ndim)}-D result; do not return a "
                "vector (e.g. a slice of the input)."
            )
        if not np.isfinite(is_perf).all() or not np.isfinite(oos_perf).all():
            raise ValueError(
                "metric returned a non-finite value; PBO requires a finite score "
                "for every configuration on every in-sample and out-of-sample "
                "split. Make the metric return a finite fallback for degenerate "
                "slices (the default sharpe scores a constant slice 0.0)."
            )
        best = int(np.argmax(is_perf))
        # Relative OOS rank of the IS-best, in (0, 1): average ranks handle
        # ties, and dividing by n_configs + 1 keeps the logit finite even
        # when the IS-best is strictly worst or best out of sample.
        ranks = stats.rankdata(oos_perf)
        omega = float(ranks[best] / (n_configs + 1))
        logits.append(float(np.log(omega / (1 - omega))))
        is_perf_best.append(float(is_perf[best]))
        oos_perf_best.append(float(oos_perf[best]))

    if n_dropped:
        warnings.warn(
            f"{n_dropped} CSCV combination(s) had an empty in-sample set after "
            "purge/embargo and were dropped from the PBO estimate.",
            stacklevel=2,
        )
    if not logits:
        raise ValueError(
            "every CSCV combination had an empty in-sample set; relax "
            "purge_horizon/embargo or reduce n_splits."
        )

    logit_arr = np.asarray(logits, dtype=float)
    is_arr = np.asarray(is_perf_best, dtype=float)
    oos_arr = np.asarray(oos_perf_best, dtype=float)
    pbo = float(np.mean(logit_arr < 0))
    var_is = float(np.var(is_arr))
    if var_is > 0 and len(is_arr) > 1:
        slope = float(np.cov(is_arr, oos_arr, ddof=0)[0, 1] / var_is)
    else:
        slope = float("nan")

    return PBOResult(
        pbo=pbo,
        logits=logit_arr,
        slope=slope,
        is_oos_performance=np.column_stack([is_arr, oos_arr]),
        n_combos=len(logits),
    )
