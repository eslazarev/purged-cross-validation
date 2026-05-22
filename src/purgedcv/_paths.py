"""Internal: backtest path reconstruction for CombinatorialPurgedCV (Domain D6).

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 12 section 12.4, figure 12.2.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import comb

import numpy as np

from ._typing import NDArrayAny
from ._validation import _validate_integer


def reconstruct_paths(
    fold_predictions: Sequence[NDArrayAny],
    fold_test_indices: Sequence[NDArrayAny],
    n_splits: int,
    n_test_groups: int,
    n_samples: int,
) -> NDArrayAny:
    """Assemble Combinatorial Purged CV fold predictions into backtest paths.

    Given the predictions and test indices for all C(N, K) folds produced by
    :class:`CombinatorialPurgedCV`, returns a ``(n_paths, n_samples)`` array
    where each row is a complete time-ordered out-of-sample prediction
    sequence built from a different combination of the folds.

    ``n_paths = C(n_splits - 1, n_test_groups - 1)``. Each sample is
    predicted in every path (no missing entries). The assignment of fold
    to path for each group is the canonical greedy positional rule: for
    group ``g``, the k-th fold (in fold-iteration order) that contains
    ``g`` as a test group contributes ``g``'s predictions to path k.

    NaN predictions in any fold propagate only to the paths that use
    that fold for the affected group(s).

    Args:
        fold_predictions: One array per fold; ``fold_predictions[f]`` is the
            prediction values for the rows in ``fold_test_indices[f]``,
            in the same order.
        fold_test_indices: One array per fold; the per-fold test_idx as
            yielded by :meth:`CombinatorialPurgedCV._iter_test_indices`.
            Used here to recover the group block layout.
        n_splits: Number of CPCV group blocks.
        n_test_groups: Number of test groups per fold.
        n_samples: Total number of samples in the dataset.

    Returns:
        ``(n_paths, n_samples)`` array of predictions.

    Raises:
        ValueError: on fold count or fold-prediction length mismatch.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import reconstruct_paths
        >>> # Synthetic 4-fold output: 4 samples per fold's test set,
        >>> # predictions equal to the fold index.
        >>> fold_test = [
        ...     np.array([0, 1, 2, 3, 4, 5, 6, 7]),     # combo (0, 1)
        ...     np.array([0, 1, 2, 3, 8, 9, 10, 11]),   # combo (0, 2)
        ...     np.array([0, 1, 2, 3, 12, 13, 14, 15]), # combo (0, 3)
        ...     np.array([4, 5, 6, 7, 8, 9, 10, 11]),   # combo (1, 2)
        ...     np.array([4, 5, 6, 7, 12, 13, 14, 15]), # combo (1, 3)
        ...     np.array([8, 9, 10, 11, 12, 13, 14, 15]), # combo (2, 3)
        ... ]
        >>> fold_preds = [
        ...     np.full(len(ti), float(f)) for f, ti in enumerate(fold_test)
        ... ]
        >>> paths = reconstruct_paths(fold_preds, fold_test, 4, 2, 16)
        >>> paths.shape
        (3, 16)
    """
    n_splits = _validate_integer("n_splits", n_splits, minimum=2)
    n_test_groups = _validate_integer("n_test_groups", n_test_groups, minimum=1)
    n_samples = _validate_integer("n_samples", n_samples, minimum=1)
    if n_test_groups >= n_splits:
        raise ValueError(
            f"n_test_groups must be in [1, n_splits-1] = [1, {n_splits - 1}], "
            f"got {n_test_groups}."
        )
    if n_samples < n_splits:
        raise ValueError(
            f"n_samples={n_samples} is smaller than n_splits={n_splits}; "
            "CPCV path reconstruction requires non-empty group blocks."
        )

    expected_folds = comb(n_splits, n_test_groups)
    if len(fold_predictions) != expected_folds:
        raise ValueError(
            f"fold count mismatch: expected C({n_splits}, {n_test_groups}) = "
            f"{expected_folds} folds, got {len(fold_predictions)}."
        )
    if len(fold_test_indices) != expected_folds:
        raise ValueError(
            f"fold count mismatch: expected {expected_folds} test-index arrays, "
            f"got {len(fold_test_indices)}."
        )
    # Reconstruct the group blocks from n_samples + n_splits.
    group_size, remainder = divmod(n_samples, n_splits)
    cursor = 0
    group_indices: list[NDArrayAny] = []
    for k in range(n_splits):
        sz = group_size + (1 if k < remainder else 0)
        group_indices.append(np.arange(cursor, cursor + sz, dtype=np.int64))
        cursor += sz

    fold_combos = list(combinations(range(n_splits), n_test_groups))
    for f, (combo, preds, test_idx) in enumerate(
        zip(fold_combos, fold_predictions, fold_test_indices, strict=True)
    ):
        if len(preds) != len(test_idx):
            raise ValueError(
                f"fold {f}: predictions length {len(preds)} does not match "
                f"test_idx length {len(test_idx)}."
            )
        expected_test_idx = np.concatenate([group_indices[i] for i in combo])
        if not np.array_equal(np.asarray(test_idx), expected_test_idx):
            raise ValueError(f"fold {f}: test_idx does not match the canonical CPCV group layout.")

    # Map each group to the list of (fold_idx, prediction_slice) pairs in
    # fold-iteration order, where prediction_slice is the portion of that
    # fold's predictions corresponding to that group's rows.
    group_to_fold_slices: dict[int, list[tuple[int, NDArrayAny]]] = {g: [] for g in range(n_splits)}
    for fold_idx, (combo, preds) in enumerate(zip(fold_combos, fold_predictions, strict=True)):
        cursor_in_fold = 0
        for g in combo:
            g_size = len(group_indices[g])
            group_to_fold_slices[g].append(
                (fold_idx, preds[cursor_in_fold : cursor_in_fold + g_size])
            )
            cursor_in_fold += g_size

    # n_paths = C(N-1, K-1). Greedy assignment: g's k-th fold occurrence -> path k.
    n_paths = comb(n_splits - 1, n_test_groups - 1)
    paths = np.full((n_paths, n_samples), np.nan, dtype=float)
    for g in range(n_splits):
        for path_idx, (_, slice_preds) in enumerate(group_to_fold_slices[g]):
            paths[path_idx, group_indices[g]] = slice_preds
    return paths
