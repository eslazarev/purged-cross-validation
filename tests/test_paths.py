"""Unit tests for reconstruct_paths (Domain D6)."""

from __future__ import annotations

import numpy as np
import pytest

from purgedcv._paths import reconstruct_paths
from purgedcv._typing import NDArrayAny


def _cpcv_fold_test_indices(n_samples: int, n_splits: int, n_test_groups: int) -> list[NDArrayAny]:
    """Helper: reproduce CombinatorialPurgedCV._iter_test_indices independently
    so reconstruct_paths tests don't depend on the splitter."""
    from itertools import combinations

    group_size, remainder = divmod(n_samples, n_splits)
    cursor = 0
    group_indices: list[NDArrayAny] = []
    for k in range(n_splits):
        sz = group_size + (1 if k < remainder else 0)
        group_indices.append(np.arange(cursor, cursor + sz, dtype=np.int64))
        cursor += sz
    return [
        np.concatenate([group_indices[i] for i in combo])
        for combo in combinations(range(n_splits), n_test_groups)
    ]


class TestReconstructPaths:
    def test_n_paths_equals_n_minus_1_choose_k_minus_1(self) -> None:
        """For N=6, K=2: n_paths = C(5, 1) = 5."""
        n_splits, n_test_groups, n_samples = 6, 2, 24
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        # Fake predictions: each fold predicts its own indices as the index itself
        # (so we can verify path assembly by reading the path values).
        fold_predictions = [test_idx.astype(float) for test_idx in fold_test_indices]
        paths = reconstruct_paths(
            fold_predictions, fold_test_indices, n_splits, n_test_groups, n_samples
        )
        assert paths.shape == (5, 24)

    def test_n_paths_n4_k2_equals_3(self) -> None:
        n_splits, n_test_groups, n_samples = 4, 2, 16
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        fold_predictions = [test_idx.astype(float) for test_idx in fold_test_indices]
        paths = reconstruct_paths(
            fold_predictions, fold_test_indices, n_splits, n_test_groups, n_samples
        )
        assert paths.shape == (3, 16)

    def test_each_sample_appears_exactly_once_per_path(self) -> None:
        """Coverage invariant: every sample is predicted in every path
        (no path has missing samples, no path has duplicates)."""
        n_splits, n_test_groups, n_samples = 6, 2, 24
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        # Use predictions = index so we can spot missing entries (would be 0 or NaN).
        fold_predictions = [test_idx.astype(float) for test_idx in fold_test_indices]
        paths = reconstruct_paths(
            fold_predictions, fold_test_indices, n_splits, n_test_groups, n_samples
        )
        # Every position in every path should be a finite float matching its
        # column index (since predictions = test_idx as float).
        for p in range(paths.shape[0]):
            expected = np.arange(n_samples, dtype=float)
            np.testing.assert_array_equal(paths[p], expected)

    def test_n4_k2_hand_enumerated_path_assignment(self) -> None:
        """For N=4, K=2 with predictions = fold_index_constant, verify which
        fold's predictions end up in which path. This pins the greedy
        positional assignment rule."""
        n_splits, n_test_groups, n_samples = 4, 2, 16
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        # Six folds in itertools.combinations order:
        # 0: (0,1)  1: (0,2)  2: (0,3)  3: (1,2)  4: (1,3)  5: (2,3)
        # Each fold's predictions are a constant equal to the fold index, so
        # we can read path assembly directly from the output values.
        fold_predictions = [
            np.full(len(test_idx), fold_idx, dtype=float)
            for fold_idx, test_idx in enumerate(fold_test_indices)
        ]
        paths = reconstruct_paths(
            fold_predictions, fold_test_indices, n_splits, n_test_groups, n_samples
        )
        assert paths.shape == (3, 16)
        # Group indices (block size 4 each): g0=[0..4), g1=[4..8), g2=[8..12), g3=[12..16).
        # Each group appears in C(3,1) = 3 folds (in fold order):
        #   g0 in folds [0, 1, 2]  -> contributes to paths 0, 1, 2 respectively
        #   g1 in folds [0, 3, 4]
        #   g2 in folds [1, 3, 5]
        #   g3 in folds [2, 4, 5]
        # Therefore path 0 reads:
        #   indices 0..4   from fold 0  (constant 0.0)
        #   indices 4..8   from fold 0  (constant 0.0)
        #   indices 8..12  from fold 1  (constant 1.0)
        #   indices 12..16 from fold 2  (constant 2.0)
        np.testing.assert_array_equal(
            paths[0], np.array([0] * 4 + [0] * 4 + [1] * 4 + [2] * 4, dtype=float)
        )
        # Path 1:
        #   g0 -> fold 1, g1 -> fold 3, g2 -> fold 3, g3 -> fold 4
        np.testing.assert_array_equal(
            paths[1], np.array([1] * 4 + [3] * 4 + [3] * 4 + [4] * 4, dtype=float)
        )
        # Path 2:
        #   g0 -> fold 2, g1 -> fold 4, g2 -> fold 5, g3 -> fold 5
        np.testing.assert_array_equal(
            paths[2], np.array([2] * 4 + [4] * 4 + [5] * 4 + [5] * 4, dtype=float)
        )

    def test_nan_in_fold_predictions_propagates(self) -> None:
        """If a fold's predictions are NaN (e.g. estimator couldn't fit
        because purge collapsed the training set), the NaN propagates
        only to paths that use that fold."""
        n_splits, n_test_groups, n_samples = 4, 2, 16
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        # Fold 2 (which is combo (0, 3)) yields NaN.
        fold_predictions = [
            np.full(len(test_idx), float(fold_idx), dtype=float)
            for fold_idx, test_idx in enumerate(fold_test_indices)
        ]
        fold_predictions[2] = np.full(len(fold_test_indices[2]), np.nan)
        paths = reconstruct_paths(
            fold_predictions, fold_test_indices, n_splits, n_test_groups, n_samples
        )
        # Path 0 uses fold 2 for g0 and g3 (per the assignment above) — so
        # path 0 has NaN at indices 0..4 and 12..16.
        # Wait — re-check: g0 in folds [0,1,2] -> path 2 uses fold 2 for g0,
        # not path 0. Let me re-verify by computing.
        # Actually the assignment from the previous test:
        #   path 0 uses fold 0 for g0, fold 0 for g1, fold 1 for g2, fold 2 for g3.
        #   path 1 uses fold 1 for g0, fold 3 for g1, fold 3 for g2, fold 4 for g3.
        #   path 2 uses fold 2 for g0, fold 4 for g1, fold 5 for g2, fold 5 for g3.
        # So fold 2 NaN affects:
        #   - path 0 at indices 12..16 (g3)
        #   - path 2 at indices 0..4   (g0)
        # Other path-2 positions remain finite.
        assert np.all(np.isnan(paths[0, 12:16]))
        assert np.all(np.isfinite(paths[0, 0:12]))  # rest of path 0 is finite
        assert np.all(np.isnan(paths[2, 0:4]))
        assert np.all(np.isfinite(paths[2, 4:16]))  # rest of path 2 is finite
        # Path 1 doesn't use fold 2 at all, so it's fully finite.
        assert np.all(np.isfinite(paths[1]))

    def test_rejects_mismatched_fold_count(self) -> None:
        """If len(fold_predictions) != C(N, K), the function refuses."""
        n_splits, n_test_groups, n_samples = 4, 2, 16
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        truncated = [test_idx.astype(float) for test_idx in fold_test_indices[:-1]]
        with pytest.raises(ValueError, match="fold count"):
            reconstruct_paths(truncated, fold_test_indices, n_splits, n_test_groups, n_samples)

    def test_rejects_fold_prediction_length_mismatch(self) -> None:
        """If a fold's prediction array length doesn't match its test_idx
        length, the function refuses."""
        n_splits, n_test_groups, n_samples = 4, 2, 16
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        bad_predictions = [test_idx.astype(float) for test_idx in fold_test_indices]
        bad_predictions[0] = bad_predictions[0][:-1]  # off by one
        with pytest.raises(ValueError, match="length"):
            reconstruct_paths(
                bad_predictions, fold_test_indices, n_splits, n_test_groups, n_samples
            )

    def test_rejects_invalid_cpcv_parameters(self) -> None:
        with pytest.raises(ValueError, match="n_splits"):
            reconstruct_paths([], [], n_splits=1, n_test_groups=1, n_samples=1)
        with pytest.raises(ValueError, match="n_test_groups"):
            reconstruct_paths([], [], n_splits=4, n_test_groups=4, n_samples=16)
        with pytest.raises(ValueError, match="n_samples"):
            reconstruct_paths([], [], n_splits=4, n_test_groups=1, n_samples=3)

    def test_rejects_non_integer_cpcv_parameters(self) -> None:
        with pytest.raises(TypeError, match="n_splits"):
            reconstruct_paths([], [], n_splits=4.5, n_test_groups=1, n_samples=16)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="n_test_groups"):
            reconstruct_paths([], [], n_splits=4, n_test_groups=1.5, n_samples=16)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="n_samples"):
            reconstruct_paths([], [], n_splits=4, n_test_groups=1, n_samples=16.5)  # type: ignore[arg-type]

    def test_rejects_noncanonical_test_indices(self) -> None:
        n_splits, n_test_groups, n_samples = 4, 2, 16
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        fold_predictions = [test_idx.astype(float) for test_idx in fold_test_indices]
        bad_test_indices = [test_idx.copy() for test_idx in fold_test_indices]
        bad_test_indices[0] = bad_test_indices[0][::-1]

        with pytest.raises(ValueError, match="canonical CPCV"):
            reconstruct_paths(
                fold_predictions,
                bad_test_indices,
                n_splits,
                n_test_groups,
                n_samples,
            )

    def test_rejects_non_integer_test_indices(self) -> None:
        n_splits, n_test_groups, n_samples = 4, 2, 16
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        fold_predictions = [test_idx.astype(float) for test_idx in fold_test_indices]
        bad_test_indices = [test_idx.copy() for test_idx in fold_test_indices]
        bad_test_indices[0] = bad_test_indices[0].astype(float)

        with pytest.raises(TypeError, match="integer"):
            reconstruct_paths(
                fold_predictions,
                bad_test_indices,
                n_splits,
                n_test_groups,
                n_samples,
            )

    def test_rejects_duplicate_test_indices(self) -> None:
        n_splits, n_test_groups, n_samples = 4, 2, 16
        fold_test_indices = _cpcv_fold_test_indices(n_samples, n_splits, n_test_groups)
        fold_predictions = [test_idx.astype(float) for test_idx in fold_test_indices]
        bad_test_indices = [test_idx.copy() for test_idx in fold_test_indices]
        bad_test_indices[0] = bad_test_indices[0].copy()
        bad_test_indices[0][1] = bad_test_indices[0][0]

        with pytest.raises(ValueError, match="duplicate"):
            reconstruct_paths(
                fold_predictions,
                bad_test_indices,
                n_splits,
                n_test_groups,
                n_samples,
            )
