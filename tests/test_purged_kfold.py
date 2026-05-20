"""Unit tests for PurgedKFold and PurgedGroupKFold (D5.2 + D5.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purgedcv._purged_kfold import PurgedGroupKFold, PurgedKFold
from purgedcv.diagnostics import assert_no_temporal_leakage


def _times(n: int = 20, horizon_days: int = 1) -> tuple[pd.Series, pd.Series]:
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return pred, evalu


class TestPurgedKFold:
    def test_yields_n_splits_folds(self) -> None:
        pred, evalu = _times(n=20)
        cv = PurgedKFold(
            n_splits=5,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        folds = list(cv.split(X))
        assert len(folds) == 5
        assert cv.get_n_splits() == 5

    def test_test_folds_partition_the_indices(self) -> None:
        """The union of all test folds must equal {0..n-1} exactly."""
        pred, evalu = _times(n=20)
        cv = PurgedKFold(
            n_splits=5,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        all_test_idx: list[int] = []
        for _, test_idx in cv.split(X):
            all_test_idx.extend(test_idx.tolist())
        assert sorted(all_test_idx) == list(range(20))

    def test_each_test_fold_is_contiguous(self) -> None:
        pred, evalu = _times(n=20)
        cv = PurgedKFold(
            n_splits=5,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        for _, test_idx in cv.split(X):
            assert np.all(np.diff(test_idx) == 1)

    def test_purge_horizon_drops_adjacent_train_rows(self) -> None:
        """With horizon=2D and purge_horizon=2D, train rows adjacent to each
        test fold must be dropped."""
        pred, evalu = _times(n=20, horizon_days=2)
        cv = PurgedKFold(
            n_splits=5,
            purge_horizon="2D",
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        for train_idx, test_idx in cv.split(X):
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="2D")

    def test_fold_layout_matches_hand_computed(self) -> None:
        """20 rows, n_splits=5, purge=embargo=0: test fold k = [4k, 4k+4)."""
        pred, evalu = _times(n=20, horizon_days=1)
        cv = PurgedKFold(
            n_splits=5,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        for k, (_, test_idx) in enumerate(cv.split(X)):
            expected = np.arange(4 * k, 4 * (k + 1))
            np.testing.assert_array_equal(test_idx, expected)

    def test_rejects_n_splits_below_two(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="at least 2"):
            PurgedKFold(
                n_splits=1,
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_zero_purge_retains_full_post_test_complement(self) -> None:
        """With zero horizons, fold 0's train must equal the full complement
        of test. This guards against future base-class regressions that
        would over-purge under the no-op zero-horizon case."""
        pred, evalu = _times(n=20, horizon_days=1)
        cv = PurgedKFold(
            n_splits=5,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        folds = list(cv.split(X))
        train0, test0 = folds[0]
        np.testing.assert_array_equal(test0, np.arange(0, 4))
        np.testing.assert_array_equal(train0, np.arange(4, 20))

    def test_more_splits_than_samples_yields_empty_folds(self) -> None:
        """When n_splits > n_samples, _iter_test_indices yields fold_size=0
        folds for the excess. The base class purge/embargo gracefully handle
        empty test arrays. Documenting this contract explicitly so a future
        refactor doesn't silently change the behavior."""
        pred, evalu = _times(n=5)
        cv = PurgedKFold(
            n_splits=10,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((5, 1))  # noqa: N806
        folds = list(cv.split(X))
        assert len(folds) == 10
        # Folds 0..4 get one row each (n_samples=5 distributed across n_splits=10
        # via floor(5/10)=0 plus 1-extra for the first remainder=5 folds).
        # Folds 5..9 are empty.
        fold_sizes = [len(test_idx) for _, test_idx in folds]
        assert fold_sizes[:5] == [1, 1, 1, 1, 1]
        assert fold_sizes[5:] == [0, 0, 0, 0, 0]


class TestPurgedGroupKFold:
    def test_yields_n_splits_folds(self) -> None:
        """6 patients, 5 observations each (30 rows), n_splits=3."""
        pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        groups = pd.Series(np.repeat([0, 1, 2, 3, 4, 5], 5))
        cv = PurgedGroupKFold(
            n_splits=3,
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        X = np.zeros((30, 1))  # noqa: N806
        folds = list(cv.split(X))
        assert len(folds) == 3

    def test_each_group_appears_in_exactly_one_test_fold(self) -> None:
        pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        groups = pd.Series(np.repeat([0, 1, 2, 3, 4, 5], 5))
        cv = PurgedGroupKFold(
            n_splits=3,
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        X = np.zeros((30, 1))  # noqa: N806
        seen_groups: dict[int, int] = {}
        for k, (_, test_idx) in enumerate(cv.split(X)):
            test_groups = set(groups.iloc[test_idx].tolist())
            for g in test_groups:
                assert (
                    g not in seen_groups
                ), f"group {g} appeared in fold {seen_groups[g]} AND fold {k}"
                seen_groups[g] = k
        assert set(seen_groups.keys()) == {0, 1, 2, 3, 4, 5}

    def test_no_group_leakage_within_fold(self) -> None:
        pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        groups = pd.Series(np.repeat([0, 1, 2, 3, 4, 5], 5))
        cv = PurgedGroupKFold(
            n_splits=3,
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        X = np.zeros((30, 1))  # noqa: N806
        for train_idx, test_idx in cv.split(X):
            train_groups = set(groups.iloc[train_idx].tolist())
            test_groups = set(groups.iloc[test_idx].tolist())
            assert train_groups & test_groups == set()

    def test_purge_horizon_applies_across_groups(self) -> None:
        """Even with group-disjointness, purge_horizon must drop rows from
        OTHER groups whose horizons overlap the test window."""
        pred = pd.Series(pd.date_range("2024-01-01", periods=12, freq="D"))
        evalu = pred + pd.Timedelta(days=3)
        groups = pd.Series(np.repeat([0, 1, 2, 3], 3))
        cv = PurgedGroupKFold(
            n_splits=4,
            purge_horizon="3D",
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        X = np.zeros((12, 1))  # noqa: N806
        for train_idx, test_idx in cv.split(X):
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="3D")

    def test_rejects_n_splits_exceeding_unique_groups(self) -> None:
        pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        groups = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # 5 unique groups
        with pytest.raises(ValueError, match=r"exceeds.*unique groups"):
            PurgedGroupKFold(
                n_splits=10,
                prediction_times=pred,
                evaluation_times=evalu,
                groups=groups,
            )

    def test_rejects_n_splits_below_two(self) -> None:
        pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        groups = pd.Series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        with pytest.raises(ValueError, match="at least 2"):
            PurgedGroupKFold(
                n_splits=1,
                prediction_times=pred,
                evaluation_times=evalu,
                groups=groups,
            )
