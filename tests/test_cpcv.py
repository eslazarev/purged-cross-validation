"""Unit tests for CombinatorialPurgedCV (D5.4)."""

from __future__ import annotations

import warnings
from math import comb

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import FitFailedWarning

from purgedcv._cpcv import CombinatorialPurgedCV
from purgedcv.diagnostics import (
    assert_embargo_respected,
    assert_no_temporal_leakage,
)


def _times(n: int = 24, horizon_days: int = 1) -> tuple[pd.Series, pd.Series]:
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return pred, evalu


class TestCombinatorialPurgedCV:
    def test_yields_n_choose_k_folds(self) -> None:
        """For N=6 groups and K=2 test groups per fold, expect C(6,2)=15."""
        pred, evalu = _times(n=24)
        cv = CombinatorialPurgedCV(
            n_splits=6,
            n_test_groups=2,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((24, 1))  # noqa: N806
        folds = list(cv.split(X))
        assert len(folds) == comb(6, 2) == 15
        assert cv.get_n_splits() == 15

    def test_hand_enumerated_n4_k2(self) -> None:
        """For N=4 groups and K=2, the 6 test-group combinations must be
        exactly {(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)}."""
        pred, evalu = _times(n=16)
        cv = CombinatorialPurgedCV(
            n_splits=4,
            n_test_groups=2,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((16, 1))  # noqa: N806
        # Each group is a contiguous block of 4 samples:
        # group 0 = [0..4), group 1 = [4..8), group 2 = [8..12), group 3 = [12..16).
        expected_combos = {
            frozenset({0, 1, 2, 3, 4, 5, 6, 7}),  # (0, 1)
            frozenset({0, 1, 2, 3, 8, 9, 10, 11}),  # (0, 2)
            frozenset({0, 1, 2, 3, 12, 13, 14, 15}),  # (0, 3)
            frozenset({4, 5, 6, 7, 8, 9, 10, 11}),  # (1, 2)
            frozenset({4, 5, 6, 7, 12, 13, 14, 15}),  # (1, 3)
            frozenset({8, 9, 10, 11, 12, 13, 14, 15}),  # (2, 3)
        }
        seen = set()
        for _, test_idx in cv.split(X):
            seen.add(frozenset(test_idx.tolist()))
        assert seen == expected_combos

    def test_each_group_appears_in_n_minus_1_choose_k_minus_1_folds(self) -> None:
        """Combinatorial identity: every group is in C(N-1, K-1) of the
        C(N, K) folds."""
        pred, evalu = _times(n=24)
        n, k = 6, 2
        cv = CombinatorialPurgedCV(
            n_splits=n,
            n_test_groups=k,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((24, 1))  # noqa: N806
        counts = [0] * n
        for _, test_idx in cv.split(X):
            group_size = 24 // n
            test_groups = set((test_idx // group_size).tolist())
            for g in test_groups:
                counts[g] += 1
        expected = comb(n - 1, k - 1)
        assert all(c == expected for c in counts)

    def test_rejects_k_greater_than_or_equal_to_n(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(
                n_splits=4,
                n_test_groups=4,
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_rejects_k_below_one(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(
                n_splits=4,
                n_test_groups=0,
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_rejects_n_splits_below_two(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="at least 2"):
            CombinatorialPurgedCV(
                n_splits=1,
                n_test_groups=1,
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_purge_and_embargo_applied_per_fold(self) -> None:
        pred, evalu = _times(n=24, horizon_days=2)
        cv = CombinatorialPurgedCV(
            n_splits=6,
            n_test_groups=2,
            purge_horizon="1D",
            embargo="1D",
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((24, 1))  # noqa: N806
        for train_idx, test_idx in cv.split(X):
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="1D")
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo="1D")


class TestCombinatorialPurgedCVBacktestPaths:
    def test_returns_n_paths_by_n_samples_matrix(self) -> None:
        """For N=4 K=2, expect 3 paths x 16 samples."""
        pred, evalu = _times(n=16, horizon_days=1)
        cv = CombinatorialPurgedCV(
            n_splits=4,
            n_test_groups=2,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.arange(16).reshape(-1, 1).astype(float)  # noqa: N806
        y = np.arange(16).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FitFailedWarning)
            paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
        assert paths.shape == (3, 16)

    def test_each_path_is_complete_with_finite_values(self) -> None:
        """N=4 K=1: each fold tests exactly one group; no fold spans the full
        timeline so no training set collapses — every path is fully finite."""
        pred, evalu = _times(n=16, horizon_days=1)
        cv = CombinatorialPurgedCV(
            n_splits=4,
            n_test_groups=1,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.arange(16).reshape(-1, 1).astype(float)  # noqa: N806
        y = np.arange(16).astype(float)
        paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
        assert np.all(np.isfinite(paths))

    def test_path_count_matches_combinatorial_identity(self) -> None:
        """n_paths = C(N-1, K-1) for every valid (N, K)."""
        for n_splits, n_test_groups in [(4, 2), (5, 2), (6, 2), (6, 3)]:
            n_samples = 6 * n_splits  # 6 rows per group
            pred = pd.Series(pd.date_range("2024-01-01", periods=n_samples, freq="D"))
            evalu = pred + pd.Timedelta(days=1)
            cv = CombinatorialPurgedCV(
                n_splits=n_splits,
                n_test_groups=n_test_groups,
                prediction_times=pred,
                evaluation_times=evalu,
            )
            X = np.zeros((n_samples, 1))  # noqa: N806
            y = np.zeros(n_samples)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FitFailedWarning)
                paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
            assert paths.shape[0] == comb(n_splits - 1, n_test_groups - 1)
            assert paths.shape[1] == n_samples

    def test_non_adjacent_test_groups_do_not_collapse_training_set(self) -> None:
        """CPCV purges each selected test block locally, so a fold testing
        groups (0, N-1) can still train on the middle groups."""
        pred = pd.Series(pd.date_range("2024-01-01", periods=24, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        cv = CombinatorialPurgedCV(
            n_splits=4,
            n_test_groups=2,
            purge_horizon="1D",
            embargo="1D",
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((24, 1))  # noqa: N806
        y = np.zeros(24)
        paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
        assert np.all(np.isfinite(paths))

        fold_by_test = {tuple(test_idx.tolist()): train_idx for train_idx, test_idx in cv.split(X)}
        non_adjacent_test = tuple([*range(0, 6), *range(18, 24)])
        train_idx = fold_by_test[non_adjacent_test]
        assert len(train_idx) > 0
        assert set(train_idx.tolist()).issubset(set(range(6, 18)))


class TestCombinatorialPurgedCVBacktestPathsAPI:
    def test_estimator_must_have_fit_and_predict(self) -> None:
        """A non-estimator object raises AttributeError or TypeError when
        backtest_paths tries to call .fit on it."""
        pred, evalu = _times(n=16)
        cv = CombinatorialPurgedCV(
            n_splits=4,
            n_test_groups=2,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((16, 1))  # noqa: N806
        y = np.zeros(16)
        with pytest.raises((AttributeError, TypeError)):
            cv.backtest_paths("not an estimator", X, y)
