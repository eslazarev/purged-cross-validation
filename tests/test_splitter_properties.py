"""Hypothesis property tests for all four D5 splitters.

The contract:
- Every fold emitted by every splitter passes assert_no_temporal_leakage
  with the splitter's own purge_horizon.
- Every fold passes assert_embargo_respected with the splitter's own
  embargo.
- Train and test indices are always disjoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from purgedcv import (
    CombinatorialPurgedCV,
    PurgedGroupKFold,
    PurgedKFold,
    WalkForwardSplit,
)
from purgedcv.diagnostics import (
    assert_embargo_respected,
    assert_groups_disjoint,
    assert_no_temporal_leakage,
)


@st.composite
def split_inputs(
    draw: st.DrawFn,
) -> tuple[pd.Series, pd.Series, int, int]:
    n = draw(st.integers(min_value=20, max_value=60))
    horizon_days = draw(st.integers(min_value=1, max_value=3))
    purge_days = draw(st.integers(min_value=0, max_value=3))
    embargo_days = draw(st.integers(min_value=0, max_value=3))
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return pred, evalu, purge_days, embargo_days


class TestWalkForwardSplitProperties:
    @settings(max_examples=100, deadline=None)
    @given(split_inputs())
    def test_each_fold_is_clean(
        self,
        case: tuple[pd.Series, pd.Series, int, int],
    ) -> None:
        pred, evalu, purge_days, embargo_days = case
        purge_h = pd.Timedelta(days=purge_days)
        embargo = pd.Timedelta(days=embargo_days)
        n = len(pred)
        cv = WalkForwardSplit(
            n_splits=3,
            test_size=max(2, n // 8),
            purge_horizon=purge_h,
            embargo=embargo,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((n, 1))  # noqa: N806
        for train_idx, test_idx in cv.split(X):
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon=purge_h)
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=embargo)


class TestPurgedKFoldProperties:
    @settings(max_examples=100, deadline=None)
    @given(split_inputs())
    def test_each_fold_is_clean(
        self,
        case: tuple[pd.Series, pd.Series, int, int],
    ) -> None:
        pred, evalu, purge_days, embargo_days = case
        purge_h = pd.Timedelta(days=purge_days)
        embargo = pd.Timedelta(days=embargo_days)
        n = len(pred)
        cv = PurgedKFold(
            n_splits=4,
            purge_horizon=purge_h,
            embargo=embargo,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((n, 1))  # noqa: N806
        for train_idx, test_idx in cv.split(X):
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon=purge_h)
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=embargo)


class TestPurgedGroupKFoldProperties:
    @settings(max_examples=100, deadline=None)
    @given(split_inputs())
    def test_each_fold_is_clean_and_group_disjoint(
        self,
        case: tuple[pd.Series, pd.Series, int, int],
    ) -> None:
        pred, evalu, purge_days, embargo_days = case
        purge_h = pd.Timedelta(days=purge_days)
        embargo = pd.Timedelta(days=embargo_days)
        n = len(pred)
        # 6 groups distributed evenly with the remainder going to group 5.
        per_group = n // 6
        groups = pd.Series(np.repeat(np.arange(6), per_group).tolist() + [5] * (n - 6 * per_group))
        cv = PurgedGroupKFold(
            n_splits=3,
            purge_horizon=purge_h,
            embargo=embargo,
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        X = np.zeros((n, 1))  # noqa: N806
        for train_idx, test_idx in cv.split(X):
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon=purge_h)
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=embargo)
            assert_groups_disjoint(train_idx, test_idx, groups)


class TestCombinatorialPurgedCVProperties:
    @settings(max_examples=50, deadline=None)
    @given(split_inputs())
    def test_each_fold_is_clean(
        self,
        case: tuple[pd.Series, pd.Series, int, int],
    ) -> None:
        pred, evalu, purge_days, embargo_days = case
        purge_h = pd.Timedelta(days=purge_days)
        embargo = pd.Timedelta(days=embargo_days)
        n = len(pred)
        cv = CombinatorialPurgedCV(
            n_splits=5,
            n_test_groups=2,
            purge_horizon=purge_h,
            embargo=embargo,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((n, 1))  # noqa: N806
        for train_idx, test_idx in cv.split(X):
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon=purge_h)
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=embargo)
