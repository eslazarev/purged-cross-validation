"""Hypothesis property tests for purge + embargo composed with diagnostics.

These tests assert the foundational contract that future D5 splitters
will lean on: for ANY valid input, the composition

    purge(...)  -> apply_embargo(...)

yields a training index set that passes both
:func:`assert_no_temporal_leakage` and :func:`assert_embargo_respected`
with the same parameters that produced it.

If a future change to the purge/embargo arithmetic or the diagnostic
arithmetic ever drifts apart, these tests will fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from purgedcv._embargo import apply_embargo
from purgedcv._purge import purge
from purgedcv._typing import NDArrayAny
from purgedcv.diagnostics import (
    assert_embargo_respected,
    assert_no_temporal_leakage,
)


@st.composite
def dataset_and_split(
    draw: st.DrawFn,
) -> tuple[pd.Series, pd.Series, NDArrayAny, NDArrayAny]:
    """Generate a dataset with prediction/evaluation times and a disjoint
    train/test split.

    The split is constructed so that train_idx and test_idx are disjoint
    (an essential precondition for the diagnostic contracts) and at least
    one row remains on each side after the split.
    """
    n = draw(st.integers(min_value=8, max_value=40))
    horizon_days = draw(st.integers(min_value=1, max_value=5))
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    all_idx = np.arange(n)
    test_size = draw(st.integers(min_value=1, max_value=n // 2))
    # test_start in [0, n - test_size]: always feasible because test_size <= n//2 < n.
    test_start = draw(st.integers(min_value=0, max_value=n - test_size))
    test_idx = all_idx[test_start : test_start + test_size]
    test_set = set(test_idx.tolist())
    train_idx = np.array([i for i in all_idx if i not in test_set])
    return pred, evalu, train_idx, test_idx


class TestPurgeProperties:
    @settings(max_examples=200, deadline=None)
    @given(dataset_and_split())
    def test_purged_train_has_no_temporal_leakage(
        self,
        case: tuple[pd.Series, pd.Series, NDArrayAny, NDArrayAny],
    ) -> None:
        pred, evalu, train_idx, test_idx = case
        purged = purge(train_idx, test_idx, pred, evalu)
        assert_no_temporal_leakage(purged, test_idx, pred, evalu)

    @settings(max_examples=200, deadline=None)
    @given(dataset_and_split(), st.integers(min_value=0, max_value=10))
    def test_purge_padding_monotonicity(
        self,
        case: tuple[pd.Series, pd.Series, NDArrayAny, NDArrayAny],
        days: int,
    ) -> None:
        """purge(..., purge_horizon=Δ) is a subset of purge(...) for Δ ≥ 0."""
        pred, evalu, train_idx, test_idx = case
        base = set(purge(train_idx, test_idx, pred, evalu).tolist())
        padded = set(
            purge(
                train_idx,
                test_idx,
                pred,
                evalu,
                purge_horizon=pd.Timedelta(days=days),
            ).tolist()
        )
        assert padded.issubset(base)

    @settings(max_examples=100, deadline=None)
    @given(dataset_and_split(), st.integers(min_value=0, max_value=10))
    def test_purged_with_padding_respects_padded_diagnostic(
        self,
        case: tuple[pd.Series, pd.Series, NDArrayAny, NDArrayAny],
        days: int,
    ) -> None:
        """The output of purge(..., purge_horizon=Δ) passes the diagnostic
        with the same purge_horizon=Δ."""
        pred, evalu, train_idx, test_idx = case
        padding = pd.Timedelta(days=days)
        purged = purge(train_idx, test_idx, pred, evalu, purge_horizon=padding)
        assert_no_temporal_leakage(purged, test_idx, pred, evalu, purge_horizon=padding)


class TestEmbargoProperties:
    @settings(max_examples=200, deadline=None)
    @given(dataset_and_split(), st.integers(min_value=0, max_value=10))
    def test_embargoed_train_respects_embargo(
        self,
        case: tuple[pd.Series, pd.Series, NDArrayAny, NDArrayAny],
        embargo_days: int,
    ) -> None:
        pred, evalu, train_idx, test_idx = case
        emb = pd.Timedelta(days=embargo_days)
        embargoed = apply_embargo(train_idx, test_idx, pred, evalu, embargo=emb)
        assert_embargo_respected(embargoed, test_idx, pred, evalu, embargo=emb)

    @settings(max_examples=200, deadline=None)
    @given(dataset_and_split(), st.integers(min_value=0, max_value=10))
    def test_embargo_zero_is_identity(
        self,
        case: tuple[pd.Series, pd.Series, NDArrayAny, NDArrayAny],
        embargo_days: int,
    ) -> None:
        """apply_embargo with embargo=0 is the identity for any input."""
        pred, evalu, train_idx, test_idx = case
        identity = apply_embargo(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(0))
        np.testing.assert_array_equal(identity, train_idx)


class TestPurgePlusEmbargoComposition:
    @settings(max_examples=200, deadline=None)
    @given(
        dataset_and_split(),
        st.integers(min_value=0, max_value=5),
        st.integers(min_value=0, max_value=5),
    )
    def test_purge_then_embargo_clean(
        self,
        case: tuple[pd.Series, pd.Series, NDArrayAny, NDArrayAny],
        purge_days: int,
        embargo_days: int,
    ) -> None:
        """The canonical composition: purge then embargo yields a split
        that passes BOTH diagnostics with the parameters that produced it.

        This is the foundational contract every D5 splitter will inherit.
        """
        pred, evalu, train_idx, test_idx = case
        purge_h = pd.Timedelta(days=purge_days)
        embargo = pd.Timedelta(days=embargo_days)
        purged = purge(train_idx, test_idx, pred, evalu, purge_horizon=purge_h)
        final = apply_embargo(purged, test_idx, pred, evalu, embargo=embargo)
        assert_no_temporal_leakage(final, test_idx, pred, evalu, purge_horizon=purge_h)
        assert_embargo_respected(final, test_idx, pred, evalu, embargo=embargo)

    @settings(max_examples=100, deadline=None)
    @given(
        dataset_and_split(),
        st.integers(min_value=0, max_value=5),
        st.integers(min_value=0, max_value=5),
    )
    def test_purge_embargo_commute_in_set(
        self,
        case: tuple[pd.Series, pd.Series, NDArrayAny, NDArrayAny],
        purge_days: int,
        embargo_days: int,
    ) -> None:
        """purge and apply_embargo commute as set operations (the final
        index set is the same regardless of which filter runs first)."""
        pred, evalu, train_idx, test_idx = case
        purge_h = pd.Timedelta(days=purge_days)
        embargo = pd.Timedelta(days=embargo_days)
        purge_then_embargo = set(
            apply_embargo(
                purge(train_idx, test_idx, pred, evalu, purge_horizon=purge_h),
                test_idx,
                pred,
                evalu,
                embargo=embargo,
            ).tolist()
        )
        embargo_then_purge = set(
            purge(
                apply_embargo(train_idx, test_idx, pred, evalu, embargo=embargo),
                test_idx,
                pred,
                evalu,
                purge_horizon=purge_h,
            ).tolist()
        )
        assert purge_then_embargo == embargo_then_purge
