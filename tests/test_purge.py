"""Unit tests for purgedcv._purge (Domain D2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purgedcv._purge import purge


def _make_horizon_dataset(horizon_days: int = 2, n: int = 20) -> tuple[pd.Series, pd.Series]:
    """n daily prediction times with a `horizon_days` evaluation horizon."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return pred, evalu


class TestPurgeBasic:
    def test_no_overlap_keeps_all_train(self) -> None:
        """Well-separated train/test → all training rows kept."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.arange(0, 5)
        test_idx = np.arange(10, 15)
        result = purge(train_idx, test_idx, pred, evalu)
        np.testing.assert_array_equal(result, train_idx)

    def test_rejects_negative_purge_horizon(self) -> None:
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        with pytest.raises(ValueError, match="non-negative"):
            purge(
                np.arange(0, 5),
                np.arange(10, 15),
                pred,
                evalu,
                purge_horizon=pd.Timedelta(days=-1),
            )

    def test_rejects_missing_purge_horizon(self) -> None:
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        with pytest.raises(ValueError, match="non-missing"):
            purge(
                np.arange(0, 5),
                np.arange(10, 15),
                pred,
                evalu,
                purge_horizon=pd.NaT,  # type: ignore[arg-type]
            )

    def test_adjacent_train_dropped_with_long_horizon(self) -> None:
        """2-day horizon: train row 9 [Jan 10, Jan 12) overlaps test starting Jan 11."""
        pred, evalu = _make_horizon_dataset(horizon_days=2)
        train_idx = np.array([9])
        test_idx = np.arange(10, 15)
        result = purge(train_idx, test_idx, pred, evalu)
        assert result.size == 0

    def test_post_test_overlap_dropped(self) -> None:
        """Train row whose pred lies inside the test window must be dropped."""
        pred, evalu = _make_horizon_dataset(horizon_days=2)
        train_idx = np.array([14])
        test_idx = np.arange(10, 14)
        result = purge(train_idx, test_idx, pred, evalu)
        assert result.size == 0

    def test_far_post_test_kept(self) -> None:
        """Train row strictly past test_end is kept."""
        pred, evalu = _make_horizon_dataset(horizon_days=2)
        train_idx = np.array([18])
        test_idx = np.arange(10, 14)
        result = purge(train_idx, test_idx, pred, evalu)
        np.testing.assert_array_equal(result, train_idx)

    def test_preserves_dtype_and_order(self) -> None:
        """Output preserves input order and dtype when no rows are dropped."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.array([5, 3, 1, 4, 2, 0], dtype=np.int64)
        test_idx = np.arange(15, 20)
        result = purge(train_idx, test_idx, pred, evalu)
        np.testing.assert_array_equal(result, train_idx)
        assert result.dtype == np.int64

    def test_empty_train(self) -> None:
        pred, evalu = _make_horizon_dataset()
        result = purge(np.array([], dtype=int), np.arange(5, 10), pred, evalu)
        assert result.size == 0

    def test_empty_test_keeps_all_train(self) -> None:
        pred, evalu = _make_horizon_dataset()
        train = np.arange(0, 10)
        result = purge(train, np.array([], dtype=int), pred, evalu)
        np.testing.assert_array_equal(result, train)

    def test_purged_train_passes_diagnostic(self) -> None:
        """The output of purge must satisfy assert_no_temporal_leakage."""
        from purgedcv.diagnostics import assert_no_temporal_leakage

        pred, evalu = _make_horizon_dataset(horizon_days=3)
        train_idx = np.arange(0, 20)
        test_idx = np.arange(10, 15)
        purged = purge(train_idx, test_idx, pred, evalu)
        assert_no_temporal_leakage(purged, test_idx, pred, evalu)

    def test_disjoint_test_blocks_do_not_purge_middle_train_block(self) -> None:
        """Non-contiguous CPCV-style test blocks must not create one giant
        convex-hull purge window."""
        pred, evalu = _make_horizon_dataset(horizon_days=1, n=12)
        train_idx = np.array([3, 4, 5, 6, 7, 8])
        test_idx = np.array([0, 1, 2, 9, 10, 11])

        result = purge(train_idx, test_idx, pred, evalu)

        np.testing.assert_array_equal(result, train_idx)


class TestPurgeWithPadding:
    def test_padding_extends_purge_zone(self) -> None:
        """1-day horizon, train row 9 [Jan 10, Jan 11) touches test start Jan 11.
        Without padding -> kept (half-open touch). With purge_horizon=1D the
        test_start shifts to Jan 10, so train_eval(Jan 11) > Jan 10 -> dropped."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.array([9])
        test_idx = np.arange(10, 15)
        result_no_pad = purge(train_idx, test_idx, pred, evalu)
        np.testing.assert_array_equal(result_no_pad, train_idx)
        result_pad = purge(train_idx, test_idx, pred, evalu, purge_horizon=pd.Timedelta(days=1))
        assert result_pad.size == 0

    def test_padding_subset_property(self) -> None:
        """For padding >= 0, the padded purge output is a subset of the
        unpadded output (more aggressive filter on a larger window)."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.arange(0, 20)
        test_idx = np.arange(10, 15)
        no_pad = set(purge(train_idx, test_idx, pred, evalu).tolist())
        with_pad = set(
            purge(
                train_idx,
                test_idx,
                pred,
                evalu,
                purge_horizon=pd.Timedelta(days=3),
            ).tolist()
        )
        assert with_pad.issubset(no_pad)

    def test_padding_zero_equals_no_padding(self) -> None:
        """purge_horizon=Timedelta(0) returns the same result as no argument."""
        pred, evalu = _make_horizon_dataset(horizon_days=2)
        train_idx = np.arange(0, 15)
        test_idx = np.arange(10, 15)
        a = purge(train_idx, test_idx, pred, evalu)
        b = purge(train_idx, test_idx, pred, evalu, purge_horizon=pd.Timedelta(0))
        np.testing.assert_array_equal(a, b)

    def test_padding_oversized_drops_all_train(self) -> None:
        """A purge_horizon larger than the data range expels every train row."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.arange(0, 20)
        test_idx = np.arange(10, 11)
        result = purge(train_idx, test_idx, pred, evalu, purge_horizon=pd.Timedelta(days=1000))
        assert result.size == 0


class TestPurgeAFMLSnippet71:
    """Reproduction of the canonical purge example from
    *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
    chapter 7 section 7.4.1, Snippet 7.1.

    Setup: ten consecutive daily samples whose labels span a 3-day horizon
    (overlapping). The test fold is samples 5, 6, 7 with combined horizon
    [Jan 6, Jan 11). All training rows whose 3-day horizon intersects the
    test horizon must be dropped; training rows whose horizon ends at or
    before Jan 6, or whose prediction time is at or after Jan 11, are kept.
    """

    def test_snippet_7_1_purges_expected_rows(self) -> None:
        pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        evalu = pred + pd.Timedelta(days=3)
        train_idx = np.array([0, 1, 2, 3, 4, 8, 9])
        test_idx = np.array([5, 6, 7])

        result = purge(train_idx, test_idx, pred, evalu)

        # Hand-verified against the half-open formulation:
        #   test_start = pred[5] = Jan 6
        #   test_end   = evalu[7] = Jan 8 + 3 days = Jan 11
        # Row | pred   | eval   | eval <= Jan 6 | pred >= Jan 11 | kept?
        #  0  | Jan 1  | Jan 4  | True          | False          | YES
        #  1  | Jan 2  | Jan 5  | True          | False          | YES
        #  2  | Jan 3  | Jan 6  | True          | False          | YES (boundary)
        #  3  | Jan 4  | Jan 7  | False         | False          | no  (overlap)
        #  4  | Jan 5  | Jan 8  | False         | False          | no  (overlap)
        #  8  | Jan 9  | Jan 12 | False         | False          | no  (overlap)
        #  9  | Jan 10 | Jan 13 | False         | False          | no  (overlap)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_snippet_7_1_with_purge_horizon_drops_boundary_too(self) -> None:
        """With purge_horizon=1D the test window expands to [Jan 5, Jan 12).
        Row 2 (evalu=Jan 6) is now strictly > Jan 5, so it is dropped too."""
        pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        evalu = pred + pd.Timedelta(days=3)
        train_idx = np.array([0, 1, 2, 3, 4, 8, 9])
        test_idx = np.array([5, 6, 7])

        result = purge(train_idx, test_idx, pred, evalu, purge_horizon=pd.Timedelta(days=1))

        expected = np.array([0, 1])
        np.testing.assert_array_equal(result, expected)
