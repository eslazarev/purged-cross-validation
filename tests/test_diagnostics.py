"""Unit tests for purgedcv.diagnostics (Domain D8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purgedcv._typing import NDArrayAny
from purgedcv.diagnostics import (
    assert_embargo_respected,
    assert_groups_disjoint,
    assert_no_temporal_leakage,
    compute_overlap_fraction,
)
from purgedcv.exceptions import (
    EmbargoViolationError,
    GroupLeakageError,
    TemporalLeakageError,
)


def _make_horizon_dataset(horizon_days: int = 1, n: int = 20) -> tuple[pd.Series, pd.Series]:
    """n daily prediction times with a constant `horizon_days` evaluation horizon."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return pred, evalu


class TestAssertNoTemporalLeakage:
    def test_clean_split_silent(self) -> None:
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.arange(0, 5)
        test_idx = np.arange(10, 15)
        assert_no_temporal_leakage(train_idx, test_idx, pred, evalu)

    def test_overlapping_horizons_raise(self) -> None:
        """Train row 10 deliberately included in both sides — must be caught."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.array([10])
        test_idx = np.arange(10, 15)
        with pytest.raises(TemporalLeakageError, match="row 10"):
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu)

    def test_adjacent_horizon_with_long_label_leaks(self) -> None:
        """2-day horizon: row 9 [Jan 10, Jan 12) overlaps test starting Jan 11."""
        pred, evalu = _make_horizon_dataset(horizon_days=2)
        train_idx = np.array([9])
        test_idx = np.array([10])
        with pytest.raises(TemporalLeakageError, match="row 9"):
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu)

    def test_purge_horizon_extends_zone(self) -> None:
        """purge_horizon padding extends the test window on both sides."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.array([9])
        test_idx = np.arange(10, 15)
        # Without padding, train row 9 [Jan 10, Jan 11) touches test start
        # Jan 11 but does not overlap (half-open).
        assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon=pd.Timedelta(0))
        # With 1-day padding on both sides, test window becomes [Jan 10, Jan 17)
        # and train row 9 [Jan 10, Jan 11) overlaps that.
        with pytest.raises(TemporalLeakageError):
            assert_no_temporal_leakage(
                train_idx, test_idx, pred, evalu, purge_horizon=pd.Timedelta(days=1)
            )

    def test_purge_horizon_accepts_string(self) -> None:
        """Diagnostic should accept the same string forms parse_horizon accepts."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.array([9])
        test_idx = np.arange(10, 15)
        with pytest.raises(TemporalLeakageError):
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="1D")

    def test_empty_train_silent(self) -> None:
        pred, evalu = _make_horizon_dataset()
        assert_no_temporal_leakage(np.array([], dtype=int), np.arange(10, 15), pred, evalu)

    def test_empty_test_silent(self) -> None:
        pred, evalu = _make_horizon_dataset()
        assert_no_temporal_leakage(np.arange(10), np.array([], dtype=int), pred, evalu)

    @pytest.mark.parametrize(
        "train_idx",
        [
            np.array([-1]),
            np.array([20]),
            np.array([1.5]),
            np.array([[1, 2]]),
            np.array([True]),
            np.array([1, 1]),
        ],
    )
    def test_rejects_invalid_positional_indices(self, train_idx: NDArrayAny) -> None:
        pred, evalu = _make_horizon_dataset()
        with pytest.raises((TypeError, ValueError)):
            assert_no_temporal_leakage(train_idx, np.array([10]), pred, evalu)

    def test_rejects_malformed_times(self) -> None:
        pred = pd.Series([1, 2, 3])
        evalu = pd.Series([2, 3, 4])
        with pytest.raises(ValueError, match="datetime-like"):
            assert_no_temporal_leakage(np.array([0]), np.array([1]), pred, evalu)

    def test_error_message_includes_horizon_bounds(self) -> None:
        """Helpful error message for debugging includes the offending interval."""
        pred, evalu = _make_horizon_dataset(horizon_days=2)
        train_idx = np.array([9])
        test_idx = np.array([10])
        with pytest.raises(TemporalLeakageError) as exc_info:
            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu)
        message = str(exc_info.value)
        assert "training horizon" in message
        assert "test horizon" in message

    def test_disjoint_test_blocks_check_local_horizons(self) -> None:
        """Middle training rows between non-contiguous test blocks are clean."""
        pred, evalu = _make_horizon_dataset(horizon_days=1, n=12)
        train_idx = np.array([3, 4, 5, 6, 7, 8])
        test_idx = np.array([0, 1, 2, 9, 10, 11])
        assert_no_temporal_leakage(train_idx, test_idx, pred, evalu)


class TestAssertEmbargoRespected:
    def test_zero_embargo_is_identity(self) -> None:
        """embargo=0 always returns silently, regardless of split layout."""
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([15])
        test_idx = np.arange(10, 15)
        assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(0))

    def test_train_inside_embargo_raises(self) -> None:
        """Closed window [eval_max, eval_max+embargo]: train row 15 (pred=Jan 16,
        which equals eval_max=Jan 16) is inside the embargo zone for embargo=2D."""
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([15])
        test_idx = np.arange(10, 15)
        with pytest.raises(EmbargoViolationError, match="row 15"):
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=2))

    def test_train_outside_embargo_silent(self) -> None:
        """Train row whose pred is strictly past the embargo cutoff is fine."""
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([18])
        test_idx = np.arange(10, 15)
        assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=2))

    def test_pre_test_train_never_flagged(self) -> None:
        """Embargo is asymmetric: rows whose pred is before eval_max are kept."""
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([0, 1, 2])
        test_idx = np.arange(10, 15)
        assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=100))

    def test_embargo_accepts_string(self) -> None:
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([15])
        test_idx = np.arange(10, 15)
        with pytest.raises(EmbargoViolationError):
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo="2D")

    def test_empty_train_silent(self) -> None:
        pred, evalu = _make_horizon_dataset()
        assert_embargo_respected(
            np.array([], dtype=int),
            np.arange(10, 15),
            pred,
            evalu,
            embargo=pd.Timedelta(days=2),
        )

    def test_rejects_invalid_positional_indices(self) -> None:
        pred, evalu = _make_horizon_dataset()
        with pytest.raises(ValueError, match="negative"):
            assert_embargo_respected(np.array([15]), np.array([-1]), pred, evalu, embargo="1D")

    def test_error_message_includes_window_bounds(self) -> None:
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([15])
        test_idx = np.arange(10, 15)
        with pytest.raises(EmbargoViolationError) as exc_info:
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=2))
        message = str(exc_info.value)
        assert "embargo" in message.lower()
        assert "row 15" in message

    def test_disjoint_test_blocks_check_each_embargo_window(self) -> None:
        pred, evalu = _make_horizon_dataset(horizon_days=1, n=12)
        test_idx = np.array([0, 1, 2, 9, 10, 11])
        with pytest.raises(EmbargoViolationError, match="row 3"):
            assert_embargo_respected(
                np.array([3]),
                test_idx,
                pred,
                evalu,
                embargo=pd.Timedelta(days=1),
            )


class TestAssertGroupsDisjoint:
    def test_disjoint_silent(self) -> None:
        groups = pd.Series([0, 0, 1, 1, 2, 2])
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5])
        assert_groups_disjoint(train_idx, test_idx, groups)

    def test_overlap_raises(self) -> None:
        """Group 2 appears in both train and test -> raise."""
        groups = pd.Series([0, 0, 1, 1, 2, 2])
        train_idx = np.array([0, 1, 4])
        test_idx = np.array([2, 3, 5])
        with pytest.raises(GroupLeakageError, match="group 2"):
            assert_groups_disjoint(train_idx, test_idx, groups)

    def test_string_groups(self) -> None:
        """Non-integer group identifiers should also work."""
        groups = pd.Series(["A", "A", "B", "B", "C"])
        train_idx = np.array([0, 1])
        test_idx = np.array([0, 2, 3])
        with pytest.raises(GroupLeakageError, match="A"):
            assert_groups_disjoint(train_idx, test_idx, groups)

    def test_rejects_missing_group_labels(self) -> None:
        groups = pd.Series(["A", np.nan, "B"])
        with pytest.raises(ValueError, match="missing"):
            assert_groups_disjoint(np.array([0, 1]), np.array([2]), groups)

    def test_empty_train_silent(self) -> None:
        groups = pd.Series([0, 0, 1, 1])
        assert_groups_disjoint(np.array([], dtype=int), np.array([2, 3]), groups)

    def test_empty_test_silent(self) -> None:
        groups = pd.Series([0, 0, 1, 1])
        assert_groups_disjoint(np.array([0, 1]), np.array([], dtype=int), groups)

    def test_rejects_invalid_positional_indices(self) -> None:
        groups = pd.Series([0, 0, 1, 1])
        with pytest.raises(ValueError, match="out-of-bounds"):
            assert_groups_disjoint(np.array([0]), np.array([4]), groups)

    def test_error_message_reports_overlap_count(self) -> None:
        """When multiple groups leak, the count is reported for triage."""
        groups = pd.Series([0, 1, 2, 3])
        train_idx = np.array([0, 1])
        test_idx = np.array([0, 1, 2, 3])
        with pytest.raises(GroupLeakageError) as exc_info:
            assert_groups_disjoint(train_idx, test_idx, groups)
        assert "2" in str(exc_info.value)


class TestComputeOverlapFraction:
    def test_clean_split_zero(self) -> None:
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.arange(0, 5)
        test_idx = np.arange(10, 15)
        assert compute_overlap_fraction(train_idx, test_idx, pred, evalu) == 0.0

    def test_all_leak(self) -> None:
        """If train_idx equals test_idx, every training row leaks."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.arange(10, 15)
        test_idx = np.arange(10, 15)
        assert compute_overlap_fraction(train_idx, test_idx, pred, evalu) == 1.0

    def test_half_leak(self) -> None:
        """Mixed: 2 of 4 training rows have horizons in the test window."""
        pred, evalu = _make_horizon_dataset(horizon_days=1)
        train_idx = np.array([10, 11, 0, 1])
        test_idx = np.arange(10, 15)
        result = compute_overlap_fraction(train_idx, test_idx, pred, evalu)
        assert result == 0.5

    def test_empty_train_is_zero(self) -> None:
        pred, evalu = _make_horizon_dataset()
        result = compute_overlap_fraction(np.array([], dtype=int), np.arange(10, 15), pred, evalu)
        assert result == 0.0

    def test_empty_test_is_zero(self) -> None:
        pred, evalu = _make_horizon_dataset()
        result = compute_overlap_fraction(np.arange(10), np.array([], dtype=int), pred, evalu)
        assert result == 0.0

    def test_rejects_invalid_positional_indices(self) -> None:
        pred, evalu = _make_horizon_dataset()
        with pytest.raises(TypeError, match="integer"):
            compute_overlap_fraction(np.array([0.5]), np.array([10]), pred, evalu)

    def test_non_raising(self) -> None:
        """compute_overlap_fraction is a diagnostic, never raises on leakage."""
        pred, evalu = _make_horizon_dataset()
        train_idx = np.arange(10, 15)
        test_idx = np.arange(10, 15)
        # Returns 1.0 instead of raising.
        result = compute_overlap_fraction(train_idx, test_idx, pred, evalu)
        assert isinstance(result, float)
        assert result == 1.0

    def test_returns_python_float(self) -> None:
        """Return type is built-in float, not np.float64."""
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([10])
        test_idx = np.arange(10, 15)
        result = compute_overlap_fraction(train_idx, test_idx, pred, evalu)
        assert type(result) is float

    def test_disjoint_test_blocks_do_not_count_middle_rows_as_overlap(self) -> None:
        pred, evalu = _make_horizon_dataset(horizon_days=1, n=12)
        train_idx = np.array([3, 4, 5, 6, 7, 8])
        test_idx = np.array([0, 1, 2, 9, 10, 11])
        assert compute_overlap_fraction(train_idx, test_idx, pred, evalu) == 0.0
