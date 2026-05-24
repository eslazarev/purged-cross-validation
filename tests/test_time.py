"""Unit tests for purgedcv._time."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from purgedcv._time import horizons_overlap, parse_horizon, validate_times

T = pd.Timestamp


class TestParseHorizon:
    def test_pandas_offset_string_days(self) -> None:
        assert parse_horizon("2D") == pd.Timedelta(days=2)

    def test_pandas_offset_string_hours(self) -> None:
        assert parse_horizon("6h") == pd.Timedelta(hours=6)

    def test_pandas_offset_string_minutes(self) -> None:
        assert parse_horizon("30min") == pd.Timedelta(minutes=30)

    def test_pandas_offset_string_weeks(self) -> None:
        assert parse_horizon("1W") == pd.Timedelta(weeks=1)

    def test_pandas_timedelta_passthrough(self) -> None:
        td = pd.Timedelta(days=3)
        assert parse_horizon(td) == td

    def test_python_timedelta(self) -> None:
        assert parse_horizon(timedelta(hours=4)) == pd.Timedelta(hours=4)

    def test_numpy_timedelta64(self) -> None:
        td = np.timedelta64(5, "D")
        assert parse_horizon(td) == pd.Timedelta(days=5)

    def test_zero_horizon_allowed(self) -> None:
        assert parse_horizon("0D") == pd.Timedelta(0)

    @pytest.mark.parametrize("value", ["NaT", "nat", "nan", np.timedelta64("NaT")])
    def test_rejects_missing_horizon(self, value: object) -> None:
        with pytest.raises(ValueError, match="non-missing"):
            parse_horizon(value)  # type: ignore[arg-type]

    def test_rejects_negative_string(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            parse_horizon("-1D")

    def test_rejects_negative_timedelta(self) -> None:
        with pytest.raises(ValueError, match="must be non-negative"):
            parse_horizon(pd.Timedelta(days=-1))

    def test_rejects_calendar_month(self) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            parse_horizon("M")

    def test_rejects_calendar_year(self) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            parse_horizon("Y")

    def test_rejects_garbage_string(self) -> None:
        with pytest.raises(ValueError):
            parse_horizon("not-a-horizon")

    def test_rejects_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="Unsupported horizon type"):
            parse_horizon(3.14)  # type: ignore[arg-type]


class TestHorizonsOverlap:
    def test_disjoint_before(self) -> None:
        assert (
            horizons_overlap(T("2024-01-01"), T("2024-01-02"), T("2024-01-03"), T("2024-01-04"))
            is False
        )

    def test_disjoint_after(self) -> None:
        assert (
            horizons_overlap(T("2024-01-03"), T("2024-01-04"), T("2024-01-01"), T("2024-01-02"))
            is False
        )

    def test_touching_no_overlap(self) -> None:
        # Half-open convention: [a, b) and [b, c) do NOT overlap.
        assert (
            horizons_overlap(T("2024-01-01"), T("2024-01-02"), T("2024-01-02"), T("2024-01-03"))
            is False
        )

    def test_partial_overlap(self) -> None:
        assert (
            horizons_overlap(T("2024-01-01"), T("2024-01-03"), T("2024-01-02"), T("2024-01-04"))
            is True
        )

    def test_one_contains_other(self) -> None:
        assert (
            horizons_overlap(T("2024-01-01"), T("2024-01-05"), T("2024-01-02"), T("2024-01-03"))
            is True
        )

    def test_identical(self) -> None:
        assert (
            horizons_overlap(T("2024-01-01"), T("2024-01-02"), T("2024-01-01"), T("2024-01-02"))
            is True
        )

    def test_rejects_missing_endpoints(self) -> None:
        with pytest.raises(ValueError, match="non-missing"):
            horizons_overlap(
                T("2024-01-01"),
                pd.NaT,  # type: ignore[arg-type]
                T("2024-01-01"),
                T("2024-01-02"),
            )

    def test_rejects_reversed_intervals(self) -> None:
        with pytest.raises(ValueError, match="a_end"):
            horizons_overlap(T("2024-01-02"), T("2024-01-01"), T("2024-01-01"), T("2024-01-02"))
        with pytest.raises(ValueError, match="b_end"):
            horizons_overlap(T("2024-01-01"), T("2024-01-02"), T("2024-01-02"), T("2024-01-01"))

    def test_symmetric(self) -> None:
        a = horizons_overlap(T("2024-01-01"), T("2024-01-03"), T("2024-01-02"), T("2024-01-04"))
        b = horizons_overlap(T("2024-01-02"), T("2024-01-04"), T("2024-01-01"), T("2024-01-03"))
        assert a == b is True

    def test_one_inside_touching_end(self) -> None:
        # [Jan 1, Jan 5) contains [Jan 4, Jan 5): the inner's end touches the
        # outer's end. Under half-open, they DO overlap (both share [Jan 4, Jan 5)).
        assert (
            horizons_overlap(T("2024-01-01"), T("2024-01-05"), T("2024-01-04"), T("2024-01-05"))
            is True
        )


class TestValidateTimes:
    def test_valid_strict_monotonic(self, daily_index_20: pd.DatetimeIndex) -> None:
        pred = pd.Series(daily_index_20)
        evalu = pred + pd.Timedelta(days=1)
        validate_times(pred, evalu, require_monotonic=True)

    def test_valid_non_monotonic_when_allowed(self) -> None:
        pred = pd.Series(pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-02"]))
        evalu = pred + pd.Timedelta(days=1)
        validate_times(pred, evalu, require_monotonic=False)

    def test_valid_with_equal_pred_and_eval(self) -> None:
        """A zero-length horizon (pred == eval) is allowed."""
        pred = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        evalu = pred.copy()
        validate_times(pred, evalu)

    def test_rejects_length_mismatch(self) -> None:
        pred = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        evalu = pd.Series(pd.to_datetime(["2024-01-02"]))
        with pytest.raises(ValueError, match="length"):
            validate_times(pred, evalu)

    def test_rejects_numeric_times(self) -> None:
        pred = pd.Series([1, 2, 3])
        evalu = pd.Series([2, 3, 4])
        with pytest.raises(ValueError, match="datetime-like"):
            validate_times(pred, evalu)

    def test_rejects_string_times_until_converted(self) -> None:
        pred = pd.Series(["2024-01-01", "2024-01-02"])
        evalu = pd.Series(["2024-01-02", "2024-01-03"])
        with pytest.raises(ValueError, match="datetime-like"):
            validate_times(pred, evalu)

    def test_accepts_timedelta_times(self) -> None:
        pred = pd.Series(pd.to_timedelta([0, 1, 2], unit="D"))
        evalu = pred + pd.Timedelta(days=1)
        validate_times(pred, evalu)

    def test_rejects_mixed_datetime_and_timedelta_times(self) -> None:
        pred = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D"))
        evalu = pd.Series(pd.to_timedelta([1, 2, 3], unit="D"))
        with pytest.raises(ValueError, match="same temporal dtype"):
            validate_times(pred, evalu)

    def test_rejects_evaluation_before_prediction(self) -> None:
        pred = pd.Series(pd.to_datetime(["2024-01-02"]))
        evalu = pd.Series(pd.to_datetime(["2024-01-01"]))
        with pytest.raises(ValueError, match="evaluation_times"):
            validate_times(pred, evalu)

    def test_error_message_names_offending_row(self) -> None:
        pred = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]))
        evalu = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
        with pytest.raises(ValueError, match=r"index 2"):
            validate_times(pred, evalu)

    def test_rejects_nat_in_prediction(self) -> None:
        pred = pd.Series([pd.Timestamp("2024-01-01"), pd.NaT])
        evalu = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"]))
        with pytest.raises(ValueError, match="NaT"):
            validate_times(pred, evalu)

    def test_rejects_nat_in_evaluation(self) -> None:
        pred = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
        evalu = pd.Series([pd.Timestamp("2024-01-02"), pd.NaT])
        with pytest.raises(ValueError, match="NaT"):
            validate_times(pred, evalu)

    def test_rejects_non_monotonic_when_required(self) -> None:
        pred = pd.Series(pd.to_datetime(["2024-01-03", "2024-01-01"]))
        evalu = pred + pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="monotonic"):
            validate_times(pred, evalu, require_monotonic=True)
