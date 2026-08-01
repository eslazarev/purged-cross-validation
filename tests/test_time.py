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


class TestCoerce1d:
    def test_numpy_datetime64_passthrough(self) -> None:
        from purgedcv._time import _coerce_1d

        arr = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
        out = _coerce_1d(arr, name="t")
        assert np.issubdtype(out.dtype, np.datetime64)
        np.testing.assert_array_equal(out, arr)

    def test_numpy_timedelta64_passthrough(self) -> None:
        from purgedcv._time import _coerce_1d

        arr = np.array([1, 2], dtype="timedelta64[D]")
        out = _coerce_1d(arr, name="t")
        assert np.issubdtype(out.dtype, np.timedelta64)

    def test_pandas_series_to_numpy(self) -> None:
        from purgedcv._time import _coerce_1d

        s = pd.Series(pd.date_range("2024-01-01", periods=3, freq="D"))
        out = _coerce_1d(s, name="t")
        assert np.issubdtype(out.dtype, np.datetime64)
        assert len(out) == 3

    def test_datetimeindex(self) -> None:
        from purgedcv._time import _coerce_1d

        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        out = _coerce_1d(idx, name="t")
        assert np.issubdtype(out.dtype, np.datetime64)

    def test_list_of_timestamps(self) -> None:
        from purgedcv._time import _coerce_1d

        out = _coerce_1d([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")], name="t")
        assert np.issubdtype(out.dtype, np.datetime64)

    def test_list_of_timedeltas(self) -> None:
        from datetime import timedelta

        from purgedcv._time import _coerce_1d

        out = _coerce_1d([timedelta(days=1), timedelta(days=2)], name="t")
        assert np.issubdtype(out.dtype, np.timedelta64)

    def test_tz_aware_normalized_to_utc_naive(self) -> None:
        from purgedcv._time import _coerce_1d

        s = pd.Series(pd.date_range("2024-01-01", periods=2, freq="D", tz="US/Eastern"))
        out = _coerce_1d(s, name="t")
        assert np.issubdtype(out.dtype, np.datetime64)
        # 2024-01-01 00:00 US/Eastern == 2024-01-01 05:00 UTC
        assert out[0] == np.datetime64("2024-01-01T05:00:00")

    def test_object_list_of_strings_stays_object(self) -> None:
        from purgedcv._time import _coerce_1d

        out = _coerce_1d(["2024-01-01", "2024-01-02"], name="t")
        assert out.dtype == object


class TestValidateTimesInputTypes:
    def test_numpy_datetime64_ok(self) -> None:
        from purgedcv import validate_times

        pred = pd.date_range("2024-01-01", periods=5, freq="D").to_numpy()
        evalu = pred + np.timedelta64(1, "D")
        validate_times(pred, evalu)

    def test_numpy_timedelta64_ok(self) -> None:
        from purgedcv import validate_times

        pred = np.array([1, 2, 3], dtype="timedelta64[D]")
        evalu = np.array([2, 3, 4], dtype="timedelta64[D]")
        validate_times(pred, evalu)

    def test_list_input_ok(self) -> None:
        from purgedcv import validate_times

        pred = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        evalu = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        validate_times(pred, evalu)

    def test_mixed_kind_rejected(self) -> None:
        from purgedcv import validate_times

        pred = pd.date_range("2024-01-01", periods=3, freq="D").to_numpy()
        evalu = np.array([1, 2, 3], dtype="timedelta64[D]")
        with pytest.raises(ValueError, match="same temporal dtype family"):
            validate_times(pred, evalu)

    def test_non_temporal_rejected(self) -> None:
        from purgedcv import validate_times

        with pytest.raises(ValueError, match="datetime-like or timedelta-like"):
            validate_times([1, 2, 3], [2, 3, 4])

    def test_nat_in_numpy_rejected(self) -> None:
        from purgedcv import validate_times

        pred = np.array(["2024-01-01", "NaT"], dtype="datetime64[ns]")
        evalu = np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match="NaT"):
            validate_times(pred, evalu, require_monotonic=False)

    def test_inverted_row_message_has_date(self) -> None:
        from purgedcv import validate_times

        pred = np.array(["2024-01-01", "2024-01-10"], dtype="datetime64[ns]")
        evalu = np.array(["2024-01-02", "2024-01-09"], dtype="datetime64[ns]")
        with pytest.raises(ValueError) as exc:
            validate_times(pred, evalu, require_monotonic=False)
        msg = str(exc.value)
        assert "index 1" in msg
        assert "2024-01-09" in msg


class TestCoerce1dDimensionality:
    def test_rejects_2d_array(self) -> None:
        from purgedcv._time import _coerce_1d

        arr = np.arange(10).reshape(-1, 1)
        with pytest.raises(ValueError, match="1-D array-like"):
            _coerce_1d(arr, name="prediction_times")

    def test_rejects_0d_numpy_scalar(self) -> None:
        from purgedcv._time import _coerce_1d

        with pytest.raises(ValueError, match="1-D array-like"):
            _coerce_1d(np.datetime64("2024-01-01"), name="prediction_times")  # type: ignore[arg-type]

    def test_rejects_pandas_timestamp_scalar(self) -> None:
        from purgedcv._time import _coerce_1d

        with pytest.raises(ValueError, match="1-D array-like"):
            _coerce_1d(pd.Timestamp("2024-01-01"), name="prediction_times")  # type: ignore[arg-type]

    def test_rejects_dataframe(self) -> None:
        from purgedcv._time import _coerce_1d

        df = pd.DataFrame({"t": pd.date_range("2024-01-01", periods=5, freq="D")})
        with pytest.raises(ValueError, match="1-D array-like"):
            _coerce_1d(df, name="prediction_times")

    def test_message_names_the_input_and_shape(self) -> None:
        from purgedcv._time import _coerce_1d

        arr = np.arange(6).reshape(-1, 1)
        with pytest.raises(ValueError) as exc:
            _coerce_1d(arr, name="groups")
        msg = str(exc.value)
        assert "groups" in msg
        assert "2-D" in msg


class TestDimensionalityAtPublicBoundary:
    def test_validate_times_rejects_2d(self) -> None:
        pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D")).to_numpy()
        with pytest.raises(ValueError, match="1-D array-like"):
            validate_times(pred.reshape(-1, 1), pred.reshape(-1, 1))

    def test_splitter_rejects_2d_times(self) -> None:
        from purgedcv import PurgedKFold

        pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D")).to_numpy()
        with pytest.raises(ValueError, match="1-D array-like"):
            PurgedKFold(
                n_splits=3,
                prediction_times=pred.reshape(-1, 1),
                evaluation_times=pred.reshape(-1, 1),
            )

    def test_group_splitter_rejects_2d_groups(self) -> None:
        from purgedcv import PurgedGroupKFold

        n = 9
        pred = pd.date_range("2024-01-01", periods=n, freq="D").to_numpy()
        evalu = pred + np.timedelta64(1, "D")
        groups = np.repeat(np.arange(3), 3).reshape(-1, 1)
        with pytest.raises(ValueError, match="1-D array-like"):
            PurgedGroupKFold(
                n_splits=2, prediction_times=pred, evaluation_times=evalu, groups=groups
            )


class TestRuntimeIntrospection:
    def test_get_type_hints_resolves_on_public_functions(self) -> None:
        import typing

        from purgedcv import apply_embargo, purge, validate_times

        # Regression: TimesLike used to be a string alias referencing names not
        # present in these modules' namespaces, so get_type_hints raised
        # NameError. It is now a concrete Union and must resolve cleanly.
        for fn in (validate_times, purge, apply_embargo):
            hints = typing.get_type_hints(fn)
            assert "prediction_times" in hints
