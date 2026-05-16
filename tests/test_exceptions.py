"""Unit tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from purgedcv.exceptions import (
    EmbargoViolationError,
    GroupLeakageError,
    TemporalCVError,
    TemporalLeakageError,
)


class TestExceptionHierarchy:
    def test_base_is_value_error(self) -> None:
        assert issubclass(TemporalCVError, ValueError)

    def test_temporal_leakage_is_subclass(self) -> None:
        assert issubclass(TemporalLeakageError, TemporalCVError)

    def test_embargo_violation_is_subclass(self) -> None:
        assert issubclass(EmbargoViolationError, TemporalCVError)

    def test_group_leakage_is_subclass(self) -> None:
        assert issubclass(GroupLeakageError, TemporalCVError)

    def test_can_raise_and_catch_base(self) -> None:
        with pytest.raises(TemporalCVError):
            raise TemporalLeakageError("test")

    def test_can_catch_as_value_error(self) -> None:
        """A user who broadly catches ValueError still gets our exceptions."""
        with pytest.raises(ValueError):
            raise EmbargoViolationError("test")

    def test_message_round_trips(self) -> None:
        with pytest.raises(TemporalLeakageError, match="row 5"):
            raise TemporalLeakageError("leak detected at row 5")

    def test_three_subclasses_are_distinct(self) -> None:
        """Catching one subclass must not catch another."""
        with pytest.raises(EmbargoViolationError):
            try:
                raise EmbargoViolationError("e")
            except TemporalLeakageError:
                pytest.fail("EmbargoViolationError should not be caught as TemporalLeakageError")
