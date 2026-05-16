"""Exception types raised by purged-cross-validation.

The hierarchy is rooted at :class:`TemporalCVError`, which itself subclasses
:class:`ValueError`. Callers who want to catch any purged-cross-validation error use
``except TemporalCVError``; callers who already broadly catch ``ValueError``
also receive our errors without code changes.
"""

from __future__ import annotations


class TemporalCVError(ValueError):
    """Base class for all purged-cross-validation errors."""


class TemporalLeakageError(TemporalCVError):
    """Raised when a training row's label horizon overlaps the test horizon."""


class EmbargoViolationError(TemporalCVError):
    """Raised when a training row falls inside the post-test embargo window."""


class GroupLeakageError(TemporalCVError):
    """Raised when a group_id appears in both training and test of the same fold."""
