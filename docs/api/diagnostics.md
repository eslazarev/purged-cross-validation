# Diagnostics and exceptions

Inspect what a splitter removed and check temporal, group, and embargo constraints.
Start with [`audit_splitter`][purgedcv.audit_splitter] for a fold report. The assertion
helpers below raise the documented exceptions when a constraint is violated.

## Diagnostics

::: purgedcv.audit_splitter

::: purgedcv.diagnostics.compute_overlap_fraction

::: purgedcv.diagnostics.assert_no_temporal_leakage

::: purgedcv.diagnostics.assert_groups_disjoint

::: purgedcv.diagnostics.assert_embargo_respected

## Exceptions

::: purgedcv.TemporalCVError

::: purgedcv.TemporalLeakageError

::: purgedcv.EmbargoViolationError

::: purgedcv.GroupLeakageError
