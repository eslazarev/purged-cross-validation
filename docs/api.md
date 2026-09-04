# API reference

All public symbols from `purgedcv.__all__`, auto-rendered from the source
docstrings. The constructors of the splitters share a single set of
keyword arguments (`prediction_times`, `evaluation_times`,
`purge_horizon`, the three embargo modes, and `groups`); see
[`BaseTemporalSplitter`](#purgedcv.BaseTemporalSplitter) for the shared
contract.

## Input type aliases

::: purgedcv.TimesLike

::: purgedcv.ArrayLike1D

::: purgedcv.HorizonLike

## Splitters

::: purgedcv.BaseTemporalSplitter

::: purgedcv.WalkForwardSplit

::: purgedcv.PurgedKFold

::: purgedcv.PurgedGroupKFold

::: purgedcv.CombinatorialPurgedCV

::: purgedcv.CombinatoriallySymmetricCV

## Backtest paths

::: purgedcv.PathMetricFn

::: purgedcv.reconstruct_paths

::: purgedcv.path_metrics

::: purgedcv.default_backtest_metrics

## Row-level primitives

::: purgedcv.purge

::: purgedcv.apply_embargo

## Time and horizon utilities

::: purgedcv.parse_horizon

::: purgedcv.horizons_overlap

::: purgedcv.validate_times

## Statistical metrics

::: purgedcv.DSRDiagnostics

::: purgedcv.probabilistic_sharpe_ratio

::: purgedcv.deflated_sharpe_ratio

::: purgedcv.deflated_sharpe_ratio_full

::: purgedcv.min_track_record_length

::: purgedcv.minimum_backtest_length

::: purgedcv.effective_n_trials

## Backtest overfitting

::: purgedcv.PBOResult

::: purgedcv.PerformanceMetric

::: purgedcv.probability_of_backtest_overfitting

## Optuna integration

::: purgedcv.optuna_integration.TrialSharpeRecorder

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
