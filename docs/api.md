# API reference

Choose a topic for signatures, parameters, and examples rendered from the source
docstrings. New to the package? Start with the [Quickstart](quickstart.md).

| Topic | What you will find |
| --- | --- |
| [Splitters](api/splitters.md) | Shared splitter parameters and concrete CV classes |
| [Purge and embargo](api/primitives.md) | Row-level training-index filters |
| [Backtest paths](api/paths.md) | CPCV path reconstruction and return summaries |
| [Statistical metrics](api/metrics.md) | PSR, DSR, track-record lengths, and trial counts |
| [Backtest overfitting and Optuna](api/overfitting.md) | PBO results and optional search callbacks |
| [Diagnostics and exceptions](api/diagnostics.md) | Fold audits, leakage assertions, and exceptions |
| [Time inputs and utilities](api/time.md) | Input aliases, horizon parsing, and validation |

## Symbol index

Old `/api/#purgedcv...` links open the matching topic when JavaScript is enabled.
Without JavaScript, follow the symbol links below. The runtime package version
is available as `purgedcv.__version__`.

## Input type aliases

- [`TimesLike`](api/time.md#purgedcv.TimesLike){ .api-legacy-link }
- [`ArrayLike1D`](api/time.md#purgedcv.ArrayLike1D){ .api-legacy-link }
- [`HorizonLike`](api/time.md#purgedcv.HorizonLike){ .api-legacy-link }

## Splitters

- [`BaseTemporalSplitter`](api/splitters.md#purgedcv.BaseTemporalSplitter){ .api-legacy-link }
- [`WalkForwardSplit`](api/splitters.md#purgedcv.WalkForwardSplit){ .api-legacy-link }
- [`PurgedKFold`](api/splitters.md#purgedcv.PurgedKFold){ .api-legacy-link }
- [`PurgedGroupKFold`](api/splitters.md#purgedcv.PurgedGroupKFold){ .api-legacy-link }
- [`CombinatorialPurgedCV`](api/splitters.md#purgedcv.CombinatorialPurgedCV){ .api-legacy-link }
- [`CombinatoriallySymmetricCV`](api/splitters.md#purgedcv.CombinatoriallySymmetricCV){ .api-legacy-link }

## Backtest paths

- [`PathMetricFn`](api/paths.md#purgedcv.PathMetricFn){ .api-legacy-link }
- [`reconstruct_paths`](api/paths.md#purgedcv.reconstruct_paths){ .api-legacy-link }
- [`path_metrics`](api/paths.md#purgedcv.path_metrics){ .api-legacy-link }
- [`default_backtest_metrics`](api/paths.md#purgedcv.default_backtest_metrics){ .api-legacy-link }

## Row-level primitives

- [`purge`](api/primitives.md#purgedcv.purge){ .api-legacy-link }
- [`apply_embargo`](api/primitives.md#purgedcv.apply_embargo){ .api-legacy-link }

## Time and horizon utilities

- [`parse_horizon`](api/time.md#purgedcv.parse_horizon){ .api-legacy-link }
- [`horizons_overlap`](api/time.md#purgedcv.horizons_overlap){ .api-legacy-link }
- [`validate_times`](api/time.md#purgedcv.validate_times){ .api-legacy-link }

## Statistical metrics

- [`DSRDiagnostics`](api/metrics.md#purgedcv.DSRDiagnostics){ .api-legacy-link }
- [`probabilistic_sharpe_ratio`](api/metrics.md#purgedcv.probabilistic_sharpe_ratio){ .api-legacy-link }
- [`deflated_sharpe_ratio`](api/metrics.md#purgedcv.deflated_sharpe_ratio){ .api-legacy-link }
- [`deflated_sharpe_ratio_full`](api/metrics.md#purgedcv.deflated_sharpe_ratio_full){ .api-legacy-link }
- [`min_track_record_length`](api/metrics.md#purgedcv.min_track_record_length){ .api-legacy-link }
- [`minimum_backtest_length`](api/metrics.md#purgedcv.minimum_backtest_length){ .api-legacy-link }
- [`effective_n_trials`](api/metrics.md#purgedcv.effective_n_trials){ .api-legacy-link }

## Backtest overfitting

- [`PBOResult`](api/overfitting.md#purgedcv.PBOResult){ .api-legacy-link }
- [`PerformanceMetric`](api/overfitting.md#purgedcv.PerformanceMetric){ .api-legacy-link }
- [`probability_of_backtest_overfitting`](api/overfitting.md#purgedcv.probability_of_backtest_overfitting){ .api-legacy-link }

## Optuna integration

- [`optuna_integration.TrialSharpeRecorder`](api/overfitting.md#purgedcv.optuna_integration.TrialSharpeRecorder){ .api-legacy-link }

## Diagnostics

- [`audit_splitter`](api/diagnostics.md#purgedcv.audit_splitter){ .api-legacy-link }
- [`diagnostics.compute_overlap_fraction`](api/diagnostics.md#purgedcv.diagnostics.compute_overlap_fraction){ .api-legacy-link }
- [`diagnostics.assert_no_temporal_leakage`](api/diagnostics.md#purgedcv.diagnostics.assert_no_temporal_leakage){ .api-legacy-link }
- [`diagnostics.assert_groups_disjoint`](api/diagnostics.md#purgedcv.diagnostics.assert_groups_disjoint){ .api-legacy-link }
- [`diagnostics.assert_embargo_respected`](api/diagnostics.md#purgedcv.diagnostics.assert_embargo_respected){ .api-legacy-link }

## Exceptions

- [`TemporalCVError`](api/diagnostics.md#purgedcv.TemporalCVError){ .api-legacy-link }
- [`TemporalLeakageError`](api/diagnostics.md#purgedcv.TemporalLeakageError){ .api-legacy-link }
- [`EmbargoViolationError`](api/diagnostics.md#purgedcv.EmbargoViolationError){ .api-legacy-link }
- [`GroupLeakageError`](api/diagnostics.md#purgedcv.GroupLeakageError){ .api-legacy-link }
