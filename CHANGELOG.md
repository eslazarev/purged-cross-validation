# Changelog

All notable changes to `purged-cross-validation` are recorded here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project follows pre-release semantic versioning until v1.0.0.

## [0.3.0a0] - 2026-05-13

### Added (Plan C — Domain D6: CPCV backtest path reconstruction)

- `reconstruct_paths(fold_predictions, fold_test_indices, n_splits, n_test_groups, n_samples)` — pure function that combines the C(N,K) fold outputs into the N-K canonical backtest paths described in Lopez de Prado (2018) chapter 12.
- `CombinatorialPurgedCV.backtest_paths(estimator, X, y)` — convenience method that runs the full CPCV fit-predict loop and calls `reconstruct_paths`, returning an `(n_paths, n_samples)` float array with `NaN` for unseen observations.

### Added (Plan D — Domain D7: statistical metrics)

- `probabilistic_sharpe_ratio(returns, benchmark_skill)` — probability that the true Sharpe ratio exceeds a benchmark, corrected for non-normality (Bailey & Lopez de Prado 2012).
- `deflated_sharpe_ratio(returns, n_trials, var_sharpe)` — PSR adjusted for multiple-comparison bias across independent strategy evaluations (Bailey & Lopez de Prado 2014).
- `min_track_record_length(observed_sharpe, target_sharpe, alpha, skew, kurtosis)` — minimum sample size required for PSR to exceed `1 - alpha` at the observed Sharpe; the analytical inverse of PSR.

## [0.2.0a0] - 2026-05-12

### Added (Plan B — Domains D4 + D5: splitter framework)

- `BaseTemporalSplitter` (D4) — abstract base class wiring `purge` + `apply_embargo` into every fold produced by concrete subclasses; enforces group-disjointness when `groups` are supplied.
- `WalkForwardSplit` (D5.1) — sliding-window and expanding-window walk-forward CV; configurable `train_size`, `test_size`, and `step`; `with_times` adapter for fluent construction.
- `PurgedKFold` (D5.2) — contiguous test folds tiling the index space, with purge and embargo applied; degrades to standard `KFold(shuffle=False)` at zero purge/embargo.
- `PurgedGroupKFold` (D5.3) — group-aware variant of `PurgedKFold` that assigns whole groups to folds so no entity leaks across the train/test boundary.
- `CombinatorialPurgedCV` (D5.4) — exhaustive C(N, K) combinatorial fold enumeration for producing multiple backtest paths from a single dataset.
- sklearn integration: all splitters satisfy the `sklearn.model_selection` splitter protocol and work inside `cross_val_score`, `GridSearchCV`, and `Pipeline`.

## [0.1.0a0] - 2026-05-12

### Added (Plan A — Foundations: Domains D1 + D2 + D3 + D8)

- `parse_horizon`, `horizons_overlap`, `validate_times` (D1) — time and horizon utilities; strict validation of monotonicity, NaN-freedom, and chronological ordering.
- `purge` (D2) — drops training rows whose half-open label horizon `[prediction_time, evaluation_time)` overlaps the test horizon; implements AFML Section 7.4.1 Snippet 7.1.
- `apply_embargo` (D3) — drops training rows whose `prediction_time` falls in the post-test asymmetric embargo window `[test_eval_max, test_eval_max + embargo]`.
- `purgedcv.diagnostics` submodule (D8) — `assert_no_temporal_leakage`, `assert_embargo_respected`, `assert_groups_disjoint`, `compute_overlap_fraction` for auditing custom splits.
- Exception hierarchy — `TemporalCVError`, `TemporalLeakageError`, `EmbargoViolationError`, `GroupLeakageError`.
- src-layout Python package using hatchling as the build backend.
- TDD test suite with hypothesis property tests; ruff + mypy strict + pre-commit quality gates.

[0.3.0a0]: https://github.com/eslazarev/purged-cross-validation/releases/tag/v0.3.0a0
[0.2.0a0]: https://github.com/eslazarev/purged-cross-validation/releases/tag/v0.2.0a0
[0.1.0a0]: https://github.com/eslazarev/purged-cross-validation/releases/tag/v0.1.0a0
