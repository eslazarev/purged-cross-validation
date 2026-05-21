# Changelog

All notable changes to `purged-cross-validation` are recorded here. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The 0.0.x series is the pre-1.0 development line. The library's feature
work was organised internally as Plans A–D (foundations → splitters →
backtest paths → metrics); the cumulative feature set delivered by each
Plan is listed under the published version it shipped in.

## [Unreleased]

The next published release will be **v0.0.5**.

### Added

- JOSS submission paper at `docs/paper.md` (`docs/paper.bib`) with the
  real ORCID baked in.
- Hosted documentation site at
  <https://eslazarev.github.io/purged-cross-validation/> using
  MkDocs Material + mkdocstrings, deployed by
  `.github/workflows/docs.yml`. The PR check in `ci.yml` runs
  `mkdocs build --strict`.
- Community files required for JOSS verification: `CITATION.cff`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1),
  `.zenodo.json`, issue and pull-request templates.
- Full-population UK Low Carbon London benchmark
  (`tools/lcl_full_benchmark.py`) — chunked enumeration over the raw
  ~8 GB corpus, K seeded subsamples of N households, mean ±95% t-interval.
  Real result (K=20, N=60, seed=0, 4,284 eligible households): temporal
  leak 1.60% (95% CI 1.27 – 1.94%), household leak 6.03% (95% CI
  4.93 – 7.12%). End-to-end test `test_e2e_lcl_full_benchmark.py`.
- Controlled competitor benchmark (`tools/competitor_benchmark.py`) and
  the empirical write-up at `paper/competitor_benchmark.md` — on the
  controlled task `purgedcv` admits 0.000 label overlap while default
  `KFold(shuffle=True)` fabricates R² = +0.92; mlfinpy is shown to be
  broken on pandas ≥ 2.0; RiskLabAI does not install on a modern stack.
  End-to-end test `test_e2e_competitor_benchmark.py`.
- `context7.json` so the docs are indexable by Context7.


### Changed

- `BaseTemporalSplitter` now requires monotonic `prediction_times` at
  construction (`require_monotonic=True`). Closes a silent
  train-from-future failure mode in `WalkForwardSplit`.
- `purge`, `apply_embargo`, and the diagnostics no longer collapse
  separated test blocks into one global interval. The new
  `src/purgedcv/_intervals.py` provides `overlaps_any_half_open_interval`
  and `points_in_any_closed_interval`; both filters operate on the union
  of local per-row intervals. Fixes CPCV folds with non-adjacent test
  groups and the artificial-NaN paths (`energy_demand_pjm` notebook now
  reports 0.0% NaN paths instead of 100%).
- Embargo is now applied per test row (`[eval_i, eval_i + embargo]`
  unioned across the fold) rather than only after `max(eval)`. For
  contiguous PurgedKFold this is strictly more conservative; for CPCV
  with non-adjacent groups it is the correct generalisation.

### Fixed

- `deflated_sharpe_ratio` returned `1.0` for every input when
  `n_trials=1`. The single-trial branch set the deflated benchmark SR\*
  to `-inf`; it now uses SR\* = 0, so DSR reduces to
  `probabilistic_sharpe_ratio(returns, 0.0)`. A losing strategy is no
  longer reported as certain skill. Covered by a regression test.
- Version desync between `pyproject.toml` and
  `src/purgedcv/__init__.py`. The release workflow now bumps both files
  (alpha-aware), and the new install-smoke test
  `test_packaging_metadata_versions_match_runtime` fails CI if they
  drift again.

## [0.0.4] — 2026-05-17

By v0.0.4 the cumulative feature set of Plans A through D below was
delivered.

### Plan D — Domain D7: statistical metrics

- `probabilistic_sharpe_ratio(returns, benchmark_skill)` — probability that the true Sharpe ratio exceeds a benchmark, corrected for non-normality (Bailey & López de Prado 2012).
- `deflated_sharpe_ratio(returns, n_trials, var_sharpe)` — PSR adjusted for multiple-comparison bias across independent strategy evaluations (Bailey & López de Prado 2014).
- `min_track_record_length(observed_sharpe, target_sharpe, alpha, skew, kurtosis)` — minimum sample size required for PSR to exceed `1 − alpha` at the observed Sharpe; the analytical inverse of PSR.

### Plan C — Domain D6: CPCV backtest path reconstruction

- `reconstruct_paths(fold_predictions, fold_test_indices, n_splits, n_test_groups, n_samples)` — pure function that combines the C(N,K) fold outputs into the C(N−1,K−1) canonical backtest paths described in López de Prado (2018) chapter 12.
- `CombinatorialPurgedCV.backtest_paths(estimator, X, y)` — convenience method that runs the full CPCV fit-predict loop and calls `reconstruct_paths`, returning an `(n_paths, n_samples)` float array with `NaN` for unseen observations.

### Plan B — Domains D4 + D5: splitter framework

- `BaseTemporalSplitter` (D4) — abstract base class wiring `purge` + `apply_embargo` into every fold produced by concrete subclasses; enforces group disjointness when `groups` are supplied.
- `WalkForwardSplit` (D5.1) — sliding-window and expanding-window walk-forward CV; configurable `train_size`, `test_size`, and `step`; `with_times` adapter for fluent construction.
- `PurgedKFold` (D5.2) — contiguous test folds tiling the index space, with purge and embargo applied; degrades to standard `KFold(shuffle=False)` at zero purge/embargo.
- `PurgedGroupKFold` (D5.3) — group-aware variant of `PurgedKFold` that assigns whole groups to folds so no entity leaks across the train/test boundary.
- `CombinatorialPurgedCV` (D5.4) — exhaustive C(N, K) combinatorial fold enumeration for producing multiple backtest paths from a single dataset.
- scikit-learn integration: every splitter satisfies the `sklearn.model_selection` splitter protocol and works inside `cross_val_score`, `GridSearchCV`, and `Pipeline`.

### Plan A — Foundations: Domains D1 + D2 + D3 + D8

- `parse_horizon`, `horizons_overlap`, `validate_times` (D1) — time and horizon utilities; strict validation of monotonicity, NaN-freedom, and chronological ordering.
- `purge` (D2) — drops training rows whose half-open label horizon `[prediction_time, evaluation_time)` overlaps the test horizon; implements AFML Section 7.4.1 Snippet 7.1.
- `apply_embargo` (D3) — drops training rows whose `prediction_time` falls in the post-test asymmetric embargo window `[test_eval_max, test_eval_max + embargo]`.
- `purgedcv.diagnostics` submodule (D8) — `assert_no_temporal_leakage`, `assert_embargo_respected`, `assert_groups_disjoint`, `compute_overlap_fraction` for auditing custom splits.
- Exception hierarchy — `TemporalCVError`, `TemporalLeakageError`, `EmbargoViolationError`, `GroupLeakageError`.
- src-layout Python package using hatchling as the build backend.
- TDD test suite with hypothesis property tests; ruff + mypy strict + pre-commit quality gates.

### Examples (added across 0.0.x development)

- Worked-example notebook gallery on real public data: PhysioNet ICU
  mortality, NASA C-MAPSS turbofan RUL, NOAA GHCN-Daily rainfall, PJM
  hourly load, Binance BTC/USDT, UK Low Carbon London smart meters, USGS
  earthquakes, UCI air quality, Premier League matches, and a controlled
  synthetic leakage proof.

## [0.0.3] — 2026-05-16

Development patch release.

## [0.0.2] — 2026-05-16

Development patch release.

## [0.0.1] — 2026-05-16

First PyPI release.

[Unreleased]: https://github.com/eslazarev/purged-cross-validation/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/eslazarev/purged-cross-validation/releases/tag/v0.0.4
[0.0.3]: https://github.com/eslazarev/purged-cross-validation/releases/tag/v0.0.3
[0.0.2]: https://github.com/eslazarev/purged-cross-validation/releases/tag/v0.0.2
[0.0.1]: https://github.com/eslazarev/purged-cross-validation/releases/tag/v0.0.1
