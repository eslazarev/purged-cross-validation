# Architecture

`purgedcv` is organised as a thin public API on top of a small kernel of
interval and time utilities. The diagram below shows the dependency
direction: user code calls a splitter (which already speaks the
scikit-learn protocol), the splitter delegates to one shared base class
that wires `purge` and `apply_embargo` into every fold, and those
primitives share a single interval implementation. The statistical
metrics and the diagnostics module sit alongside the splitters as
independent, leakage-aware tools.

The internal grouping `D1`–`D8` refers to the implementation plan in the
project's CHANGELOG.

```mermaid
flowchart TB
    user["User estimator + features + timestamps<br/>(prediction_times, evaluation_times)"]

    subgraph public["purgedcv public API"]
        direction TB
        splitters["Splitters · sklearn protocol<br/>WalkForwardSplit (D5.1)<br/>PurgedKFold (D5.2)<br/>PurgedGroupKFold (D5.3)<br/>CombinatorialPurgedCV (D5.4)"]
        paths["reconstruct_paths /<br/>backtest_paths (D6)"]
        metrics["Statistical metrics (D7)<br/>probabilistic_sharpe_ratio<br/>deflated_sharpe_ratio<br/>min_track_record_length"]
        diagnostics["Diagnostics (D8)<br/>audit_splitter<br/>compute_overlap_fraction<br/>assert_no_temporal_leakage<br/>assert_groups_disjoint<br/>assert_embargo_respected"]
    end

    subgraph internal["Internal kernel"]
        direction TB
        base["BaseTemporalSplitter (D4)<br/>shared splitter contract<br/>monotonic check, group disjointness"]
        primitives["purge (D2) · apply_embargo (D3)<br/>row-level filters"]
        intervals["_intervals<br/>half-open & closed<br/>interval merges (O(n log n))"]
        time["_time (D1)<br/>parse_horizon<br/>validate_times<br/>horizons_overlap"]
    end

    user -->|"cv = SplitterClass(...)"| splitters
    splitters --> base
    base --> primitives
    primitives --> intervals
    base --> time
    splitters -.via cross_val_score / GridSearchCV.-> user

    user --> paths
    paths --> metrics

    user -.audit any custom split.-> diagnostics
    diagnostics -.audit purgedcv fold stages.-> base
    diagnostics --> intervals
    diagnostics --> time
```

## Design rules

Three small invariants keep the surface honest.

1. **One implementation per boundary kind.** Time-based overlap and embargo
   checks go through `src/purgedcv/_intervals.py`; positional embargo windows
   are implemented once in `src/purgedcv/_embargo.py`. Splitters do not
   duplicate either boundary rule. A CPCV fold with non-adjacent test groups
   is filtered by the *union* of local windows rather than by the convex hull
   between them.

2. **Label-aware purge.** Splitters take framework-agnostic
   `prediction_times` and `evaluation_times` containers rather than a single
   integer gap. pandas, NumPy, Python sequences, and Polars are accepted and
   coerced to one-dimensional NumPy arrays at the public boundary.
   The purge follows the real label horizon, so a label that overlaps by
   twelve half-hours and one that overlaps by twenty cannot share a
   single hand-tuned `gap` and still be correct — the user passes the
   actual horizon and the splitter does the right thing per row.

3. **Row-level diagnostics are independent; splitter audits share the real
   pipeline.** `compute_overlap_fraction` and the three `assert_*` functions
   accept positional indices and never assume their input came from
   `purgedcv`, so they can audit sklearn, tscv, or hand-rolled splits equally.
   `audit_splitter` deliberately targets `BaseTemporalSplitter`: it consumes
   the same candidate → purge → embargo → final stages as `split()` so its
   per-stage counts are observed rather than inferred. Subclasses must express
   fold customization through the stage hooks; overriding `split()` is
   rejected because the report could no longer guarantee it audits the folds
   users actually receive.

## Module layout

```
src/purgedcv/
├── __init__.py          public re-exports; __version__
├── _base.py             BaseTemporalSplitter
├── _walk_forward.py     WalkForwardSplit
├── _purged_kfold.py     PurgedKFold, PurgedGroupKFold
├── _cpcv.py             CombinatorialPurgedCV
├── _paths.py            reconstruct_paths
├── _purge.py            purge
├── _embargo.py          apply_embargo
├── _intervals.py        half-open / closed interval merges
├── _time.py             parse_horizon, validate_times, horizons_overlap
├── _metrics.py          probabilistic_sharpe_ratio, deflated_sharpe_ratio,
│                        min_track_record_length
├── _typing.py           NDArrayAny, HorizonLike
├── diagnostics.py       public assertions and per-fold audit report
├── exceptions.py        TemporalCVError tree
└── py.typed             PEP 561 marker — the package is fully typed
```

Wheel contents follow exactly this layout; `tools/` and `examples/`
ride along with the source distribution but are not packaged.
