---
title: "purgedcv: scikit-learn-compatible purged and combinatorial cross-validation for time-series and financial machine learning"
tags:
  - Python
  - machine learning
  - cross-validation
  - time series
  - financial machine learning
  - backtesting
  - scikit-learn
authors:
  - name: Evgenii Lazarev
    orcid: "0009-0000-1398-7842"
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 20 May 2026
bibliography: paper.bib
---

# Summary

`purgedcv` is a Python library that brings leakage-aware cross-validation to
ordinary scikit-learn workflows. It implements purging and embargoing of
overlapping labels, expanding and rolling walk-forward validation, purged and
group-purged k-fold, Combinatorial Purged Cross-Validation (CPCV) with
backtest-path reconstruction, and the Probabilistic Sharpe Ratio, Deflated
Sharpe Ratio, and Minimum Track Record Length. Every splitter follows the
scikit-learn splitter protocol, so it drops into `cross_val_score`,
`GridSearchCV`, and `Pipeline` without glue code. The package is typed,
ships `py.typed`, passes `mypy --strict`, and is pinned by 354 tests
(unit, property-based, doctest, and end-to-end) at 98% line coverage,
reproducible with `pytest --collect-only -q`.

The algorithms are not new. They are Marcos López de Prado's, from
*Advances in Financial Machine Learning* [@lopezdeprado2018afml], with the
statistical metrics from Bailey and López de Prado [@bailey2012psr;
@bailey2014dsr]. What `purgedcv` adds is an open, MIT-licensed, maintained,
sklearn-native implementation checked against the original papers, plus a
gallery of worked examples on real public data that shows both the dramatic
failures and the undramatic ones.

# Statement of need

k-fold cross-validation assumes the rows are independent. Time-series labels
are not. A label such as "return over the next twenty bars" overlaps the
labels of its neighbours, so a shuffled split puts near-duplicates of a test
point into training and the answer leaks. There is a second, quieter leak
just after the test window, where serial correlation pulls test-period
structure back into training. Both make a model look better than it is. This
is a recognised failure mode in machine learning generally [@kaufman2012] and
a documented cause of irreproducible results in applied science
[@mcdermott2021].

The demand for the fix is on the record. A 2022 auto-sklearn feature request
asking for purging, embargoing, and CPCV, citing the same book, has stayed
open for more than three years. A 2024 mlpack request for a `PurgedKFoldCV`
class was closed as "not planned." scikit-learn recommends purging in its own
documentation, yet ships only a single fixed `gap` on `TimeSeriesSplit`: no
label-overlap purging, no embargo as a fraction of the data, no group
awareness, no combinatorial paths.

The space is not empty, and that is the problem. mlfinlab, the canonical
implementation, was relicensed as a paid closed-source product and cannot be
a dependency for an open project. timeseriescv, the main free combinatorial
implementation, has not been released since 2018 and has known correctness
issues. RiskLabAI is research-grade reference code rather than a typed,
tested, `pip install`-able drop-in. In practice that has left three poor
options: pay for a closed product, vendor an abandoned one, or copy the
algorithms out of a textbook by hand. `purgedcv` aims at that gap.

# State of the field

Purging, embargoing, and CPCV come from chapters 7 and 12 of
@lopezdeprado2018afml. The companion statistics that separate genuine skill
from selection bias are the Probabilistic and Deflated Sharpe Ratio
[@bailey2012psr; @bailey2014dsr] and the Probability of Backtest Overfitting
[@bailey2017pbo], set against the wider multiple-testing problem in empirical
finance [@harvey2016]. The topic is current, not a 2018 curiosity. A 2024
study in *Knowledge-Based Systems* compares CPCV against k-fold, purged
k-fold, and walk-forward and finds it ahead on both backtest-overfitting
probability and Deflated Sharpe Ratio [@backtestoverfitting2024], and recent
preprints continue to report leakage from overlapping-label cross-validation
outside finance [@hiddenleaks2025]. scikit-learn [@pedregosa2011sklearn]
defines the splitter protocol that `purgedcv` targets so the methods reach
the practitioners who need them.

# Software design

The public API maps one to one onto the textbook constructs. `purge` and
`apply_embargo` are the row-level primitives. `WalkForwardSplit` does
expanding or rolling chronological validation. `PurgedKFold` and
`PurgedGroupKFold` purge by label horizon, the latter holding whole groups
out for entity-level questions. `CombinatorialPurgedCV` with
`reconstruct_paths` builds the multiple backtest paths of chapter 12.
`probabilistic_sharpe_ratio`, `deflated_sharpe_ratio`, and
`min_track_record_length` score the result and are numerically exact to the
worked example in §7.4.1 of @lopezdeprado2018afml. A `diagnostics` module
(`compute_overlap_fraction`, `assert_no_temporal_leakage`,
`assert_groups_disjoint`, `assert_embargo_respected`) turns "trust me" into
an assertion you can put in a test.

Splitters accept timestamps and label horizons rather than a single integer
gap, so the purge tracks the real label, not a guess. Everything is strictly
typed and the suite includes property-based tests that check invariants such
as "no training row's label window overlaps any test label."

The implementation deliberately separates interval arithmetic from the
scikit-learn adapters. Internal helpers normalize prediction, evaluation,
purge, and embargo windows once; the splitters then compose those primitives
without duplicating boundary logic. This keeps the simple cases small while
allowing variable label horizons, rolling or expanding windows, and
entity-level group holds without replacing the user's estimator workflow.

# Research impact statement

The repository ships a controlled proof and ten notebooks on real public
data. The point is to show the honest range of outcomes, not only the
alarming ones.

The controlled proof builds a target that nothing can predict, then runs
naive shuffled k-fold next to `PurgedKFold`. Naive scores R² between 0.83 and
0.91 on noise. `PurgedKFold` removes the label overlap, the score collapses
below a predict-the-mean baseline, and `compute_overlap_fraction` confirms
the train/test overlap drops from 100% to 0%. It is deterministic and needs
no download.

Three real datasets reproduce the same failure. On Binance BTC/USDT daily
bars with a 20-bar forward return, naive shuffled k-fold reports R² of about
+0.85 for a feature with no economic reason to forecast returns, and
`PurgedKFold` takes it to about −1.2. On the USGS catalogue, earthquake
magnitude is unpredictable from past magnitudes by the Gutenberg–Richter law,
and the empirical autocorrelation here is +0.02. Naive shuffled k-fold still
prints R² = +0.65; purged (−0.75), blocked (−1.13), and walk-forward (−1.24)
all return the correct verdict of no skill. The UCI air-quality notebook
isolates the mechanism. Forecasting mean benzene over the next 72 hours is a
genuinely solvable task. With three ordinary lag features naive R² is a
modest 0.07. Add one innocuous cumulative-hour counter and naive R² leaps to
0.99, while `PurgedKFold` (−1.52) and `WalkForwardSplit` (−0.81) do not
move. One monotone feature plus overlapping labels plus a shuffled split is
the whole trick.

The undramatic cases matter as much. On the full Low Carbon London
smart-meter population [@ukpn_lcl], 4,284 eligible households measured
offline with confidence intervals by `tools/lcl_full_benchmark.py`, the
temporal-leakage gap between naive shuffled k-fold and walk-forward is small
(1.60%, 95% CI 1.27–1.94%). The leak that actually bites is by household:
scoring on unseen customers is 6.03% worse than the pooled temporal estimate
(95% CI 4.93–7.12%). The lesson is that which split you need depends on
what you deploy on, and that an honest pipeline can also report "small gap
here." In a model-selection notebook on the same BTC data, once the Deflated
Sharpe Ratio corrects for having tried several models, no model clears
DSR ≥ 0.95. Reporting no edge is the correct outcome, and the package makes
it easy to report.

The remaining notebooks exercise the rest of the API on their natural
domains: `PurgedGroupKFold` on PhysioNet ICU mortality
[@physionet2012goldberger; @physionet2012silva], `WalkForwardSplit` on NASA
C-MAPSS turbofan remaining-useful-life [@nasa_cmapss], `PurgedKFold` on NOAA
GHCN-Daily rainfall [@noaa_ghcnd], and the full CPCV plus PSR/DSR/MinTRL
workflow on PJM electricity load [@pjm_load]. A Premier League match
prediction notebook adds a low-signal sports example where the honest result
is calibration drift rather than a headline accuracy gap.

# AI usage disclosure

Generative AI tools, including OpenAI Codex and ChatGPT in the GPT-5
family and Anthropic Claude in the Claude 4 family, were used as
assistants for code review, refactoring suggestions, test scaffolding,
documentation drafting, copy-editing, and pre-submission checks. The
author made the design decisions, reviewed and edited all AI-assisted
changes, and validated the outputs with the unit, property, doctest,
end-to-end, type-checking, linting, notebook-execution, and benchmark
checks described above. No AI-generated claim was accepted without
source or executable verification.

# Acknowledgements

This work uses open datasets from UK Power Networks and the London Datastore
[@ukpn_lcl], the U.S. Geological Survey [@usgs_catalog], the UCI Machine
Learning Repository [@devito2008airquality], NOAA NCEI [@noaa_ghcnd], the
NASA Prognostics Center of Excellence [@nasa_cmapss], PhysioNet
[@physionet2012goldberger; @physionet2012silva], PJM Interconnection
[@pjm_load], and Binance market data via the `pricehub` package
[@binance_pricehub]. The methods implemented here are due to López de Prado,
Bailey, and colleagues; any errors in the implementation are mine.

# References
