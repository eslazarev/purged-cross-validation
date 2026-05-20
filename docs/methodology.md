# Why this exists

Most cross-validation code assumes the rows don't depend on each other. For
time-series and financial data that assumption is simply wrong, and the usual
fixes are either behind a paywall, abandoned, or missing from the libraries
people actually use. That mismatch is what this package is for.

## The underlying problem

k-fold cross-validation treats observations as independent. Time-series labels
are not. If a label is something like "return over the next five days," the
label for row *t* overlaps the labels for the rows around it. Shuffle those rows
into separate folds and the test answer leaks straight into training. There is a
second, quieter leak on the other side of the test window: the rows that come
just after it are serially correlated with it, so training on them pulls
test-period structure backwards in time.

Both leaks make a model look better than it is. The standard treatment comes
from Marcos López de Prado's *Advances in Financial Machine Learning* (Wiley,
2018) — purging and embargoing in chapter 7, Combinatorial Purged
Cross-Validation in chapter 12. The statistical companion, which separates real
skill from selection bias, is Bailey and López de Prado's work on the
Probabilistic and Deflated Sharpe Ratio (2012, 2014).

None of this is obscure. scikit-learn's own documentation tells you to do it:
*"it is a good idea to remove the training observations close to the validation
set to increase the independence between training and validation."* What
scikit-learn actually ships is a single `gap` parameter on `TimeSeriesSplit`:
one fixed gap, no label-aware purging, no embargo as a fraction of the data, no
group handling, no combinatorial paths.

## The demand is on the record

This isn't a need I'm guessing at. People have opened issues on major ML
libraries asking for exactly these primitives, and citing the same book this
package implements.

| Where | Who / when | What was asked for | Status (as of 2026-05) |
|---|---|---|---|
| **auto-sklearn** [#1589](https://github.com/automl/auto-sklearn/issues/1589) | cryptocoinserver, Sep 2022 | Purging + embargoing + Combinatorial Purged CV, citing López de Prado AFML | **Open, never implemented** (3+ years) |
| **mlpack** [#3830](https://github.com/mlpack/mlpack/issues/3830) | Patschkowski, Nov 2024 | A `PurgedKFoldCV` class, citing AFML pp. 105–109 | **Closed as "not planned"** |
| **scikit-learn** | — | Docs recommend purging; only `gap` (one fixed gap, no label-overlap, no embargo %, no CPCV, no group purge) exists | Not on the roadmap |

Two requests, two well-known libraries, both pointing at the same literature.
One has sat open for more than three years. The other was closed as "not
planned." scikit-learn recommends the technique in its own docs and doesn't
implement it.

## What already exists, and why it isn't enough

The space isn't empty, which is reassuring — an empty space usually means nobody
wants the thing. The trouble is that none of the existing options is something
you can actually depend on in an open project.

| Project | What it is | Why it does not close the gap |
|---|---|---|
| **mlfinlab** (Hudson & Thames) | The canonical purged / CPCV implementation, based directly on López de Prado | **Closed-source, paid subscription.** Originally open source, then relicensed commercial. Community members have since started open forks (e.g. `mlfinpy`) specifically because access was withdrawn. Cannot be a dependency for an open project. |
| **timeseriescv** (sam31415) | The main free `CombPurgedKFoldCV` implementation | **Unmaintained since 2018** (last release v0.2). Known correctness issues — third-party write-ups note the combinatorial class *"required some repairs"*; open bug reports from 2022 are unanswered. Not safe to depend on. |
| **RiskLabAI** | Research-lab purged k-fold (Python + Julia) | Research-grade code, not a clean `pip install`, sklearn-native, typed, tested drop-in. Useful as a reference, not as a production dependency. |
| **scikit-learn `TimeSeriesSplit(gap=...)`** | A single fixed gap before each test fold | Not real purging (ignores actual label horizons), no embargo, no combinatorial paths, no group awareness, no PSR/DSR. Addresses a fraction of the problem. |

In practice that leaves three poor choices: pay for a closed product, vendor an
abandoned package with known bugs, or copy the algorithms out of the textbook by
hand.

## Where this package sits

`purged-cross-validation` aims at the empty column in the table below: open,
MIT-licensed, maintained, sklearn-native, strictly typed, and checked against
the original papers rather than reimplemented from memory.

| Requirement | mlfinlab | timeseriescv | sklearn `gap` | **purged-cross-validation** |
|---|---|---|---|---|
| Open source, permissive (MIT) | ✗ (paid) | ✓ | ✓ | ✓ |
| Actively maintained | n/a (commercial) | ✗ (2018) | ✓ | ✓ |
| Label-overlap purging | ✓ | ✓ | ✗ | ✓ |
| Percentage / duration embargo | ✓ | ✓ | ✗ | ✓ |
| Group-aware purged k-fold | partial | ✗ | ✗ | ✓ |
| Combinatorial Purged CV + path reconstruction | ✓ | partial (buggy) | ✗ | ✓ |
| PSR / DSR / Min Track Record Length | ✓ | ✗ | ✗ | ✓ (exact to Bailey & López de Prado) |
| sklearn splitter protocol (`GridSearchCV`, `Pipeline`) | partial | partial | ✓ | ✓ |
| Strict typing + property + e2e tests + doctests | unknown | ✗ | ✓ | ✓ (285 tests, mypy strict) |
| Real-data worked examples | docs | minimal | ✓ | ✓ (10 real-data notebooks) |

The algorithms here aren't new. They're López de Prado's, implemented
faithfully. What is different is that they're available, maintained, and pinned
by 285 tests, instead of paywalled, abandoned, or sitting in a book.

## This is still a live topic

These methods aren't a 2018 curiosity. A 2024 study in *Knowledge-Based Systems*
(Elsevier), "Backtest overfitting in the machine learning era," compares CPCV
against k-fold, purged k-fold, and walk-forward, and finds it ahead on both
Probability of Backtest Overfitting and Deflated Sharpe Ratio. Tutorials and
write-ups keep appearing — QuantInsti, Towards AI, the Wikipedia article on
purged cross-validation — and each one sends more people looking for an
implementation they can install.

## Sources

- [auto-sklearn #1589 — purging/embargoing feature request](https://github.com/automl/auto-sklearn/issues/1589)
- [mlpack #3830 — Purged K-Fold request, closed "not planned"](https://github.com/mlpack/mlpack/issues/3830)
- [scikit-learn — common pitfalls (purging recommendation)](https://scikit-learn.org/stable/common_pitfalls.html)
- [scikit-learn — `TimeSeriesSplit` (only a `gap` parameter)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [mlfinlab — closed-source / commercial pivot](https://hudsonthames.org/mlfinlab/)
- [mlfinlab #496 — community asking for a permissive license](https://github.com/hudson-and-thames/mlfinlab/issues/496)
- [mlfinpy — community open-source successor](https://pypi.org/project/mlfinpy/)
- [timeseriescv on PyPI — last release v0.2 (2018), unmaintained](https://pypi.org/project/timeseriescv/)
- [RiskLab AI — cross-validation in finance](https://www.risklab.ai/research/financial-modeling/cross_validation)
- [Backtest overfitting in the ML era — Knowledge-Based Systems, 2024 (Elsevier)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
- [Purged cross-validation — Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)

The scikit-learn documentation, López de Prado (2018), and Bailey & López de
Prado (2012, 2014) are also listed under Methodology references in the
[README](https://github.com/eslazarev/purged-cross-validation#readme).
