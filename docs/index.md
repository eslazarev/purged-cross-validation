# purgedcv

Cross-validation, the textbook kind, assumes the rows are exchangeable.
For a lot of real data they are not. Time-series labels overlap with
their neighbours, panel data has repeated customers, and a backtest
needs more than a single train-then-test cut to be honest about its own
overfitting. `purgedcv` is the scikit-learn-compatible answer to that.

It ships purging and embargoing of overlapping labels, expanding and
rolling walk-forward validation, purged and group-purged k-fold,
Combinatorial Purged Cross-Validation with backtest-path reconstruction,
and the Probabilistic, Deflated, and Minimum-Track-Record Sharpe
statistics. Every splitter speaks the standard sklearn splitter protocol.
It drops straight into `cross_val_score`, `GridSearchCV`, and `Pipeline`.

The algorithms are not new. They are Marcos López de Prado's (2018), with
the statistical metrics from Bailey and López de Prado (2012, 2014). This
library is an open, MIT-licensed, typed implementation, checked against
the original papers and pinned by 354 tests (98% line coverage).

## Where to start

- [Installation](installation.md) — `pip install purgedcv` or `conda install -c conda-forge purgedcv`, optional extras.
- [Quickstart](quickstart.md) — three short runnable snippets.
- [API reference](api.md) — autodoc for every public symbol.
- [Examples](examples.md) — eleven worked notebooks on real public data.
- [Methodology](methodology.md) — the underlying problem and the prior-art gap.
- [Paper (JOSS)](https://github.com/eslazarev/purged-cross-validation/blob/main/paper/paper.md) — the software paper.

## What the library is for

The point is not to push accuracy up. It is to stop naive shuffled
cross-validation from quietly raising a model's *reported* accuracy by
leaking the answer through overlapping labels or by quietly remembering
the customer. Done correctly, the honest score is usually lower than the
naive one. That is the whole point.

A controlled example: on a target built from pure noise (nothing can
predict it), default `KFold(shuffle=True)` reports R² = +0.92 with 100 %
train/test label overlap. `PurgedKFold` returns the correct verdict — no
skill, zero overlap. Same model, same data, different split.

A measured example: on the full UK Low Carbon London smart-meter
population (4,284 households), the temporal leak between naive shuffled
k-fold and walk-forward is small. 1.60 % in relative WAPE terms, 95 % CI
1.27 to 1.94. The leak that actually bites is by household: scoring on
unseen customers is 6.03 % worse than the pooled temporal estimate
(95 % CI 4.93 to 7.12). Which cross-validation scheme you need follows
from what you intend to deploy on, and a pipeline that cannot also say
"small gap here" is not a measurement.

## Cite this software

See [`CITATION.cff`](https://github.com/eslazarev/purged-cross-validation/blob/main/CITATION.cff)
or the [JOSS paper](https://github.com/eslazarev/purged-cross-validation/blob/main/paper/paper.md).
