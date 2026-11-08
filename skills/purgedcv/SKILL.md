---
name: purgedcv
description: Leakage-aware cross-validation for time-series and financial machine learning with the purgedcv Python package. Use when the user validates or tunes a model on time-ordered data with forward-looking labels (returns over the next N bars, events within a horizon, forecasts), asks about purged k-fold, embargo, walk-forward, combinatorial purged CV (CPCV), López de Prado, deflated or probabilistic Sharpe ratio, probability of backtest overfitting (PBO), minimum backtest length, or why a backtest or CV score looks too good to be true.
license: MIT
---

# purgedcv

`pip install purgedcv` (or `conda install -c conda-forge purgedcv`). Python 3.11+.
Depends only on numpy, pandas, scipy and scikit-learn. All splitters follow the
scikit-learn splitter protocol, so they go straight into `cross_val_score`,
`GridSearchCV`, `cross_validate` and `Pipeline`.

## When to use it

Use it when a label is computed over a future window, so neighbouring rows
share part of their outcome. Shuffled `KFold` then puts overlapping labels on
both sides of the split and inflates the score.

`TimeSeriesSplit(gap=h)` is enough for a single series whose label always ends
`h` rows later. Use purgedcv when the horizon varies per row (barrier or event
labels), when several rows share a timestamp (panels), when you want k-fold or
CPCV with purge and embargo, or when you need the overfitting statistics. If
labels resolve immediately and rows are independent, plain scikit-learn is
fine; say so instead of adding the dependency.

## Two inputs every splitter needs

- `prediction_times`: when the prediction is made (row timestamp). Must be
  sorted, non-decreasing.
- `evaluation_times`: when the label is known (end of the label window).

Build `evaluation_times` by shifting within each entity, not by calendar
arithmetic, so holidays and missing bars are handled. `shift()` follows row
order, not dates, so sort each entity chronologically first:

```text
df = df.sort_values(["asset", "date"])
df["label_end"] = df.groupby("asset")["date"].shift(-h)
df = df.dropna(subset=["label_end"]).sort_values(["date", "asset"]).reset_index(drop=True)
```

Compute the label values (future prices, returns) the same way, before the final
sort. Keep `X`, `y` and both time columns in the same row order.

## Pick a splitter

| Situation | Splitter |
|---|---|
| Replay deployment: train on the past, test the next block | `WalkForwardSplit(n_splits, test_size, window="expanding" or "sliding", ...)` |
| Short history, every period tested once | `PurgedKFold(n_splits, ...)` |
| Panel: many rows per timestamp (assets, stores, sensors) | `PurgedGroupKFold(n_splits, ..., groups=df["date"])` |
| Spread of backtest paths, not one | `CombinatorialPurgedCV(n_splits, n_test_groups, ...)` with `.backtest_paths(...)` |
| After a search over many configs | `probability_of_backtest_overfitting`, `deflated_sharpe_ratio`, `minimum_backtest_length` |

All splitters take `prediction_times=` and `evaluation_times=`. Training rows
whose label interval overlaps the test labels are always purged; that needs no
extra argument. Optional extras:

- `purge_horizon="2D"`: symmetric padding added around the test label window.
  `None` (default) means no padding, not "no purge". Do not set it to the label
  horizon again; `evaluation_times` already carries the horizon.
- One embargo unit for serial dependence after each test block:
  `embargo="5D"` (duration), `embargo_observations=10` (rows) or
  `embargo_fraction=0.01`. Never pass more than one.

## Minimal example

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from purgedcv import PurgedKFold, audit_splitter

n, h = 1000, 5
rng = np.random.default_rng(0)
X = rng.standard_normal((n, 4))
y = rng.standard_normal(n)
pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=h)

cv = PurgedKFold(
    n_splits=5,
    prediction_times=pred,
    evaluation_times=evalu,  # overlap with these label windows is purged
    embargo=f"{h}D",
)
scores = cross_val_score(GradientBoostingRegressor(random_state=0), X, y, cv=cv)
report = audit_splitter(cv, X)  # per-fold rows removed, overlap, empty folds
```

## Panel data

Build labels per entity, sort the panel by date, and group by the date so each
test fold holds whole dates:

```python
from purgedcv import PurgedGroupKFold

dates = pd.bdate_range("2023-01-02", periods=250)
panel = pd.DataFrame([(d, a) for d in dates for a in ["AAA", "BBB", "CCC"]], columns=["date", "asset"])
panel = panel.sort_values(["asset", "date"])
panel["label_end"] = panel.groupby("asset")["date"].shift(-5)
panel = panel.dropna().sort_values(["date", "asset"]).reset_index(drop=True)

panel_cv = PurgedGroupKFold(
    n_splits=5,
    prediction_times=panel["date"],
    evaluation_times=panel["label_end"],
    groups=panel["date"],
)
assert audit_splitter(panel_cv, np.zeros((len(panel), 1)))["train_nonempty"].all()
```

Do not set `groups` to the asset on a date-sorted panel: every test fold then
spans all dates and purge leaves zero training rows. No error is raised;
`audit_splitter` shows `train_nonempty == False`.

## After a model search

The best of many trials is inflated even under honest CV. PBO and DSR answer
different questions, so report the one that matches the claim:

- PBO: does the in-sample ranking carry over out of sample? About 0.5 means
  picking the winner was no better than picking any candidate. It does not test
  whether the candidates have an edge; a set where all share the same real edge
  also gives about 0.5.
- DSR: is the winner's Sharpe ratio above zero after correcting for the number
  of trials?
- MinBTL: how many years of history a given Sharpe needs after `n_trials`.

Inputs and outputs:

- `probability_of_backtest_overfitting(returns, n_splits=16)` takes
  `(n_configs, n_obs)`. A `(days, variants)` matrix must be transposed first.
  `n_splits` must be even and at most `n_obs`; it returns a `PBOResult`, read
  `.pbo`.
- `deflated_sharpe_ratio(returns, n_trials, var_sharpe)` takes the winner's
  per-period returns and the variance of the per-period (not annualised)
  Sharpe ratios across trials. It returns a float probability.
- `minimum_backtest_length(n_trials, target_sharpe)` returns years as a float,
  for an annualised `target_sharpe`.
- Trials from a sequential sampler (Optuna TPE, CMA-ES) are correlated. Pass
  `effective_n_trials(trial_sharpes_in_run_order)` as `n_trials` to DSR. For an
  unordered grid, keep the raw count, which is conservative.

```python
from purgedcv import deflated_sharpe_ratio, minimum_backtest_length, probability_of_backtest_overfitting

trial_returns = rng.normal(0.0, 0.01, size=(50, 960))  # (n_configs, n_obs) per-period returns
pbo = probability_of_backtest_overfitting(trial_returns, n_splits=8).pbo  # selection stability, not edge
sharpes = trial_returns.mean(axis=1) / trial_returns.std(axis=1, ddof=1)
best = trial_returns[int(np.argmax(sharpes))]
dsr = deflated_sharpe_ratio(best, n_trials=len(trial_returns), var_sharpe=float(sharpes.var(ddof=1)))
years_needed = minimum_backtest_length(n_trials=50, target_sharpe=1.0)
```

## Rules for code you write

- Never use shuffled `KFold` on overlapping labels, not even as the "main"
  score. It is fine as an explicitly labelled naive baseline.
- Put scalers, imputers, encoders and feature selection in a `Pipeline` so they
  are refitted inside each fold.
- Build the splitter after filtering rows; use `cv.with_times(pred, evalu)` to
  reuse settings with new times. A length mismatch raises `ValueError`.
- For a final holdout, drop training rows whose `evaluation_times` reach the
  holdout start.
- Call `audit_splitter(cv, X)` before a long search.
- Report measured numbers. If honest CV shows no edge, say that.
- Methods are due to López de Prado (2018) and Bailey and López de Prado. Cite
  those for the methods and DOI 10.5281/zenodo.20312695 for the software.

## Errors and what they mean

- `prediction_times must be monotonic non-decreasing.`: sort rows by time.
- `evaluation_times falls before prediction_times at index i`: the label was
  shifted backwards (`shift(5)` instead of `shift(-5)`).
- `X length N does not match bound prediction_times length M.`: rows were
  filtered after the splitter was built.
- `embargo, embargo_observations, and embargo_fraction are mutually exclusive`:
  pass one embargo unit.
- `TemporalLeakageError`, `EmbargoViolationError`, `GroupLeakageError`: raised
  by `purgedcv.diagnostics.assert_*` helpers when a custom split leaks.

## More

- Choosing a splitter: https://eslazarev.github.io/purged-cross-validation/choosing-a-splitter/
- Common mistakes: https://eslazarev.github.io/purged-cross-validation/common-mistakes/
- Quickstart: https://eslazarev.github.io/purged-cross-validation/quickstart/
- Full docs as one text file: https://eslazarev.github.io/purged-cross-validation/llms-full.txt
