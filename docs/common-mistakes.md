# Common mistakes

Most leakage bugs in time-series validation are one of a handful of
mistakes. Where purgedcv can catch one, the page shows the message it
raises. Some it cannot catch, and those are listed too. All snippets share
this setup:

```python
import numpy as np
import pandas as pd
from purgedcv import PurgedGroupKFold, PurgedKFold, audit_splitter

n = 200
pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=5)
X = np.random.default_rng(0).standard_normal((n, 3))
```

## Label end times from calendar arithmetic

`evaluation_times` must be the time the label is actually known. For a
label that ends five trading rows later, `date + 5 business days` is
wrong around holidays: starting on 28 March 2024, it gives 4 April, while
the fifth London trading day after it is 8 April. Shift within each
entity instead. `shift()` follows row order, not dates, so sort each
entity's rows by time before shifting, then drop incomplete rows and sort
the whole frame by time:

```python
# London trading days: Good Friday (29 March) and Easter Monday (1 April) have no rows.
days = pd.bdate_range("2024-03-25", "2024-04-12").drop(pd.to_datetime(["2024-03-29", "2024-04-01"]))
prices = pd.DataFrame({"date": list(days) * 2, "asset": ["AAA"] * len(days) + ["BBB"] * len(days)})

prices = prices.sort_values(["asset", "date"])
prices["label_end"] = prices.groupby("asset")["date"].shift(-5)
prices["calendar_end"] = prices["date"] + pd.offsets.BDay(5)  # the wrong way
prices = prices.dropna().sort_values(["date", "asset"]).reset_index(drop=True)
print(prices.loc[prices["date"] == "2024-03-28", ["date", "label_end", "calendar_end"]].iloc[0])
# label_end is 2024-04-08, calendar_end is 2024-04-04
```

## Rows not sorted by time

Splitters cut contiguous blocks, so `prediction_times` must be
non-decreasing. A frame sorted by asset, then date, fails:

```python
order = np.random.default_rng(1).permutation(n)
try:
    PurgedKFold(
        n_splits=5,
        prediction_times=pred.iloc[order].reset_index(drop=True),
        evaluation_times=evalu.iloc[order].reset_index(drop=True),
    )
except ValueError as err:
    print(err)  # prediction_times must be monotonic non-decreasing.
```

Sort by time first, and keep `X`, `y` and both time columns in that same
order.

## The label shifted the wrong way

`shift(5)` looks five rows back. The resulting end times fall before the
prediction times:

```python
try:
    PurgedKFold(n_splits=5, prediction_times=pred, evaluation_times=pred - pd.Timedelta(days=1))
except ValueError as err:
    print(err)  # evaluation_times falls before prediction_times at index 0 ...
```

Use `shift(-5)` for a forward label.

## Grouping a date-sorted panel by asset

Nothing warns you about this one. In a panel sorted by date, every asset spans the
whole period. A test fold made of whole assets therefore covers every
date, and purge removes all training rows:

```python
days = pd.bdate_range("2024-01-01", periods=100)
panel = pd.DataFrame([(d, a) for d in days for a in "ABCDE"], columns=["date", "asset"])
panel["label_end"] = panel.groupby("asset")["date"].shift(-5)
panel = panel.dropna().reset_index(drop=True)
X_panel = np.zeros((len(panel), 1))

by_asset = PurgedGroupKFold(
    n_splits=5,
    prediction_times=panel["date"],
    evaluation_times=panel["label_end"],
    groups=panel["asset"],
)
print(audit_splitter(by_asset, X_panel)[["fold", "final_train_size", "train_nonempty"]])
```

Every fold has `final_train_size` 0, and `cross_val_score` fails with
"All the 5 fits failed". Group by the date instead. Run `audit_splitter`
before a long search: it costs nothing and catches empty folds early.

## Preprocessing fitted on all rows

A scaler, imputer, or target encoder fitted before CV has seen the test
rows. purgedcv cannot see this. Put every fitted step in a `Pipeline` so it is
refitted inside each fold, and treat feature selection the same way.

## Filtering X after building the splitter

The splitter keeps the time columns it was given. Drop rows from `X`
afterwards and the lengths no longer match:

```python
cv = PurgedKFold(n_splits=5, prediction_times=pred, evaluation_times=evalu)
try:
    next(cv.split(X[:150]))
except ValueError as err:
    print(err)  # X length 150 does not match bound prediction_times length 200.
```

Filter first, then build the splitter from the filtered times. To keep the
settings and change the times, use `cv.with_times(...)`.

## More than one embargo unit

`embargo`, `embargo_observations` and `embargo_fraction` are alternatives:

```python
try:
    PurgedKFold(
        n_splits=5, prediction_times=pred, evaluation_times=evalu,
        embargo="2D", embargo_observations=3,
    )
except ValueError as err:
    print(err)  # embargo, embargo_observations, and embargo_fraction are mutually exclusive ...
```

## A holdout whose labels start in training

Cutting a final holdout by date is not enough when labels look forward.
Rows just before the cut have labels that end inside the holdout. Drop
training rows whose `evaluation_times` reach the holdout start.

## Reporting the best score of a search as is

Try 200 configurations with no edge at all and the best one will still
look good.
Report the deflated Sharpe ratio or PBO next to it. See
[Choosing a splitter](choosing-a-splitter.md#after-a-search-pbo-dsr-and-minbtl).
