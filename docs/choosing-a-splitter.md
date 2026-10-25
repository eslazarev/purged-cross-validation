# Choosing a splitter

Pick the splitter by asking how the model will be used once it is
deployed. Every splitter below needs two time columns. `prediction_times`
is when you make the prediction, and `evaluation_times` is when its label
becomes known. If the label is the return over the next five bars, the
evaluation time is five bars later.

| Your situation | Use |
|---|---|
| You retrain on the past and predict the next period, and you want CV to replay that | `WalkForwardSplit` |
| History is short and you want every period tested once | `PurgedKFold` |
| Many rows share a timestamp, such as a panel of assets or one row per store per day | `PurgedGroupKFold` with `groups` set to the date |
| You want a distribution of backtest paths instead of a single one | `CombinatorialPurgedCV` |
| You already ran a search and want to know how likely the winner is overfit | `CombinatoriallySymmetricCV` and `probability_of_backtest_overfitting` |

## When the scikit-learn splitters are enough

Shuffled `KFold` assumes rows are exchangeable. That fails as soon as
labels overlap in time: a five-day forward return computed on Monday
shares four days with Tuesday's, so a shuffled split trains on almost the
same answer it is tested on. Flexible models notice. A deep random forest
can look excellent in shuffled CV and then lose, on the next year of
data, to the simpler model an honest split would have picked. The FTSE and
crypto example notebooks in this repository both show this.

`TimeSeriesSplit` keeps order, and its `gap` parameter drops a fixed
number of rows before each test fold. For a single series whose label
always ends `h` rows later, `gap=h` removes the overlap, and that is a
perfectly good setup. A fixed gap stops being enough in three cases:

- the horizon varies from row to row, for example labels that end when a
  price hits a barrier or an event happens;
- several rows share a timestamp, as in a panel, so `h` dates are not `h`
  rows and a fold boundary can cut through a date;
- you want k-fold with test blocks in the middle of the sample, where the
  rows after each block also need an embargo.

purgedcv works from each row's label interval instead of a row count, so
these cases need no hand-tuned gap.

If your labels resolve immediately and rows are independent, use plain
scikit-learn. purgedcv adds nothing there.

## WalkForwardSplit

The training set is always strictly before the test fold. Use
`window="expanding"` to keep all history, or `window="sliding"` with
`train_size` to keep a fixed amount. This is the closest match to how most
trading and forecasting systems run. The cost is that early folds train
on little data. `test_size` is in rows, so for a panel choose it as a whole
number of dates.

## PurgedKFold

Contiguous test blocks tile the whole sample, so earlier folds train on
later data. Purge removes training rows whose label window overlaps the
test window, and an embargo drops a buffer after each test block. Use it
when history is too short for walk-forward. The price is an assumption:
the relationship should not drift much over the sample.

## PurgedGroupKFold for panels

Splitters require `prediction_times` to be sorted. In a panel the same
date appears once per asset, so sort by date first and use the date as the
group. Each test fold then contains whole dates, and purge still drops
overlapping labels from neighbouring dates.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from purgedcv import PurgedGroupKFold, audit_splitter

rng = np.random.default_rng(0)
dates = pd.bdate_range("2023-01-02", periods=250)
panel = pd.DataFrame(
    [(d, a) for d in dates for a in ["AAA", "BBB", "CCC", "DDD"]],
    columns=["date", "asset"],
)
panel["close"] = 100 * np.exp(rng.normal(0, 0.01, len(panel)).cumsum())
# shift() follows row order, so put each asset's rows in date order first.
panel = panel.sort_values(["asset", "date"])
# The label ends five rows later for the same asset, not five calendar days later.
panel["label_end"] = panel.groupby("asset")["date"].shift(-5)
panel["future_close"] = panel.groupby("asset")["close"].shift(-5)
panel = panel.dropna().sort_values(["date", "asset"]).reset_index(drop=True)

X = rng.standard_normal((len(panel), 3))
y = (panel["future_close"] > panel["close"]).astype(int)

cv = PurgedGroupKFold(
    n_splits=5,
    prediction_times=panel["date"],
    evaluation_times=panel["label_end"],
    groups=panel["date"],
)
model = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
report = audit_splitter(cv, X)
print(report[["fold", "final_train_size", "rows_removed_by_purge"]])
```

Grouping by asset does not work on a date-sorted panel. Every asset spans
the whole period, so each test fold covers every date and purge removes
all training rows. `audit_splitter` shows this as `train_nonempty` equal
to `False`. See [Common mistakes](common-mistakes.md) for the details.

## CombinatorialPurgedCV

CPCV splits the sample into `n_splits` groups and tests on every
combination of `n_test_groups` of them. The fold predictions can be put
back together into several full backtest paths with `backtest_paths` or
`reconstruct_paths`, which gives you a spread of outcomes instead of one
number. It is the right choice when the question is "how much does this
result depend on which period happened to be the test set?".

## After a search: PBO, DSR and MinBTL

Honest folds do not fix selection. Pick the best of many configurations
and its score is inflated even when every fold was clean. Three tools
measure by how much.

- `probability_of_backtest_overfitting` takes a `(n_configs, n_obs)`
  matrix of per-period returns and estimates how often the in-sample best
  falls below the median out of sample. It measures the selection step:
  about 0.5 means the in-sample ranking says nothing about the
  out-of-sample ranking, so picking the winner was no better than picking
  any candidate. It does not say whether the candidates make money. A set
  in which every configuration has the same real edge also scores about
  0.5.
- `deflated_sharpe_ratio` answers the other question: is the winner's
  Sharpe ratio still above zero once you account for the number of trials
  and their spread?
- `minimum_backtest_length` gives the number of years a backtest must
  cover before a given annualised Sharpe beats what `n_trials` random tries
  would produce.

```python
from purgedcv import (
    deflated_sharpe_ratio,
    minimum_backtest_length,
    probability_of_backtest_overfitting,
)

def pbo_and_dsr(returns):
    sharpes = returns.mean(axis=1) / returns.std(axis=1, ddof=1)
    winner = returns[int(np.argmax(sharpes))]
    pbo = probability_of_backtest_overfitting(returns, n_splits=8).pbo
    dsr = deflated_sharpe_ratio(winner, n_trials=len(returns), var_sharpe=float(sharpes.var(ddof=1)))
    return pbo, dsr

# 50 configurations, 960 business days each
noise = np.random.default_rng(7).normal(0.0, 0.01, size=(50, 960))
edge = np.random.default_rng(7).normal(0.002, 0.01, size=(50, 960))
for name, returns in [("no edge", noise), ("same edge in all", edge)]:
    pbo, dsr = pbo_and_dsr(returns)
    print(f"{name}: PBO = {pbo:.2f}, DSR = {dsr:.2f}")
print(f"MinBTL for Sharpe 1.0 after 50 trials: {minimum_backtest_length(50, 1.0):.1f} years")
```

Both sets give a PBO near 0.5, because in both the in-sample winner is
just one of many equivalent candidates. Only the deflated Sharpe ratio
tells them apart.

Section 4 of the [Quickstart](quickstart.md) runs all three on one search.
