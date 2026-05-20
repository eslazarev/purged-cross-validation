# Quickstart

Three short snippets that cover the full surface: the row-level
primitives, the sklearn splitter you will use most, and Combinatorial
Purged Cross-Validation with backtest-path reconstruction and the
deflated-Sharpe statistics. All snippets are runnable on a fresh
`pip install purgedcv`.

## 1. Row-level primitives

The two functions `purge` and `apply_embargo` are the building blocks
every splitter ships on top of. They take positional indices, the
prediction and evaluation timestamps for every row, and return the
training indices that survive.

```python
import numpy as np
import pandas as pd
from purgedcv import apply_embargo, purge

n = 1000
pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=5)  # 5-day forward label

train_idx = np.arange(0, 800)
test_idx = np.arange(800, 900)

train_kept = purge(
    train_idx, test_idx,
    prediction_times=pred,
    evaluation_times=evalu,
    purge_horizon="5D",
)
train_kept = apply_embargo(
    train_kept, test_idx,
    prediction_times=pred,
    evaluation_times=evalu,
    embargo="2D",
)
print(len(train_idx), "->", len(train_kept), "after purge + embargo")
```

## 2. PurgedKFold in `cross_val_score`

Every splitter follows the scikit-learn splitter protocol, so it works
inside `cross_val_score`, `GridSearchCV`, and `Pipeline` without glue
code.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from purgedcv import PurgedKFold

n, h = 1000, 5
rng = np.random.default_rng(0)
features = rng.standard_normal((n, 4))
labels = rng.standard_normal(n)
pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=h)

cv = PurgedKFold(
    n_splits=5,
    prediction_times=pred,
    evaluation_times=evalu,
    purge_horizon=f"{h}D",
    embargo=f"{h}D",
)
scores = cross_val_score(GradientBoostingRegressor(), features, labels, cv=cv)
print("honest R^2 per fold:", scores)
```

`PurgedGroupKFold` (entity-level holdout) and `WalkForwardSplit`
(chronological train-on-the-past, expanding or rolling) follow the same
constructor pattern.

## 3. CPCV + backtest paths + deflated Sharpe

The full workflow from chapter 12 of *Advances in Financial Machine
Learning*: enumerate `C(N, K)` purged folds, fit-predict on each, assemble
the per-path out-of-sample predictions with `reconstruct_paths`, and
correct the resulting Sharpe ratios for the number of model trials.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from purgedcv import CombinatorialPurgedCV, deflated_sharpe_ratio

n = 800
rng = np.random.default_rng(0)
features = rng.standard_normal((n, 5))
labels = rng.standard_normal(n)
pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=3)

cpcv = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    prediction_times=pred,
    evaluation_times=evalu,
    purge_horizon="3D",
)
paths = cpcv.backtest_paths(Ridge(), features, labels)
# paths.shape == (n_paths, n_samples); NaN = unseen position

# Treat each per-path mean predicted score as a per-strategy Sharpe-like
# series and correct for the number of model trials we actually ran.
per_path_sharpe = np.nanmean(paths, axis=1) / np.nanstd(paths, axis=1)
dsr = deflated_sharpe_ratio(
    per_path_sharpe.mean(),
    n_trials=len(per_path_sharpe),
    var_sharpe=per_path_sharpe.var(ddof=1),
)
print("Deflated Sharpe (probability skill is real):", dsr)
```

For the full numerical example matching §7.4.1 of the book, see
[`examples/energy_demand_pjm.ipynb`](examples.md).
