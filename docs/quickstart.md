# Quickstart

Start with the row-level primitives, then use a splitter with sklearn.
Section 3 builds CPCV prediction paths and shows a separate
deflated-Sharpe example. Section 4 checks a whole search: paths from
your own fitting loop, CSCV and PBO, and the minimum backtest length. Each numbered section runs after a fresh
`pip install purgedcv`. Run it from top to bottom: the smaller follow-on
snippets reuse variables from that section.

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
    purge_horizon=pd.Timedelta("5D"),
)
train_kept = apply_embargo(
    train_kept, test_idx,
    prediction_times=pred,
    evaluation_times=evalu,
    embargo="2D",
)
print(len(train_idx), "->", len(train_kept), "after purge + embargo")
```

The standalone `purge` primitive takes a `pd.Timedelta` for its padding.
Splitter constructors also accept strings such as `"5D"`.

### Choosing an embargo unit

Every splitter, `apply_embargo`, PBO, and `assert_embargo_respected` accepts
the same three mutually exclusive modes:

```python
# Wall-clock duration
apply_embargo(train_idx, test_idx, pred, evalu, embargo="2D")

# Fixed number of rows after each contiguous test block
apply_embargo(train_idx, test_idx, pred, evalu, embargo_observations=10)

# floor(n_samples * 0.01) rows after each contiguous test block
apply_embargo(train_idx, test_idx, pred, evalu, embargo_fraction=0.01)
```

Use a duration when elapsed time carries the dependency. Use rows or a
fraction for bar-based data where serial dependence is naturally expressed in
observations. CPCV applies the positional window separately after every
non-adjacent test block. Supplying more than one mode raises `ValueError`.

`WalkForwardSplit` accepts the same parameters for constructor consistency,
but its training candidates are always strictly before the test fold, so an
asymmetric post-test embargo has no rows to remove. Its effective boundary
control is `purge_horizon`.

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
scores = cross_val_score(GradientBoostingRegressor(random_state=0), features, labels, cv=cv)
print("honest R^2 per fold:", scores)
```

`PurgedGroupKFold` (entity-level holdout) and `WalkForwardSplit`
(chronological train-on-the-past, expanding or rolling) follow the same
constructor pattern.

### Audit what each fold removed

Inspect the splitter first. Before a large model search, call
`audit_splitter`. It uses the same internal fold pipeline as `split()` and
reports each filtering stage rather than trying to reconstruct it from the
final indices.

```python
from purgedcv import audit_splitter

report = audit_splitter(cv, features)
print(report[[
    "fold",
    "candidate_train_size",
    "final_train_size",
    "train_nonempty",
    "rows_removed_by_purge",
    "rows_removed_by_embargo",
    "rows_removed_by_finalization",
    "rows_added_by_finalization",
    "candidate_overlap_fraction",
    "final_overlap_fraction",
    "train_block_count",
    "test_block_count",
]])
```

`candidate_overlap_fraction` describes the unfiltered candidate train set
using the splitter's configured `purge_horizon`. For a non-empty candidate set
it equals `rows_removed_by_purge / candidate_train_size`; it is defined as zero
when the candidate set is empty. Indices returned by `cv.split()` are already
final, so passing them to `compute_overlap_fraction` reproduces
`final_overlap_fraction`, not the candidate value. The standalone function can
reproduce the candidate value only when you already own the pre-purge indices
of a custom split and pass the same purge horizon.

For built-in splitters, `final_overlap_fraction` should be zero and
`temporal_leakage_free` true. On an empty train set that truth is vacuous, so
check `train_nonempty` before treating a fold as usable. A sliding
`WalkForwardSplit` performs its window truncation during finalization, hence
the count appears under `rows_removed_by_finalization`. Built-in splitters
never add rows there; a positive `rows_added_by_finalization` flags a custom
finalizer that introduced or swapped indices.

The four time-envelope columns are outer bounds, not claims that the rows form
one continuous window. CPCV folds can have several disjoint blocks whose train
and test envelopes overlap while the actual horizons remain clean; use
`train_block_count`, `test_block_count`, and `final_overlap_fraction` together.
Subclasses that override `split()` cannot be audited and raise a clear error;
customize the documented internal hooks instead. Malformed inputs and final
indices also fail validation normally.

### Sample weights

The splitters carry no scorer of their own; they stay drop-in to
scikit-learn, so sample weights travel through sklearn's own metadata
routing rather than through any `purgedcv` argument. Enable routing once,
then have the estimator request the weight for `fit`. The split itself is
unaffected: weights ride along with the rows each fold keeps.

```python
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from purgedcv import PurgedKFold

n, h = 300, 5
rng = np.random.default_rng(0)
features = rng.standard_normal((n, 3))
labels = rng.standard_normal(n)
weights = rng.uniform(0.5, 1.5, n)
pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=h)
cv = PurgedKFold(n_splits=5, prediction_times=pred, evaluation_times=evalu,
                 purge_horizon=f"{h}D")

sklearn.set_config(enable_metadata_routing=True)

# Train-time weighting only. The estimator requests the weight for fit and
# explicitly declines it for score, so cross_val_score knows where it goes.
est = Ridge().set_fit_request(sample_weight=True).set_score_request(sample_weight=False)
scores = cross_val_score(est, features, labels, cv=cv, params={"sample_weight": weights})

# Weight the score as well: route the weight into fit and into a scorer that
# also requests it. Without this, the default scorer leaves the test fold
# unweighted.
est = Ridge().set_fit_request(sample_weight=True)
scorer = make_scorer(mean_squared_error, greater_is_better=False).set_score_request(
    sample_weight=True
)
scores_w = cross_val_score(est, features, labels, cv=cv, scoring=scorer,
                           params={"sample_weight": weights})
```

The gotcha is the second case: if you route `sample_weight` but leave the
default scorer, sklearn raises `UnsetMetadataPassedError` because the scorer
neither requested nor declined the weight. Decline it with
`set_score_request(sample_weight=False)` for train-only weighting, or pass a
scorer that requests it for weighted evaluation.

## 3. CPCV + backtest paths + deflated Sharpe

CPCV fits on each purged fold and assembles out-of-sample predictions
into backtest paths. With six groups and two test groups per fold, there
are 15 folds and five paths. The example below uses random regression
targets, not market returns.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from purgedcv import CombinatorialPurgedCV

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
print("Prediction paths:", paths.shape)  # (5, 800)
```

These are predictions. They are not strategy returns, and five CPCV paths
are not five independent model trials. A trading application must define
how predictions become positions and then account for realised returns
and costs before computing a Sharpe ratio.

### Deflated Sharpe needs a return series

Here is a separate example. Simulate 20 independent candidate return
series, then select the candidate with the highest observed Sharpe.
DSR takes that candidate's full return series, not its scalar Sharpe.
The correction uses all 20 trials and the variance of their per-observation
Sharpes, including the candidates that lost the selection.

```python
import numpy as np
from purgedcv import deflated_sharpe_ratio

rng = np.random.default_rng(42)
n_trials, n_obs = 20, 252
trial_returns = rng.normal(0.0, 0.01, size=(n_trials, n_obs))
trial_sharpes = trial_returns.mean(axis=1) / trial_returns.std(axis=1, ddof=1)
best_trial = int(np.argmax(trial_sharpes))
selected_returns = trial_returns[best_trial]
var_sharpe = float(trial_sharpes.var(ddof=1))
dsr = deflated_sharpe_ratio(
    selected_returns,
    n_trials=n_trials,
    var_sharpe=var_sharpe,
)
print("Deflated Sharpe probability:", dsr)
```

The simulated trials are independent by construction. Real model searches
often produce correlated candidates; the trial count then needs more care.
See the [Optuna cookbook](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/optuna_dsr_cookbook.py)
for recording a search and estimating an effective count. Keep Sharpe units
consistent: this example does not annualise them.

For a worked time-series regression example, see
[`examples/energy_demand_pjm.ipynb`](examples.md).

## 4. Checking a search: own fit loop, CSCV, PBO and MinBTL

`backtest_paths` fits a scikit-learn estimator for you. When you fit each
fold yourself, for example with a custom training loop or a model outside
scikit-learn, collect one prediction array per fold in `split` order and
pass the list to `reconstruct_paths`. The result has the same layout as
`backtest_paths`.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from purgedcv import CombinatorialPurgedCV

n = 600
rng = np.random.default_rng(1)
features = rng.standard_normal((n, 4))
labels = features @ np.array([0.4, -0.2, 0.0, 0.1]) + rng.standard_normal(n)
pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=2)

cpcv = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    prediction_times=pred,
    evaluation_times=evalu,
    purge_horizon="2D",
)

fold_predictions = []
for train_idx, test_idx in cpcv.split(features):
    model = Ridge().fit(features[train_idx], labels[train_idx])
    fold_predictions.append(model.predict(features[test_idx]))

manual_paths = cpcv.reconstruct_paths(fold_predictions)
print("Paths from the manual loop:", manual_paths.shape)  # (5, 600)
```

The module-level `purgedcv.reconstruct_paths` does the same job when you
only have the fold predictions and fold test indices, without the
splitter object.

### CSCV and the probability of backtest overfitting

`CombinatoriallySymmetricCV` is CPCV with half of the blocks held out in
every split. `probability_of_backtest_overfitting` runs it over a matrix
of candidate return series. In each split it picks the candidate with the
best in-sample Sharpe and checks where that candidate ranks out of sample.
PBO is the share of splits where it lands below the median. In the example
none of the 50 candidates has an edge, so PBO should sit near 0.5.

```python
import numpy as np
import pandas as pd
from purgedcv import CombinatoriallySymmetricCV, probability_of_backtest_overfitting

rng = np.random.default_rng(7)
n_configs, n_obs = 50, 960
candidate_returns = rng.normal(0.0, 0.01, size=(n_configs, n_obs))
days = pd.Series(pd.date_range("2022-01-03", periods=n_obs, freq="B"))

cscv = CombinatoriallySymmetricCV(
    n_splits=8, prediction_times=days, evaluation_times=days
)
in_sample, out_of_sample = next(cscv.split(candidate_returns.T))
print("CSCV splits:", cscv.get_n_splits())  # C(8, 4) = 70
print("Rows per half:", len(in_sample), len(out_of_sample))  # 480 480

result = probability_of_backtest_overfitting(candidate_returns, n_splits=8)
print(f"PBO {result.pbo:.2f} over {result.n_combos} splits")  # PBO 0.57 over 70 splits
```

### Minimum backtest length

The best of 50 noise candidates still shows a positive annualised Sharpe.
`minimum_backtest_length` gives the number of years a backtest needs
before an annualised Sharpe of a given size is more than the best of that
many lucky tries. This sample has fewer years than that, so even a Sharpe
of 1 would not be evidence of skill.

```python
from purgedcv import minimum_backtest_length

annual_sharpes = (
    candidate_returns.mean(axis=1) / candidate_returns.std(axis=1, ddof=1) * np.sqrt(252)
)
years_of_data = n_obs / 252
min_years = minimum_backtest_length(n_configs, target_sharpe=1.0)
print(f"Best annualised Sharpe: {annual_sharpes.max():.2f}")  # 0.85
print(f"{years_of_data:.1f} years of data; MinBTL {min_years:.1f} years")  # 3.8 vs 5.2
```

[`examples/backtest_overfitting_audit.ipynb`](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/backtest_overfitting_audit.ipynb) runs the same
checks on a real Optuna search over BTC/USDT strategies.
