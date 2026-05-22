# Purged cross validation

**scikit-learn-compatible cross-validation for time-series machine learning: purging, embargoes, and combinatorial backtest paths.**

[![CI](https://github.com/eslazarev/purged-cross-validation/actions/workflows/ci.yml/badge.svg)](https://github.com/eslazarev/purged-cross-validation/actions/workflows/ci.yml)
![Coverage](https://raw.githubusercontent.com/eslazarev/purged-cross-validation/refs/heads/main/.github/badges/coverage.svg)
[![PyPI version](https://img.shields.io/pypi/v/purgedcv)](https://pypi.org/project/purgedcv/)
[![PyPI downloads](https://static.pepy.tech/badge/purgedcv)](https://pepy.tech/project/purgedcv)
[![PyPI wheel](https://img.shields.io/pypi/wheel/purgedcv)](https://pypi.org/project/purgedcv/#files)

[![Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Development status: alpha](https://img.shields.io/badge/status-alpha-orange)](https://pypi.org/project/purgedcv/)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-526CFE?logo=materialformkdocs)](https://eslazarev.github.io/purged-cross-validation/)
[![DOI](https://zenodo.org/badge/1239914629.svg)](https://doi.org/10.5281/zenodo.20312695)

**[Documentation →](https://eslazarev.github.io/purged-cross-validation/)** · **[Example notebooks →](examples/)** — purge/embargo, walk-forward, and CPCV with PSR/DSR worked end to end on real ICU-mortality, turbofan-RUL, rainfall, and electricity-demand data.

*Cite this software:* see [`CITATION.cff`](CITATION.cff) and [`paper/paper.md`](paper/paper.md) (JOSS paper).

---

## The problem

Standard k-fold cross-validation assumes the rows are independent. Time-series data is not. When a label resolves over the next few days, it overlaps the labels sitting right next to it, so an ordinary shuffle-split leaks tomorrow's answer back into training. The rows immediately after a test window leak too, because they are serially correlated with it. Both effects quietly inflate backtested Sharpe ratios and hand you strategies that look great on a chart and bleed money once they go live. This library removes both.

Why write another one? People have asked scikit-learn, auto-sklearn, and mlpack for purging and embargo support and been turned down or left waiting for years. The one mature implementation, mlfinlab, went closed-source and paid. The free alternative has been unmaintained since 2018. That gap is the reason this exists.

---

## Does it actually catch leakage?

A controlled check on synthetic data whose target is built so that **no feature can predict it**. The honest out-of-sample score must never be positive. Naive shuffled k-fold runs against `PurgedKFold` side by side ([examples/synthetic_leakage_proof.ipynb](examples/synthetic_leakage_proof.ipynb), deterministic, no download):

| model | naive shuffled KFold R² | PurgedKFold R² |
|---|--:|--:|
| predict-the-mean (reference) | -0.01 | -0.13 |
| k-NN | **0.83** | -1.31 |
| RandomForest | **0.91** | -1.94 |

Train/test label overlap: **100% under naive → 0% under PurgedKFold**.

![Out-of-sample R² on an unpredictable target: naive shuffled KFold scores far above zero (fabricated), PurgedKFold collapses below it.](https://raw.githubusercontent.com/eslazarev/purged-cross-validation/main/.github/images/synthetic_leakage_proof.png)

Naive CV reports R² ≈ 0.83–0.91 on a target nothing can predict. That is pure leakage from the overlap. `PurgedKFold` removes the overlap and the fabricated skill collapses below a predict-the-mean baseline. The negative number is not the point; *no positive skill* is the correct answer, and only the purged split reports it. The library does not make models look better; it stops them looking better than they are.

---

## Installation

```bash

pip install purgedcv

# Directly from the repository
pip install git+https://github.com/eslazarev/purged-cross-validation.git
```

---

## Quickstart

### 1. The core primitive: `purge`

`purge` removes training observations that share data with the test set. Here a model uses a 5-day sliding feature window to predict the next day, so every observation occupies a 5-day span and the spans of neighbours overlap. Any training observation whose window reaches into the test period has already seen test data and must be dropped.

```python
import numpy as np
import pandas as pd
from purgedcv import purge

WINDOW = 5  # feature look-back in days

# 16 days of data; each observation uses a 5-day window to predict the next day
days        = pd.date_range("2024-01-01", periods=16, freq="D")
predict_day = np.arange(WINDOW + 1, len(days) + 1)                    # 11 observations
pred        = pd.Series([days[d - WINDOW - 1] for d in predict_day])  # first feature day
evalu       = pd.Series([days[d - 1] for d in predict_day])           # label day

train_idx = np.arange(0, 7)    # observations predicting days 6..12
test_idx  = np.arange(7, 11)   # observations predicting days 13..16

# Drop training observations whose 5-day feature window overlaps the test window
kept_idx   = purge(train_idx, test_idx, pred, evalu)
purged_idx = np.setdiff1d(train_idx, kept_idx)

print(f"Kept:   {kept_idx.tolist()}")    # [0, 1, 2]    -> predict days 6, 7, 8
print(f"Purged: {purged_idx.tolist()}")  # [3, 4, 5, 6] -> predict days 9, 10, 11, 12
```

Each bar below is one observation's 5-day feature window. The four red bars cross into the test window (dashed line) — their features overlap the test period, so `purge` drops them. The three green bars stay fully before it; `→ day 8` only touches the boundary and is kept, because label horizons are half-open.

![Purge on a 5-day sliding window: training observations whose feature window overlaps the test window are dropped, leaving a clean gap before the test block.](https://raw.githubusercontent.com/eslazarev/purged-cross-validation/main/.github/images/purge_example.png)

---

### 2. Splitters with scikit-learn: `PurgedKFold` inside `cross_val_score`

Drop-in replacement for `KFold` that applies purge and embargo automatically on every fold.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from purgedcv import PurgedKFold

rng   = np.random.default_rng(0)
n     = 200
pred  = pd.Series(pd.date_range("2022-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=3)
X     = rng.standard_normal((n, 5))
y     = X @ rng.standard_normal(5) + rng.standard_normal(n) * 0.5

cv = PurgedKFold(
    n_splits=5,
    prediction_times=pred,
    evaluation_times=evalu,
    purge_horizon="3D",   # matches label horizon
    embargo="1D",         # 1-day post-test buffer
)

scores = cross_val_score(Ridge(), X, y, cv=cv, scoring="r2")
print(f"R² per fold: {scores.round(3)}")
```

All four splitters (`WalkForwardSplit`, `PurgedKFold`, `PurgedGroupKFold`, `CombinatorialPurgedCV`) satisfy the sklearn splitter protocol and work inside `GridSearchCV` and `Pipeline`.

---

### 3. CPCV + path reconstruction + metrics: the full workflow

Combinatorial Purged CV produces C(N, K) folds that tile into multiple out-of-sample backtest paths. Use PSR and DSR to evaluate them with corrections for non-normality and selection bias.

```python
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from purgedcv import (
    CombinatorialPurgedCV,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    min_track_record_length,
)

rng   = np.random.default_rng(42)
n     = 120
pred  = pd.Series(pd.date_range("2023-01-01", periods=n, freq="D"))
evalu = pred + pd.Timedelta(days=2)
X     = rng.standard_normal((n, 3))
y     = X @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(n) * 0.1

# N=6, K=2  →  C(6,2) = 15 folds  →  C(5,1) = 5 backtest paths
cv = CombinatorialPurgedCV(
    n_splits=6,
    n_test_groups=2,
    prediction_times=pred,
    evaluation_times=evalu,
)

# paths.shape == (n_paths, n_samples); NaN only if a fold could not be fit
paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
print(f"Backtest paths: {paths.shape}")  # (5, 120)

# Derive a toy "return" series and compute per-path PSR
per_path_returns = paths - y[np.newaxis, :]
per_path_psr = [
    probabilistic_sharpe_ratio(row[np.isfinite(row)], benchmark_skill=0.0)
    for row in per_path_returns
]
print(f"PSR per path: {[round(p, 3) for p in per_path_psr]}")

# DSR corrects for testing 5 paths simultaneously
first = per_path_returns[0]
dsr = deflated_sharpe_ratio(first[np.isfinite(first)], n_trials=5, var_sharpe=0.01**2)
print(f"Deflated SR (first path): {dsr:.3f}")

# Minimum observations needed to prove SR=0.7 beats benchmark SR=0.5 at 95% confidence
n_min = min_track_record_length(
    observed_sharpe=0.7, target_sharpe=0.5, alpha=0.05, skew=0.0, kurtosis=3.0
)
print(f"MinTRL: {int(n_min)} observations")
```

---

## API summary

| Symbol | Domain | Description |
|---|---|---|
| `purge` | D2 | Remove overlapping-horizon training rows |
| `apply_embargo` | D3 | Remove post-test buffer rows |
| `WalkForwardSplit` | D5.1 | Sliding / expanding walk-forward CV |
| `PurgedKFold` | D5.2 | Contiguous test folds with purge + embargo |
| `PurgedGroupKFold` | D5.3 | Group-aware purged k-fold |
| `CombinatorialPurgedCV` | D5.4 | C(N,K) combinatorial folds |
| `reconstruct_paths` | D6 | Assemble CPCV folds into backtest paths |
| `probabilistic_sharpe_ratio` | D7 | PSR: P(true SR > benchmark) |
| `deflated_sharpe_ratio` | D7 | DSR: PSR corrected for multiple testing |
| `min_track_record_length` | D7 | Minimum observations to establish SR |
| `diagnostics.*` | D8 | Leakage and embargo audit functions |

---


## Methodology references

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 7 (purge/embargo) and 12 (CPCV).
- Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. *Journal of Risk*, 15(2).
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality. *Journal of Portfolio Management*, 40(5).

---

## License

MIT. See [LICENSE](LICENSE).
