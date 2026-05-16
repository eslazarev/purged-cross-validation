"""End-to-end tests verifying purged-cross-validation splitters integrate with sklearn.

These tests are THE proof that the Q1 API decision (constructor-bound
times) works correctly with sklearn's CV entrypoints. If sklearn changes
its API in a way that breaks our splitters, these tests catch it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from purgedcv import (
    BaseTemporalSplitter,
    CombinatorialPurgedCV,
    PurgedKFold,
    WalkForwardSplit,
)
from purgedcv._typing import NDArrayAny


def _make_dataset(
    n: int = 60, horizon_days: int = 1
) -> tuple[NDArrayAny, NDArrayAny, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((n, 4))  # noqa: N806
    y = X @ rng.standard_normal(4) + rng.standard_normal(n) * 0.1
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return X, y, pred, evalu


@pytest.mark.e2e
@pytest.mark.parametrize(
    "make_cv",
    [
        lambda pred, evalu: WalkForwardSplit(
            n_splits=4,
            test_size=5,
            purge_horizon="1D",
            embargo="1D",
            prediction_times=pred,
            evaluation_times=evalu,
        ),
        lambda pred, evalu: PurgedKFold(
            n_splits=4,
            purge_horizon="1D",
            embargo="1D",
            prediction_times=pred,
            evaluation_times=evalu,
        ),
    ],
    ids=["WalkForwardSplit", "PurgedKFold"],
)
def test_cross_val_score_accepts_splitter(
    make_cv: Callable[[pd.Series, pd.Series], BaseTemporalSplitter],
) -> None:
    """WalkForwardSplit and PurgedKFold must produce one finite score per
    fold under cross_val_score (no fold collapse expected)."""
    X, y, pred, evalu = _make_dataset()  # noqa: N806
    cv = make_cv(pred, evalu)
    scores = cross_val_score(
        DummyRegressor(strategy="mean"),
        X=X,
        y=y,
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    assert len(scores) == cv.get_n_splits()
    assert np.all(np.isfinite(scores))


@pytest.mark.e2e
def test_cross_val_score_accepts_cpcv_with_structural_fold_collapse() -> None:
    """CombinatorialPurgedCV is the one splitter that intentionally produces
    a fold with empty train: combo (group 0, group N-1) spans the entire
    timeline, so purge eliminates every training candidate. The remaining
    C(N, K) - 1 folds must still produce finite scores.

    sklearn emits FitFailedWarning for the empty-train fold and assigns NaN
    via error_score=np.nan. We assert exactly one NaN score, and that
    FitFailedWarning was raised."""
    from math import comb

    from sklearn.exceptions import FitFailedWarning

    X, y, pred, evalu = _make_dataset()  # noqa: N806
    cv = CombinatorialPurgedCV(
        n_splits=4,
        n_test_groups=2,
        purge_horizon="1D",
        embargo="1D",
        prediction_times=pred,
        evaluation_times=evalu,
    )
    with pytest.warns(FitFailedWarning):
        scores = cross_val_score(
            DummyRegressor(strategy="mean"),
            X=X,
            y=y,
            cv=cv,
            scoring="neg_mean_squared_error",
            error_score=np.nan,
        )
    assert len(scores) == comb(4, 2) == 6
    n_finite = int(np.sum(np.isfinite(scores)))
    n_nan = int(np.sum(np.isnan(scores)))
    assert n_finite == 5
    assert n_nan == 1


@pytest.mark.e2e
def test_grid_search_cv_accepts_splitter() -> None:
    X, y, pred, evalu = _make_dataset(n=80)  # noqa: N806
    cv = PurgedKFold(
        n_splits=4,
        purge_horizon="1D",
        embargo="1D",
        prediction_times=pred,
        evaluation_times=evalu,
    )
    search = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    search.fit(X, y)
    assert search.n_splits_ == 4
    assert len(search.cv_results_["params"]) == 3


@pytest.mark.e2e
def test_pipeline_inside_cross_val_score() -> None:
    X, y, pred, evalu = _make_dataset()  # noqa: N806
    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    cv = WalkForwardSplit(
        n_splits=4,
        test_size=5,
        prediction_times=pred,
        evaluation_times=evalu,
    )
    scores = cross_val_score(pipeline, X=X, y=y, cv=cv)
    assert len(scores) == 4
    assert np.all(np.isfinite(scores))
