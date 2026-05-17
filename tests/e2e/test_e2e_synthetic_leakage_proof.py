"""E2E: the controlled leakage result must keep holding.

This mirrors ``examples/synthetic_leakage_proof.ipynb`` but as a library-level
behavioural test (notebooks are docs here, not run under pytest). On a target
that is unpredictable by construction, naive shuffled CV fabricates large
positive skill; PurgedKFold removes the label overlap and that skill collapses
below a predict-the-mean baseline. Deterministic via the fixed seed.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from purgedcv import PurgedKFold
from purgedcv.diagnostics import compute_overlap_fraction

SEED = 0
N = 1000
H = 20
N_SPLITS = 5


@pytest.mark.e2e
def test_synthetic_leakage_proof_holds() -> None:
    rng = np.random.default_rng(SEED)
    e = rng.standard_normal(N + H)
    y = np.array([e[t : t + H].mean() for t in range(N)])  # depends only on noise
    x = np.cumsum(rng.gamma(shape=2.0, scale=1.0, size=N))  # monotone, unrelated
    features = x.reshape(-1, 1)
    pred = pd.Series(pd.date_range("2020-01-01", periods=N, freq="D"))
    evalu = pred + pd.Timedelta(days=H)

    naive = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    purged = PurgedKFold(
        n_splits=N_SPLITS,
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon=f"{H}D",
        embargo=f"{H}D",
    )

    def r2(model: object, cv: object) -> float:
        return float(cross_val_score(model, features, y, cv=cv, scoring="r2").mean())

    def mean_overlap(cv: KFold | PurgedKFold) -> float:
        return float(
            np.mean(
                [compute_overlap_fraction(tr, te, pred, evalu) for tr, te in cv.split(features, y)]
            )
        )

    # Mechanism: every training row leaks under naive, none under purged.
    assert mean_overlap(naive) == 1.0
    assert mean_overlap(purged) == 0.0

    # Reference: predicting the mean is ~0 either way.
    assert abs(r2(DummyRegressor(strategy="mean"), naive)) < 0.1

    for model in (
        KNeighborsRegressor(n_neighbors=10),
        RandomForestRegressor(n_estimators=120, random_state=SEED),
    ):
        naive_r2 = r2(model, naive)
        purged_r2 = r2(model, purged)
        # Naive fabricates clearly positive skill on an unpredictable target.
        assert naive_r2 > 0.5, (model, naive_r2)
        # Purged shows no positive skill -- the correct verdict.
        assert purged_r2 <= 0.0, (model, purged_r2)
        # The fabricated skill is large and collapses once the overlap is gone.
        assert naive_r2 - purged_r2 > 1.0, (model, naive_r2, purged_r2)


@pytest.mark.e2e
def test_purge_horizon_dose_response() -> None:
    """Residual leakage falls monotonically and is gone exactly when the
    purge horizon reaches the true label horizon -- not a step sooner."""
    rng = np.random.default_rng(SEED)
    e = rng.standard_normal(N + H)
    y = np.array([e[t : t + H].mean() for t in range(N)])
    x = np.cumsum(rng.gamma(shape=2.0, scale=1.0, size=N))
    features = x.reshape(-1, 1)
    pred = pd.Series(pd.date_range("2020-01-01", periods=N, freq="D"))
    true_evalu = pred + pd.Timedelta(days=H)

    horizons = list(range(0, int(1.5 * H) + 1, 2))
    residual: list[float] = []
    for assumed in horizons:
        cv_p = PurgedKFold(
            n_splits=N_SPLITS,
            prediction_times=pred,
            evaluation_times=pred + pd.Timedelta(days=assumed),
        )
        fr = [
            compute_overlap_fraction(tr, te, pred, true_evalu) for tr, te in cv_p.split(features, y)
        ]
        residual.append(float(np.mean(fr)))

    # Under-purging leaks; monotone non-increasing as the horizon grows.
    assert residual[0] > 0.0
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(residual))

    # The crossing is exactly at the true horizon: zero only once purge >= H,
    # strictly positive for every horizon below it.
    for h, r in zip(horizons, residual, strict=True):
        if h < H:
            assert r > 0.0, (h, r)
        else:
            assert r == 0.0, (h, r)
