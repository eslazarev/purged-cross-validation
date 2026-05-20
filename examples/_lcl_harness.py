"""Verbatim extract of the LCL notebook's feature + four-scheme WAPE cells.

Single source of truth shared by `examples/uk_smart_meter_lcl.ipynb`
(documentation, runs the inline copy), `tools/lcl_full_benchmark.py`
(offline full-population run), and the e2e test. Keeping one copy here
stops the offline run and the demo drifting apart. Repo-only helper:
`pyproject.toml` ships only `src/purgedcv`, so this is never packaged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold

from purgedcv import WalkForwardSplit
from purgedcv._typing import NDArrayAny

FEAT_COLS = ["lag_1", "lag_48", "lag_336", "hh", "dow", "weekend"]


def build_features(
    df: pd.DataFrame, *, h: int
) -> tuple[NDArrayAny, NDArrayAny, pd.Series, pd.Series, NDArrayAny]:
    """Lag/calendar features and the next-`h`-half-hour mean-demand label.

    `df` has columns ``LCLid``, ``tstp`` (datetime), ``load``. Lags and the
    forward window are computed per household; the label of nearby rows
    overlaps by construction. Returns ``(features, y, pred, evalu, groups)``.
    """
    df = df.sort_values(["tstp", "LCLid"]).reset_index(drop=True)
    g = df.groupby("LCLid")["load"]
    df["lag_1"] = g.shift(1)
    df["lag_48"] = g.shift(48)  # same half-hour, yesterday
    df["lag_336"] = g.shift(336)  # same half-hour, last week
    # At row t the target is the mean of rows t+1 ... t+h.
    # `shift(-h).rolling(h)` aligns that forward window back to row t.
    df["target"] = g.transform(lambda v: v.shift(-h).rolling(h).mean())
    df["hh"] = df["tstp"].dt.hour * 2 + df["tstp"].dt.minute // 30
    df["dow"] = df["tstp"].dt.dayofweek
    df["weekend"] = (df["dow"] >= 5).astype(int)

    data = df.dropna(subset=[*FEAT_COLS, "target"]).reset_index(drop=True)
    features: NDArrayAny = data[FEAT_COLS].to_numpy()
    y: NDArrayAny = data["target"].to_numpy()
    pred = data["tstp"].reset_index(drop=True)
    evalu = pred + pd.Timedelta(minutes=30 * h)
    groups: NDArrayAny = data["LCLid"].to_numpy()
    return features, y, pred, evalu, groups


def four_scheme_wape(
    features: NDArrayAny,
    y: NDArrayAny,
    pred: pd.Series,
    evalu: pd.Series,
    groups: NDArrayAny,
    *,
    seed: int,
    n_splits: int,
    h: int,
) -> dict[str, float]:
    """WAPE (%) of the same HistGBR model under four CV schemes.

    WAPE = sum|err| / sum|actual|, robust to the many near-zero half-hourly
    household readings that make raw MAPE undefined. Returns a dict keyed by
    scheme name: naive shuffled k-fold, blocked k-fold, WalkForwardSplit,
    GroupKFold (household).
    """

    def fold_wape(splits: list[tuple[NDArrayAny, NDArrayAny]]) -> float:
        vals: list[float] = []
        for tr, te in splits:
            model = HistGradientBoostingRegressor(random_state=seed)
            model.fit(features[tr], y[tr])
            pr = model.predict(features[te])
            vals.append(float(np.abs(y[te] - pr).sum() / np.abs(y[te]).sum()))
        return float(np.mean(vals)) * 100.0

    wf = WalkForwardSplit(
        n_splits=n_splits,
        test_size=len(y) // 8,
        window="expanding",
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon=pd.Timedelta(minutes=30 * h),
    )
    splits = {
        "naive shuffled k-fold": list(
            KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(features)
        ),
        "blocked k-fold": list(KFold(n_splits=n_splits, shuffle=False).split(features)),
        "WalkForwardSplit": list(wf.split(features)),
        "GroupKFold (household)": list(GroupKFold(n_splits=n_splits).split(features, y, groups)),
    }
    return {name: round(fold_wape(sp), 2) for name, sp in splits.items()}
