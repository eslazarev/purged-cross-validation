"""Selection regret with an *innocent*, best-practice entity feature.

The companion ``selection_regret_lcl_seeds.py`` uses a raw household-identifier
integer as the leakage-enabling feature. A reviewer can object that a competent
practitioner would never feed a raw identifier to a tree, so the effect might be
an artifact of a contrived feature.

This script replaces the raw identifier with the most common realistic
entity-correlated feature in load forecasting: the customer's average daily
consumption ("baseline load"). It is built the way a careful practitioner would,
as a *causal* encoder fit per fold on the training side only:

  * for each row, the household's mean target is computed over that household's
    *earlier* training targets only, so the encoder reads no validation-fold
    targets, no future targets, and never the row's own target;
  * a household absent from the training fold, or a row with no earlier history
    (a genuinely unseen customer, the deployment condition), takes a cold-start
    prior pooled, inside each fit, from the pre-2013 daily consumption of the
    training-fold households only. A held-out household never contributes.

The encoder is thus free of target leakage, temporal look-ahead, and
fold-contaminated preprocessing. Shuffled k-fold nevertheless retains the intended
entity-overlap leakage: because the same household sits in both the train and test
side of a shuffled fold, the encoder gives test rows their own household's mean,
which is unavailable for a truly new customer. Group-aware cross-validation puts
whole households on the test side, so they take the training-only prior, exactly as
at deployment. The question
this script answers empirically: does the innocent feature reproduce the
selection-regret gap?

Outputs:
  - examples/data/selection_regret_lcl_targetenc.csv
  - examples/data/selection_regret_lcl_targetenc_summary.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from purgedcv import PurgedGroupKFold

N_SELECT_HH = 48
N_SEEDS = 30
YEAR_START, YEAR_END = "2013-01-01", "2014-01-01"

DATA_DIR = Path("data") if Path("data").exists() else Path("examples/data")
# Numeric features, then a date ordinal and the household id as the LAST two
# columns. The encoder below consumes the date, replaces the id with a causal
# per-household target mean (earlier dates only), and drops the date column.
NUM_FEATURES = ["lag_1", "lag_7"] + [f"dow_{d}" for d in range(7)]
MODEL_NAMES = ["kNN(k=1)", "kNN(k=5)", "kNN(k=50)", "Ridge a=.01",
               "Ridge a=1", "Ridge a=100", "RF d=None", "RF d=4"]


class CausalHouseholdMeanEncoder(BaseEstimator, TransformerMixin):
    """Replace the household-id column with a *causal* expanding mean of the
    household's target: for each row, the mean of that household's training
    targets on strictly earlier dates. A row with no earlier same-household
    history (an unseen customer, or a household's earliest day) takes a cold-start
    prior pooled, inside ``fit``, from the pre-window data of the households present
    in the *training fold only*; a held-out household never contributes. The
    encoder is fit on the training fold only and reads no validation-fold targets,
    no future targets, no row's own target, and no held-out household's data, so its
    construction is free of target leakage, temporal look-ahead, and
    fold-contaminated preprocessing. Shuffled folds still expose the intended
    entity-overlap leakage, the mechanism under study. Input columns are the numeric
    features, then a date ordinal, then the household id (last two columns); the date
    column is consumed and dropped, so the output width equals the numeric features
    plus the encoding.

    ``pre2013_stats`` maps integer household id to ``(sum, count)`` of that
    household's pre-window daily consumption; only the training fold's households
    are read."""

    def __init__(self, pre2013_stats=None):
        self.pre2013_stats = pre2013_stats

    def fit(self, X, y):  # noqa: N803
        X = np.asarray(X, dtype=float)  # noqa: N806
        y = np.asarray(y, dtype=float)
        date, ids = X[:, -2], X[:, -1]
        train_ids = np.unique(ids)
        stats = self.pre2013_stats or {}
        s = c = 0.0
        for g in train_ids:
            st = stats.get(float(g))
            if st:
                s += st[0]
                c += st[1]
        self.cold_start_prior_ = s / c if c else 0.0  # training-fold households only
        self.by_id_ = {}
        for g in train_ids:
            m = ids == g
            o = np.argsort(date[m], kind="mergesort")
            self.by_id_[float(g)] = (date[m][o], np.concatenate(([0.0], np.cumsum(y[m][o]))))
        return self

    def transform(self, X):  # noqa: N803
        X = np.asarray(X, dtype=float)  # noqa: N806
        date, ids = X[:, -2], X[:, -1]
        enc = np.full(len(X), float(self.cold_start_prior_))  # training-only pre-window prior
        for g, (sd, pref) in self.by_id_.items():
            m = ids == g
            if not m.any():
                continue
            k = np.searchsorted(sd, date[m], side="left")  # strictly earlier same-household dates
            enc[m] = np.where(k > 0, pref[k] / np.maximum(k, 1), enc[m])
        return np.column_stack([X[:, :-2], enc])


def load_daily() -> pd.DataFrame:
    raw = pd.read_csv(DATA_DIR / "lcl_halfhourly.csv", parse_dates=["tstp"])
    raw = raw[(raw["tstp"] >= YEAR_START) & (raw["tstp"] < YEAR_END)]
    raw["date"] = raw["tstp"].dt.normalize()
    daily = (
        raw.groupby(["LCLid", "date"])["energy_kwh"].sum().reset_index()
        .rename(columns={"energy_kwh": "kwh"})
    )
    daily = daily.sort_values(["LCLid", "date"]).reset_index(drop=True)
    daily["lag_1"] = daily.groupby("LCLid")["kwh"].shift(1)
    daily["lag_7"] = daily.groupby("LCLid")["kwh"].shift(7)
    hh_to_int = {h: i for i, h in enumerate(sorted(daily["LCLid"].unique()))}
    daily["hh_int"] = daily["LCLid"].map(hh_to_int).astype(float)
    for d in range(7):
        daily[f"dow_{d}"] = (daily["date"].dt.dayofweek == d).astype(float)
    daily["date_ord"] = (daily["date"] - pd.Timestamp(YEAR_START)).dt.days.astype(float)
    daily["target"] = daily["kwh"]
    cols = [*NUM_FEATURES, "hh_int", "target"]
    return daily.dropna(subset=cols).reset_index(drop=True)


def pre2013_stats() -> dict:
    """Per-household ``(sum, count)`` of daily consumption strictly before the
    modelling window (pre-2013), keyed by the integer household id used in the
    features. The encoder pools these over the *training-fold* households only to
    form the cold-start prior, so a held-out household never contributes to the
    prior used on it (the prior uses no held-out, future, or in-window-target data)."""
    raw = pd.read_csv(DATA_DIR / "lcl_halfhourly.csv", parse_dates=["tstp"])
    hh_to_int = {h: i for i, h in enumerate(sorted(raw["LCLid"].unique()))}
    raw["date"] = raw["tstp"].dt.normalize()
    pre = raw[raw["tstp"] < YEAR_START]
    daily = pre.groupby(["LCLid", "date"])["energy_kwh"].sum().reset_index()
    agg = daily.groupby("LCLid")["energy_kwh"].agg(["sum", "count"])
    return {float(hh_to_int[h]): (float(r["sum"]), int(r["count"])) for h, r in agg.iterrows()}


def build_grid(seed: int, stats: dict):
    """Each candidate is a pipeline: causal household-mean encoder (with a
    training-only pre-window cold-start prior), then a scaler, then the estimator."""
    enc = CausalHouseholdMeanEncoder
    return [
        ("kNN(k=1)", make_pipeline(enc(pre2013_stats=stats),StandardScaler(), KNeighborsRegressor(n_neighbors=1))),
        ("kNN(k=5)", make_pipeline(enc(pre2013_stats=stats),StandardScaler(), KNeighborsRegressor(n_neighbors=5))),
        ("kNN(k=50)", make_pipeline(enc(pre2013_stats=stats),StandardScaler(), KNeighborsRegressor(n_neighbors=50))),
        ("Ridge a=.01", make_pipeline(enc(pre2013_stats=stats),StandardScaler(), Ridge(alpha=0.01))),
        ("Ridge a=1", make_pipeline(enc(pre2013_stats=stats),StandardScaler(), Ridge(alpha=1.0))),
        ("Ridge a=100", make_pipeline(enc(pre2013_stats=stats),StandardScaler(), Ridge(alpha=100.0))),
        ("RF d=None", make_pipeline(enc(pre2013_stats=stats),RandomForestRegressor(n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1))),
        ("RF d=4", make_pipeline(enc(pre2013_stats=stats),RandomForestRegressor(n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1))),
    ]


# Feature columns fed to the pipeline: numeric features, date ordinal, then id.
# The causal encoder consumes the date ordinal and id, emitting numeric + mean.
COLS = [*NUM_FEATURES, "date_ord", "hh_int"]


def run_one(clean: pd.DataFrame, all_hh: list, seed: int, stats: dict) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_hh))
    sel_hh = {all_hh[i] for i in perm[:N_SELECT_HH]}
    sel = clean[clean["LCLid"].isin(sel_hh)].sort_values(["date", "LCLid"]).reset_index(drop=True)
    dep = clean[~clean["LCLid"].isin(sel_hh)].sort_values(["date", "LCLid"]).reset_index(drop=True)
    x_s, y_s = sel[COLS].to_numpy(), sel["target"].to_numpy()
    x_d, y_d = dep[COLS].to_numpy(), dep["target"].to_numpy()
    g_s = pd.Series(sel["LCLid"].to_numpy())
    p_s = sel["date"].reset_index(drop=True)

    grid = build_grid(seed, stats)
    naive_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    honest_cv = PurgedGroupKFold(n_splits=5, prediction_times=p_s, evaluation_times=p_s, groups=g_s)

    def score(cv_obj, model):
        return -cross_val_score(model, x_s, y_s, cv=cv_obj,
                                scoring="neg_mean_absolute_error", n_jobs=1).mean()

    naive = {nm: score(naive_cv, m) for nm, m in grid}
    honest = {nm: score(honest_cv, m) for nm, m in grid}

    def deploy(model):
        mm = clone(model)
        mm.fit(x_s, y_s)
        p = mm.predict(x_d)
        return mean_absolute_error(y_d, p), r2_score(y_d, p)

    dep_mae, dep_r2 = {}, {}
    for nm, m in grid:
        dep_mae[nm], dep_r2[nm] = deploy(m)

    nw = min(naive, key=naive.get)
    hw = min(honest, key=honest.get)
    best = min(dep_mae, key=dep_mae.get)

    return {
        "seed": seed, "naive_pick": nw, "honest_pick": hw, "best_deploy_pick": best,
        "naive_cv_mae": naive[nw], "honest_cv_mae": honest[hw],
        "naive_dep_mae": dep_mae[nw], "naive_dep_r2": dep_r2[nw],
        "honest_dep_mae": dep_mae[hw], "honest_dep_r2": dep_r2[hw],
        "best_dep_mae": dep_mae[best],
        "regret_naive": dep_mae[nw] - dep_mae[best],
        "regret_honest": dep_mae[hw] - dep_mae[best],
        "delta_mae": dep_mae[nw] - dep_mae[hw],
        "delta_pct": (dep_mae[nw] - dep_mae[hw]) / dep_mae[nw] * 100.0,
        **{f"naive_cv[{nm}]": naive[nm] for nm in MODEL_NAMES},
        **{f"honest_cv[{nm}]": honest[nm] for nm in MODEL_NAMES},
        **{f"dep_mae[{nm}]": dep_mae[nm] for nm in MODEL_NAMES},
    }


def quart(a: np.ndarray) -> tuple:
    return float(np.percentile(a, 25)), float(np.median(a)), float(np.percentile(a, 75))


def main() -> None:
    clean = load_daily()
    stats = pre2013_stats()
    print(f"pre-2013 cold-start prior pooled over all 60 households: "
          f"{sum(v[0] for v in stats.values()) / sum(v[1] for v in stats.values()):.4f} kWh/day "
          f"(the experiment pools only training-fold households inside each fit)")
    all_hh = sorted(clean["LCLid"].unique())
    rows = []
    for s in range(N_SEEDS):
        r = run_one(clean, all_hh, s, stats)
        rows.append(r)
        print(f"seed {s:>2}: naive {r['naive_pick']:<11} {r['naive_dep_mae']:.4f}  "
              f"honest {r['honest_pick']:<11} {r['honest_dep_mae']:.4f}  "
              f"best {r['best_deploy_pick']:<11} {r['best_dep_mae']:.4f}  "
              f"delta {r['delta_mae']:+.4f}  reg_n {r['regret_naive']:.4f} reg_h {r['regret_honest']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "selection_regret_lcl_targetenc.csv"
    df.to_csv(out_csv, index=False)

    d = df["delta_mae"].to_numpy()
    dp = df["delta_pct"].to_numpy()
    rn = df["regret_naive"].to_numpy()
    rh = df["regret_honest"].to_numpy()
    n = len(d)
    win = int((d > 0).sum())
    q1, med, q3 = quart(d)
    q1p, medp, q3p = quart(dp)
    rn_q1, rn_med, rn_q3 = quart(rn)
    rh_q1, rh_med, rh_q3 = quart(rh)
    naive_top = df["naive_pick"].value_counts()
    honest_top = df["honest_pick"].value_counts()
    honest_is_best = int((df["honest_pick"] == df["best_deploy_pick"]).sum())

    def fmt_counts(vc):
        return ", ".join(f"{k} {v}/{n}" for k, v in vc.items())

    summary = (
        f"# Selection regret with an innocent entity feature (LCL, fixed N=60, 48/12 split, {n} partitions)\n\n"
        f"Generated by `examples/selection_regret_lcl_targetenc.py`.\n\n"
        f"The raw household identifier is replaced by a causal target-mean encoding "
        f"of the household (the customer's average daily consumption), computed per fold from "
        f"earlier dates only, with a training-only pre-2013 cold-start prior for unseen households. "
        f"Each seed is a different 48/12 partition of the same 60-household sample (correlated "
        f"resamples, not independent draws), so spreads are descriptive.\n\n"
        f"## Model selection\n"
        f"- Naive shuffled KFold picks: {fmt_counts(naive_top)}.\n"
        f"- PurgedGroupKFold picks: {fmt_counts(honest_top)}.\n"
        f"- The honest pick is the best-deploying candidate in {honest_is_best}/{n} partitions.\n\n"
        f"## Deployment-regret (selected minus best-deploying candidate)\n"
        f"- Naive selector regret: median {rn_med:.4f} kWh (IQR {rn_q1:.4f} to {rn_q3:.4f}).\n"
        f"- Honest selector regret: median {rh_med:.4f} kWh (IQR {rh_q1:.4f} to {rh_q3:.4f}).\n\n"
        f"## Naive minus honest deployment MAE\n"
        f"- Honest deploys at lower MAE in {win}/{n} partitions.\n"
        f"- Gap: median {med:.4f} kWh (IQR {q1:.4f} to {q3:.4f}), range {d.min():.4f} to {d.max():.4f}.\n"
        f"- Relative gap: median {medp:.2f}% (IQR {q1p:.2f} to {q3p:.2f}%).\n"
    )
    (DATA_DIR / "selection_regret_lcl_targetenc_summary.md").write_text(summary)
    print("\n" + summary)
    print(f"csv -> {out_csv}")


if __name__ == "__main__":
    main()
