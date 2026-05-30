"""Multi-seed robustness sweep for the LCL selection-regret experiment.

Companion to ``selection_regret_lcl.ipynb``. The notebook reports a single
48/12 household split plus a five-seed preview. This script repeats the full
pipeline over many partitions and, for each one, records:

  * the model each cross-validator selects under three schemes: naive shuffled
    KFold, sklearn GroupKFold, and purgedcv PurgedGroupKFold;
  * the deployment MAE of *every* candidate on the held-out households, so the
    best-deploying candidate (M_OOS) is known and model-selection regret can be
    computed directly, regret(selector) = deployMAE(selected) - deployMAE(M_OOS),
    following Teodorescu and Obreja Brasoveanu (2025).

Important statistical caveat: every seed is a different 48/12 partition of the
*same* fixed 60-household sample. These are correlated resamples without
replacement, not independent draws from the household population, so we report
descriptive spread (median, IQR, range) and a win rate rather than a population
confidence interval. Population-level inference is left to the independent
20-subsample full-population benchmark.

Outputs:
  - examples/data/selection_regret_lcl_seeds.csv
  - examples/data/selection_regret_lcl_seeds_summary.md
  - .github/images/selection_regret_lcl_seeds.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from purgedcv import PurgedGroupKFold

N_SELECT_HH = 48
N_SEEDS = 30
YEAR_START, YEAR_END = "2013-01-01", "2014-01-01"

DATA_DIR = Path("data") if Path("data").exists() else Path("examples/data")
IMG_DIR = Path("../.github/images") if Path("../.github/images").is_dir() else Path(".github/images")
FEATURES = ["lag_1", "lag_7", "hh_int"] + [f"dow_{d}" for d in range(7)]
MODEL_NAMES = ["kNN(k=1)", "kNN(k=5)", "kNN(k=50)", "Ridge a=.01",
               "Ridge a=1", "Ridge a=100", "RF d=None", "RF d=4"]


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
    daily["target"] = daily["kwh"]
    return daily.dropna(subset=[*FEATURES, "target"]).reset_index(drop=True)


def build_grid(seed: int):
    return [
        ("kNN(k=1)", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1))),
        ("kNN(k=5)", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))),
        ("kNN(k=50)", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=50))),
        ("Ridge a=.01", make_pipeline(StandardScaler(), Ridge(alpha=0.01))),
        ("Ridge a=1", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("Ridge a=100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("RF d=None", RandomForestRegressor(n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1)),
        ("RF d=4", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1)),
    ]


def run_one(clean: pd.DataFrame, all_hh: list, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_hh))
    sel_hh = {all_hh[i] for i in perm[:N_SELECT_HH]}
    sel = clean[clean["LCLid"].isin(sel_hh)].sort_values(["date", "LCLid"]).reset_index(drop=True)
    dep = clean[~clean["LCLid"].isin(sel_hh)].sort_values(["date", "LCLid"]).reset_index(drop=True)
    x_s, y_s = sel[FEATURES].to_numpy(), sel["target"].to_numpy()
    x_d, y_d = dep[FEATURES].to_numpy(), dep["target"].to_numpy()
    g_s = pd.Series(sel["LCLid"].to_numpy())
    p_s = sel["date"].reset_index(drop=True)

    grid = build_grid(seed)
    naive_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    group_cv = GroupKFold(n_splits=5)
    honest_cv = PurgedGroupKFold(n_splits=5, prediction_times=p_s, evaluation_times=p_s, groups=g_s)

    def score(cv_obj, model, **kw):
        return -cross_val_score(model, x_s, y_s, cv=cv_obj,
                                scoring="neg_mean_absolute_error", n_jobs=1, **kw).mean()

    naive = {nm: score(naive_cv, m) for nm, m in grid}
    group = {nm: score(group_cv, m, groups=g_s) for nm, m in grid}
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
    gw = min(group, key=group.get)
    hw = min(honest, key=honest.get)
    best = min(dep_mae, key=dep_mae.get)  # M_OOS: best-deploying candidate

    return {
        "seed": seed,
        "naive_pick": nw, "group_pick": gw, "honest_pick": hw, "best_deploy_pick": best,
        "naive_dep_mae": dep_mae[nw], "naive_dep_r2": dep_r2[nw],
        "honest_dep_mae": dep_mae[hw], "honest_dep_r2": dep_r2[hw],
        "group_dep_mae": dep_mae[gw],
        "best_dep_mae": dep_mae[best],
        "regret_naive": dep_mae[nw] - dep_mae[best],
        "regret_honest": dep_mae[hw] - dep_mae[best],
        "delta_mae": dep_mae[nw] - dep_mae[hw],
        "delta_pct": (dep_mae[nw] - dep_mae[hw]) / dep_mae[nw] * 100.0,
        "group_eq_honest": gw == hw,
        **{f"dep_mae[{nm}]": dep_mae[nm] for nm in MODEL_NAMES},
    }


def quart(a: np.ndarray) -> tuple:
    return float(np.percentile(a, 25)), float(np.median(a)), float(np.percentile(a, 75))


def main() -> None:
    clean = load_daily()
    all_hh = sorted(clean["LCLid"].unique())
    print(f"households in cached sample: {len(all_hh)}")
    rows = []
    for s in range(N_SEEDS):
        r = run_one(clean, all_hh, s)
        rows.append(r)
        print(f"seed {s:>2}: naive {r['naive_pick']:<11} {r['naive_dep_mae']:.4f}  "
              f"honest {r['honest_pick']:<11} {r['honest_dep_mae']:.4f}  "
              f"best {r['best_deploy_pick']:<11} {r['best_dep_mae']:.4f}  "
              f"delta {r['delta_mae']:+.4f}  reg_n {r['regret_naive']:.4f} reg_h {r['regret_honest']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "selection_regret_lcl_seeds.csv"
    df.to_csv(out_csv, index=False)

    d = df["delta_mae"].to_numpy()
    dp = df["delta_pct"].to_numpy()
    rn = df["regret_naive"].to_numpy()
    rh = df["regret_honest"].to_numpy()
    n = len(d)
    win_rate = int((d > 0).sum())
    q1, med, q3 = quart(d)
    q1p, medp, q3p = quart(dp)
    naive_rf = int((df["naive_pick"] == "RF d=None").sum())
    honest_ridge = int((df["honest_pick"] == "Ridge a=.01").sum())
    group_eq = int(df["group_eq_honest"].sum())
    honest_is_best = int((df["honest_pick"] == df["best_deploy_pick"]).sum())
    rn_q1, rn_med, rn_q3 = quart(rn)
    rh_q1, rh_med, rh_q3 = quart(rh)

    summary = (
        f"# Selection-regret robustness across {n} partitions (LCL, fixed N=60 sample, 48/12 split)\n\n"
        f"Generated by `examples/selection_regret_lcl_seeds.py`.\n\n"
        f"Each seed is a different 48/12 partition of the *same* 60-household cached sample. "
        f"These are correlated resamples without replacement, not independent draws from the "
        f"household population, so figures below are descriptive spread plus a win rate, not a "
        f"population confidence interval.\n\n"
        f"## Model selection\n"
        f"- Naive shuffled KFold selects `RF d=None` in {naive_rf}/{n} partitions.\n"
        f"- PurgedGroupKFold selects `Ridge a=.01` in {honest_ridge}/{n} partitions.\n"
        f"- sklearn GroupKFold selects the same model as PurgedGroupKFold in {group_eq}/{n} "
        f"partitions (instantaneous labels reduce the purged split to a group-disjoint split).\n"
        f"- The honest pick is the best-deploying candidate (M_OOS) in {honest_is_best}/{n} partitions.\n\n"
        f"## Deployment-regret (deploy MAE of selected minus deploy MAE of best candidate)\n"
        f"- Naive selector regret: median {rn_med:.4f} kWh (IQR {rn_q1:.4f} to {rn_q3:.4f}).\n"
        f"- Honest selector regret: median {rh_med:.4f} kWh (IQR {rh_q1:.4f} to {rh_q3:.4f}).\n\n"
        f"## Naive minus honest deployment MAE (the practitioner-visible gap)\n"
        f"- Honest deploys at lower MAE in {win_rate}/{n} partitions.\n"
        f"- Gap: median {med:.4f} kWh (IQR {q1:.4f} to {q3:.4f}), range {d.min():.4f} to {d.max():.4f}.\n"
        f"- Relative gap: median {medp:.2f}% (IQR {q1p:.2f} to {q3p:.2f}%), "
        f"range {dp.min():.2f} to {dp.max():.2f}%.\n"
    )
    (DATA_DIR / "selection_regret_lcl_seeds_summary.md").write_text(summary)
    print("\n" + summary)

    # Distribution figure: paired deployment MAE + delta histogram.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#cccccc")

    for _, row in df.iterrows():
        ax1.plot([0, 1], [row["naive_dep_mae"], row["honest_dep_mae"]],
                 color="#bbbbbb", lw=0.8, zorder=1)
    ax1.scatter(np.zeros(n), df["naive_dep_mae"], color="#e0564c", s=28, zorder=3, label="naive")
    ax1.scatter(np.ones(n), df["honest_dep_mae"], color="#3a9d6e", s=28, zorder=3, label="honest")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["naive shuffled\nKFold", "PurgedGroupKFold\n(honest)"])
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylabel("deployment MAE on held-out households (kWh)", fontsize=10, color="#444444")
    ax1.tick_params(colors="#666666", labelsize=9)
    ax1.grid(axis="y", ls=":", color="#dddddd", alpha=0.8, zorder=0)

    ax2.hist(d, bins=12, color="#3a9d6e", edgecolor="white", zorder=3)
    ax2.axvline(0, color="#e0564c", ls="--", lw=1.4, zorder=4)
    ax2.set_xlabel("naive minus honest deployment MAE (kWh)", fontsize=10, color="#444444")
    ax2.set_ylabel("partitions", fontsize=10, color="#444444")
    ax2.tick_params(colors="#666666", labelsize=9)
    ax2.grid(axis="y", ls=":", color="#dddddd", alpha=0.8, zorder=0)

    fig.text(0.5, 0.97, f"Honest CV deploys better in {win_rate} of {n} partitions",
             ha="center", fontsize=13, fontweight="bold", color="#222222")
    fig.text(0.5, 0.92,
             f"median gap {med:.3f} kWh (IQR {q1:.3f} to {q3:.3f}); "
             f"positive means honest deploys better",
             ha="center", fontsize=9.5, color="#777777")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.97, wspace=0.25)

    if IMG_DIR.is_dir():
        fig.savefig(IMG_DIR / "selection_regret_lcl_seeds.png", dpi=150, bbox_inches="tight")
        print(f"figure -> {IMG_DIR / 'selection_regret_lcl_seeds.png'}")
    print(f"csv    -> {out_csv}")


if __name__ == "__main__":
    main()
