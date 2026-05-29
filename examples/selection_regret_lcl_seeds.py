"""Multi-seed robustness sweep for the LCL selection-regret experiment.

Companion to ``selection_regret_lcl.ipynb``. The notebook reports a single
48/12 household split plus a five-seed preview. A single split is one
realisation, and the dominant source of model-selection regret is known to be
the train/test split itself (Teodorescu and Obreja Brasoveanu, 2025). To turn
one number into a distribution, this script repeats the full pipeline over
many seeds and reports the deployment-regret delta with a 95% confidence
interval, the win rate, and which model each cross-validator picks.

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
from scipy import stats
from sklearn.base import clone
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
IMG_DIR = Path("../.github/images") if Path("../.github/images").is_dir() else Path(".github/images")
FEATURES = ["lag_1", "lag_7", "hh_int"] + [f"dow_{d}" for d in range(7)]


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
        ("kNN(k=1)",    make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1))),
        ("kNN(k=5)",    make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))),
        ("kNN(k=50)",   make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=50))),
        ("Ridge a=.01", make_pipeline(StandardScaler(), Ridge(alpha=0.01))),
        ("Ridge a=1",   make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("Ridge a=100", make_pipeline(StandardScaler(), Ridge(alpha=100.0))),
        ("RF d=None",   RandomForestRegressor(n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1)),
        ("RF d=4",      RandomForestRegressor(n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1)),
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
    honest_cv = PurgedGroupKFold(n_splits=5, prediction_times=p_s, evaluation_times=p_s, groups=g_s)

    def score(cv_obj, model):
        return -cross_val_score(model, x_s, y_s, cv=cv_obj,
                                scoring="neg_mean_absolute_error", n_jobs=1).mean()

    naive = {nm: score(naive_cv, m) for nm, m in grid}
    honest = {nm: score(honest_cv, m) for nm, m in grid}
    nw = min(naive, key=naive.get)
    hw = min(honest, key=honest.get)
    picks = dict(grid)

    def deploy(model):
        mm = clone(model)
        mm.fit(x_s, y_s)
        p = mm.predict(x_d)
        return mean_absolute_error(y_d, p), r2_score(y_d, p)

    nmae, nr2 = deploy(picks[nw])
    hmae, hr2 = deploy(picks[hw])
    return {
        "seed": seed,
        "naive_pick": nw, "naive_dep_mae": nmae, "naive_dep_r2": nr2,
        "honest_pick": hw, "honest_dep_mae": hmae, "honest_dep_r2": hr2,
        "delta_mae": nmae - hmae,
        "delta_pct": (nmae - hmae) / nmae * 100.0,
    }


def main() -> None:
    clean = load_daily()
    all_hh = sorted(clean["LCLid"].unique())
    rows = []
    for s in range(N_SEEDS):
        r = run_one(clean, all_hh, s)
        rows.append(r)
        print(f"seed {s:>2}: naive {r['naive_pick']:<11} {r['naive_dep_mae']:.4f}  "
              f"honest {r['honest_pick']:<11} {r['honest_dep_mae']:.4f}  "
              f"delta {r['delta_mae']:+.4f} ({r['delta_pct']:+.1f}%)")

    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "selection_regret_lcl_seeds.csv"
    df.to_csv(out_csv, index=False)

    d = df["delta_mae"].to_numpy()
    dp = df["delta_pct"].to_numpy()
    n = len(d)
    win_rate = float((d > 0).mean()) * 100.0
    mean_d, sem_d = float(d.mean()), float(stats.sem(d))
    tcrit = float(stats.t.ppf(0.975, n - 1))
    lo, hi = mean_d - tcrit * sem_d, mean_d + tcrit * sem_d
    mean_p, sem_p = float(dp.mean()), float(stats.sem(dp))
    lo_p, hi_p = mean_p - tcrit * sem_p, mean_p + tcrit * sem_p
    naive_rf = float((df["naive_pick"] == "RF d=None").mean()) * 100.0
    honest_ridge = float((df["honest_pick"] == "Ridge a=.01").mean()) * 100.0

    summary = (
        f"# Selection-regret robustness across {n} seeds (LCL, N=60, 48/12 split)\n\n"
        f"Generated by `examples/selection_regret_lcl_seeds.py`.\n\n"
        f"- Naive shuffled KFold picks `RF d=None` in {naive_rf:.0f}% of seeds.\n"
        f"- PurgedGroupKFold picks `Ridge a=.01` in {honest_ridge:.0f}% of seeds.\n"
        f"- Honest pick deploys at lower MAE in {win_rate:.0f}% of seeds "
        f"({int((d > 0).sum())}/{n}).\n"
        f"- Deployment-regret delta (naive minus honest MAE): "
        f"mean {mean_d:+.4f} kWh (95% CI {lo:+.4f} to {hi:+.4f}), "
        f"median {float(np.median(d)):+.4f} kWh, "
        f"range {d.min():+.4f} to {d.max():+.4f}.\n"
        f"- Relative regret: mean {mean_p:+.2f}% (95% CI {lo_p:+.2f} to {hi_p:+.2f}%), "
        f"median {float(np.median(dp)):+.2f}%.\n"
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
    ax2.set_xlabel("deployment-regret delta: naive minus honest MAE (kWh)",
                   fontsize=10, color="#444444")
    ax2.set_ylabel("seeds", fontsize=10, color="#444444")
    ax2.tick_params(colors="#666666", labelsize=9)
    ax2.grid(axis="y", ls=":", color="#dddddd", alpha=0.8, zorder=0)

    fig.text(0.5, 0.97, f"Honest CV deploys better in {win_rate:.0f}% of {n} seeds",
             ha="center", fontsize=13, fontweight="bold", color="#222222")
    fig.text(0.5, 0.92,
             f"mean regret {mean_d:+.3f} kWh (95% CI {lo:+.3f} to {hi:+.3f}); "
             f"positive means honest deploys better",
             ha="center", fontsize=9.5, color="#777777")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.97, wspace=0.25)

    if IMG_DIR.is_dir():
        fig.savefig(IMG_DIR / "selection_regret_lcl_seeds.png", dpi=150, bbox_inches="tight")
        print(f"figure -> {IMG_DIR / 'selection_regret_lcl_seeds.png'}")
    print(f"csv    -> {out_csv}")


if __name__ == "__main__":
    main()
