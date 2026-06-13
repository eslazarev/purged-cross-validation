"""Companion quantities for the LCL selection-regret sweep.

Companion to ``selection_regret_lcl_seeds.py``. That script records, per
partition, which model each scheme selects and the deployment MAE of every
candidate. This one re-runs the identical protocol (same seeds, same
partitions, same grid construction) for the two picks the sweep settles on,
RF d=None under shuffled KFold and Ridge a=.01 under the group-aware split,
and records the quantities the headline sweep does not store:

  * the CV score of each scheme's own pick next to its deployment MAE, which
    separates ranking fidelity from score-level calibration: the shuffled
    pick's score is one-sidedly optimistic (leakage), the group-aware pick's
    score differs from deployment by split noise centered near zero;
  * deployment WAPE and household-averaged (macro) MAE for both picks, the
    scale-normalized companions to the pooled MAE of the main sweep;
  * how many of the 12 deployment households individually favor each pick,
    and, for seed 0, the per-household paired differences plus a
    household-cluster bootstrap interval for the pooled gap.

Recomputed deployment MAEs are asserted equal to the committed sweep CSV for
every seed, so the protocol match is verified rather than assumed.

Outputs:
  - examples/data/selection_regret_lcl_companions.csv
  - examples/data/selection_regret_lcl_companions_summary.md
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from purgedcv import PurgedGroupKFold

N_SELECT_HH = 48
N_SEEDS = 30
YEAR_START, YEAR_END = "2013-01-01", "2014-01-01"
BOOT_ITER = 10_000
BOOT_SEED = 12345

DATA_DIR = Path("data") if Path("data").exists() else Path("examples/data")
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


def wape_pct(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / np.abs(y).sum() * 100.0)


def run_one(clean: pd.DataFrame, all_hh: list, seed: int, ref_row: pd.Series) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_hh))
    sel_hh = {all_hh[i] for i in perm[:N_SELECT_HH]}
    sel = clean[clean["LCLid"].isin(sel_hh)].sort_values(["date", "LCLid"]).reset_index(drop=True)
    dep = clean[~clean["LCLid"].isin(sel_hh)].sort_values(["date", "LCLid"]).reset_index(drop=True)
    x_s, y_s = sel[FEATURES].to_numpy(), sel["target"].to_numpy()
    x_d, y_d = dep[FEATURES].to_numpy(), dep["target"].to_numpy()
    g_s = pd.Series(sel["LCLid"].to_numpy())
    p_s = sel["date"].reset_index(drop=True)

    ridge = make_pipeline(StandardScaler(), Ridge(alpha=0.01))
    rf = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=seed, n_jobs=-1)
    shuffled_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    group_cv = PurgedGroupKFold(n_splits=5, prediction_times=p_s, evaluation_times=p_s, groups=g_s)

    cv_ridge_group = -cross_val_score(
        ridge, x_s, y_s, cv=group_cv, scoring="neg_mean_absolute_error", n_jobs=1
    ).mean()
    cv_rf_shuffled = -cross_val_score(
        rf, x_s, y_s, cv=shuffled_cv, scoring="neg_mean_absolute_error", n_jobs=1
    ).mean()

    p_ridge = clone(ridge).fit(x_s, y_s).predict(x_d)
    p_rf = clone(rf).fit(x_s, y_s).predict(x_d)
    mae_ridge = mean_absolute_error(y_d, p_ridge)
    mae_rf = mean_absolute_error(y_d, p_rf)
    assert abs(mae_ridge - ref_row["honest_dep_mae"]) < 1e-9, f"seed {seed}: Ridge MAE mismatch"
    assert abs(mae_rf - ref_row["naive_dep_mae"]) < 1e-9, f"seed {seed}: RF MAE mismatch"

    dd = dep[["LCLid"]].copy()
    dd["y"], dd["p_ridge"], dd["p_rf"] = y_d, p_ridge, p_rf
    per_hh = dd.groupby("LCLid").apply(
        lambda g: pd.Series({
            "mae_ridge": np.abs(g["y"] - g["p_ridge"]).mean(),
            "mae_rf": np.abs(g["y"] - g["p_rf"]).mean(),
        }),
        include_groups=False,
    )

    row = {
        "seed": seed,
        "cv_mae_group_pick": cv_ridge_group,
        "cv_mae_shuffled_pick": cv_rf_shuffled,
        "dep_mae_group_pick": mae_ridge,
        "dep_mae_shuffled_pick": mae_rf,
        "dep_minus_cv_group_pick": mae_ridge - cv_ridge_group,
        "dep_minus_cv_shuffled_pick": mae_rf - cv_rf_shuffled,
        "wape_group_pick": wape_pct(y_d, p_ridge),
        "wape_shuffled_pick": wape_pct(y_d, p_rf),
        "macro_mae_group_pick": float(per_hh["mae_ridge"].mean()),
        "macro_mae_shuffled_pick": float(per_hh["mae_rf"].mean()),
        "hh_favoring_group_pick": int((per_hh["mae_ridge"] < per_hh["mae_rf"]).sum()),
        "n_dep_households": len(per_hh),
        "dep_mean_kwh": float(np.mean(y_d)),
    }

    detail = {}
    if seed == 0:
        boot_rng = np.random.default_rng(BOOT_SEED)
        hh_ids = per_hh.index.to_numpy()
        by_hh = {h: g for h, g in dd.groupby("LCLid")}
        diffs = np.empty(BOOT_ITER)
        for b in range(BOOT_ITER):
            pick = boot_rng.choice(hh_ids, size=len(hh_ids), replace=True)
            gb = pd.concat([by_hh[h] for h in pick])
            diffs[b] = np.abs(gb["y"] - gb["p_rf"]).mean() - np.abs(gb["y"] - gb["p_ridge"]).mean()
        paired = per_hh["mae_rf"] - per_hh["mae_ridge"]
        detail = {
            "boot_lo": float(np.percentile(diffs, 2.5)),
            "boot_hi": float(np.percentile(diffs, 97.5)),
            "paired_min": float(paired.min()),
            "paired_median": float(paired.median()),
            "paired_max": float(paired.max()),
            "favoring": int((paired > 0).sum()),
            "n_hh": len(per_hh),
            "dep_mean_kwh": row["dep_mean_kwh"],
        }
    return row, detail


def main() -> None:
    ref = pd.read_csv(DATA_DIR / "selection_regret_lcl_seeds.csv")
    assert (ref["naive_pick"] == "RF d=None").all()
    assert (ref["honest_pick"] == "Ridge a=.01").all()

    clean = load_daily()
    all_hh = sorted(clean["LCLid"].unique())
    print(f"households in cached sample: {len(all_hh)}")

    rows, seed0 = [], {}
    for s in range(N_SEEDS):
        row, detail = run_one(clean, all_hh, s, ref.loc[ref["seed"] == s].iloc[0])
        rows.append(row)
        if detail:
            seed0 = detail
        print(f"seed {s:>2}: dep-cv group {row['dep_minus_cv_group_pick']:+.4f}  "
              f"shuffled {row['dep_minus_cv_shuffled_pick']:+.4f}  "
              f"wape {row['wape_group_pick']:.2f}/{row['wape_shuffled_pick']:.2f}  "
              f"hh favoring group pick {row['hh_favoring_group_pick']}/12")

    df = pd.DataFrame(rows)
    out_csv = DATA_DIR / "selection_regret_lcl_companions.csv"
    df.to_csv(out_csv, index=False)

    def med_iqr(a) -> str:
        a = np.asarray(a)
        return (f"median {np.median(a):.4f} "
                f"(IQR {np.percentile(a, 25):.4f} to {np.percentile(a, 75):.4f}), "
                f"range {a.min():.4f} to {a.max():.4f}")

    wape_gap = df["wape_shuffled_pick"] - df["wape_group_pick"]
    macro_gap = df["macro_mae_shuffled_pick"] - df["macro_mae_group_pick"]
    summary = (
        "# Companion quantities across 30 partitions (LCL selection-regret sweep)\n\n"
        "Generated by `examples/selection_regret_lcl_companions.py`. Protocol identical to\n"
        "`selection_regret_lcl_seeds.py`; recomputed deployment MAEs asserted equal to the\n"
        "committed sweep CSV for all 30 seeds. Same statistical caveat applies: the 30\n"
        "partitions are correlated resamples of one fixed 60-household sample, so spreads\n"
        "are descriptive.\n\n"
        "## Deployment MAE minus own CV score (kWh)\n"
        f"- group-aware pick (Ridge a=.01): {med_iqr(df['dep_minus_cv_group_pick'])}, "
        f"positive in {(df['dep_minus_cv_group_pick'] > 0).sum()}/30\n"
        f"- shuffled pick (RF d=None): {med_iqr(df['dep_minus_cv_shuffled_pick'])}, "
        f"positive in {(df['dep_minus_cv_shuffled_pick'] > 0).sum()}/30\n\n"
        "## Deployment WAPE (%)\n"
        f"- group-aware pick: median {np.median(df['wape_group_pick']):.2f}; "
        f"shuffled pick: median {np.median(df['wape_shuffled_pick']):.2f}\n"
        f"- group-aware pick better in {(wape_gap > 0).sum()}/30; "
        f"gap {med_iqr(wape_gap)} points\n\n"
        "## Household-averaged (macro) MAE (kWh)\n"
        f"- group-aware pick better in {(macro_gap > 0).sum()}/30; gap {med_iqr(macro_gap)}\n\n"
        "## Households (of 12) individually favoring the group-aware pick\n"
        f"- median {df['hh_favoring_group_pick'].median():.0f}, "
        f"min {df['hh_favoring_group_pick'].min()}, max {df['hh_favoring_group_pick'].max()}\n\n"
        "## Seed 0 detail\n"
        f"- households favoring the group-aware pick: {seed0['favoring']} of {seed0['n_hh']}\n"
        f"- per-household paired diff (shuffled minus group-aware pick), kWh: "
        f"min {seed0['paired_min']:.4f}, median {seed0['paired_median']:.4f}, "
        f"max {seed0['paired_max']:.4f}\n"
        f"- household-cluster bootstrap 95% interval for the pooled MAE gap: "
        f"{seed0['boot_lo']:.4f} to {seed0['boot_hi']:.4f} kWh "
        f"({BOOT_ITER} resamples, seed {BOOT_SEED})\n"
        f"- deployment-set mean daily consumption: {seed0['dep_mean_kwh']:.2f} kWh\n"
    )
    (DATA_DIR / "selection_regret_lcl_companions_summary.md").write_text(summary)
    print("\n" + summary)
    print(f"csv -> {out_csv}")


if __name__ == "__main__":
    main()
