"""Regenerate the two MDPI-paper figures with neutral scheme labels.

The public repo figures (``.github/images``) and the README intentionally use
the "naive / honest" framing as their hook. The MDPI submission, however, uses
the neutral terms "shuffled k-fold" and "group-aware CV" throughout its text and
captions. This script reproduces both paper figures from the committed seed sweep
(``examples/data/selection_regret_lcl_seeds.csv``) with the neutral wording and
writes them into ``paper/figures/`` only, leaving the public figures untouched.

No model selection is re-run: every value is read from the committed CSV, so the
figures stay numerically identical to the ones the manuscript text cites
(seed 0: 6.6% gap; 30/30 partitions; median gap 0.283 kWh).

Run from the repo root or from ``paper/figures/``:

    python paper/figures/make_neutral_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SHUFFLED_COLOR = "#e0564c"
GROUP_COLOR = "#3a9d6e"

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV = REPO_ROOT / "examples" / "data" / "selection_regret_lcl_seeds.csv"
OUT_DIR = REPO_ROOT / "paper" / "figures"


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.tick_params(colors="#666666", labelsize=9)
    ax.grid(axis="y", ls=":", color="#dddddd", alpha=0.8, zorder=0)


def figure_headline(df: pd.DataFrame) -> None:
    """Figure 1: seed-0 deployment MAE for each scheme's selected model."""
    s0 = df[df["seed"] == 0].iloc[0]
    maes = [s0["naive_dep_mae"], s0["honest_dep_mae"]]
    picks = [s0["naive_pick"], s0["honest_pick"]]
    labels = ["shuffled\nk-fold", "group-aware\nCV"]

    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.patch.set_facecolor("white")
    _style_axis(ax)
    bars = ax.bar(labels, maes, color=[SHUFFLED_COLOR, GROUP_COLOR],
                  edgecolor="white", linewidth=1.4, zorder=3)
    for rect, m, name in zip(bars, maes, picks, strict=True):
        ax.text(rect.get_x() + rect.get_width() / 2, m + 0.05,
                f"{m:.3f}\n({name})", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#333333")
    ax.set_ylabel("deployment MAE on 12 held-out households (kWh)",
                  fontsize=10, color="#444444")
    ax.set_ylim(0, max(maes) * 1.25)

    improve_pct = (maes[0] - maes[1]) / maes[0] * 100
    fig.text(0.5, 0.96, "Group-aware CV selects a model that deploys better on new households",
             ha="center", fontsize=12.5, fontweight="bold", color="#222222")
    fig.text(0.5, 0.91,
             f"Same grid, same data, same 12 held-out households; "
             f"the group-aware pick is {improve_pct:.1f}% lower MAE.",
             ha="center", fontsize=9.5, color="#777777")
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.10, right=0.96)

    out = OUT_DIR / "selection_regret_lcl.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


def figure_robust(df: pd.DataFrame) -> None:
    """Figure 2: paired deployment MAE across 30 partitions + gap histogram."""
    n = len(df)
    d = (df["naive_dep_mae"] - df["honest_dep_mae"]).to_numpy()
    win_rate = int((d > 0).sum())
    med = float(np.median(d))
    q1, q3 = float(np.percentile(d, 25)), float(np.percentile(d, 75))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        _style_axis(ax)

    for _, row in df.iterrows():
        ax1.plot([0, 1], [row["naive_dep_mae"], row["honest_dep_mae"]],
                 color="#bbbbbb", lw=0.8, zorder=1)
    ax1.scatter(np.zeros(n), df["naive_dep_mae"], color=SHUFFLED_COLOR, s=28,
                zorder=3, label="shuffled")
    ax1.scatter(np.ones(n), df["honest_dep_mae"], color=GROUP_COLOR, s=28,
                zorder=3, label="group-aware")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["shuffled\nk-fold", "group-aware\nCV"])
    ax1.set_xlim(-0.3, 1.3)
    ax1.set_ylabel("deployment MAE on held-out households (kWh)",
                   fontsize=10, color="#444444")

    ax2.hist(d, bins=12, color=GROUP_COLOR, edgecolor="white", zorder=3)
    ax2.axvline(0, color=SHUFFLED_COLOR, ls="--", lw=1.4, zorder=4)
    ax2.set_xlabel("shuffled minus group-aware deployment MAE (kWh)",
                   fontsize=10, color="#444444")
    ax2.set_ylabel("partitions", fontsize=10, color="#444444")

    fig.text(0.5, 0.97, f"Group-aware CV deploys better in {win_rate} of {n} partitions",
             ha="center", fontsize=13, fontweight="bold", color="#222222")
    fig.text(0.5, 0.92,
             f"median gap {med:.3f} kWh (IQR {q1:.3f} to {q3:.3f}); "
             f"positive means group-aware deploys better",
             ha="center", fontsize=9.5, color="#777777")
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.08, right=0.97, wspace=0.25)

    out = OUT_DIR / "selection_regret_lcl_seeds.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


def main() -> None:
    df = pd.read_csv(CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_headline(df)
    figure_robust(df)


if __name__ == "__main__":
    main()
