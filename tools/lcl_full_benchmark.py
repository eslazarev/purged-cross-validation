"""Full-population LCL benchmark with confidence intervals (offline, gitignored).

The committed notebook runs one 60-household subsample: a single point, no
error bars. This scans the entire raw Low Carbon London corpus
(`/Users/elazarev/Downloads/Small LCL Data`, 168 CSVs, ~168M rows, ~8 GB),
keeps the Standard-tariff households with at least a year of data, then draws
K seeded subsamples and runs the *exact* notebook harness
(`examples/_lcl_harness.py`) on each. The four CV WAPEs and the two leakage
gaps come back as mean +/- 95% t-interval, so the notebook's single-sample
result becomes a measured interval over the real population.

Repo-only: `tools/` is gitignored and the wheel ships only `src/purgedcv`.
Memory-safe: every file is read in chunks; pass 1 keeps one small dict, pass 2
keeps only the chosen households' rows. Deterministic: one `--seed` drives a
`SeedSequence` spawn per subsample and the model's `random_state`, so two runs
produce a byte-identical CSV.

Usage:
    python tools/lcl_full_benchmark.py                 # full Std population
    python tools/lcl_full_benchmark.py --quick         # tiny smoke (CI/e2e)
    python tools/lcl_full_benchmark.py --k 20 --n 300  # explicit
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(EXAMPLES.parent) not in sys.path:
    sys.path.insert(0, str(EXAMPLES.parent))

from examples._lcl_harness import build_features, four_scheme_wape  # noqa: E402

RAW_COLS = ["LCLid", "stdorToU", "DateTime", "energy"]
CHUNK = 2_000_000
H = 12  # forecast horizon (half-hours), identical to the notebook
N_SPLITS = 5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=Path.home() / "Downloads" / "Small LCL Data")
    p.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "examples" / "data"
    )
    p.add_argument("--k", type=int, default=20, help="number of seeded subsamples")
    p.add_argument("--n", type=int, default=300, help="households per subsample")
    p.add_argument("--min-half-hours", type=int, default=48 * 365)
    p.add_argument("--min-span-days", type=int, default=365)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke run (k=2, n=6, ~3-day minimum) for the e2e fixture",
    )
    args = p.parse_args(argv)
    if args.quick:
        args.k = min(args.k, 2)
        args.n = min(args.n, 6)  # >= 5 so GroupKFold(n_splits=5) has enough groups
        args.min_half_hours = min(args.min_half_hours, 200)
        args.min_span_days = min(args.min_span_days, 3)
    return args


def _raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files under {raw_dir}")
    return files


def enumerate_eligible(
    files: list[Path], *, min_half_hours: int, min_span_days: int
) -> tuple[list[str], int]:
    """Pass 1: per-household count, tariff and time span across every file.

    ISO ``YYYY-MM-DD HH:MM:SS`` sorts chronologically as text, so the min/max
    timestamp need no parsing in the hot loop -- only the surviving households'
    spans are parsed once at the end. Returns ``(eligible_ids, total_rows)``.
    """
    count: dict[str, int] = {}
    tmin: dict[str, str] = {}
    tmax: dict[str, str] = {}
    tariff: dict[str, str] = {}
    total = 0
    for f in files:
        for chunk in pd.read_csv(
            f, header=0, names=RAW_COLS, usecols=["LCLid", "stdorToU", "DateTime"], chunksize=CHUNK
        ):
            total += len(chunk)
            ts = chunk["DateTime"].astype(str).str.slice(0, 19)
            for lclid, grp in ts.groupby(chunk["LCLid"]):
                count[lclid] = count.get(lclid, 0) + len(grp)
                lo, hi = grp.min(), grp.max()
                tmin[lclid] = lo if lclid not in tmin else min(tmin[lclid], lo)
                tmax[lclid] = hi if lclid not in tmax else max(tmax[lclid], hi)
            for lclid, tar in zip(chunk["LCLid"], chunk["stdorToU"], strict=True):
                tariff.setdefault(lclid, tar)
    eligible: list[str] = []
    for lclid, c in count.items():
        if tariff.get(lclid) != "Std" or c < min_half_hours:
            continue
        span = pd.Timestamp(tmax[lclid]) - pd.Timestamp(tmin[lclid])
        if span.days >= min_span_days:
            eligible.append(lclid)
    return sorted(eligible), total


def load_subsample(files: list[Path], chosen: set[str]) -> pd.DataFrame:
    """Pass 2: read only the chosen households' rows into a clean panel."""
    parts: list[pd.DataFrame] = []
    for f in files:
        for chunk in pd.read_csv(
            f, header=0, names=RAW_COLS, usecols=["LCLid", "DateTime", "energy"], chunksize=CHUNK
        ):
            sub = chunk[chunk["LCLid"].isin(chosen)]
            if sub.empty:
                continue
            tstp = pd.to_datetime(
                sub["DateTime"].astype(str).str.slice(0, 19), format="%Y-%m-%d %H:%M:%S"
            )
            load = pd.to_numeric(sub["energy"].astype(str).str.strip(), errors="coerce")
            part = pd.DataFrame({"LCLid": sub["LCLid"].to_numpy(), "tstp": tstp, "load": load})
            parts.append(part.dropna(subset=["load"]))
    if not parts:
        raise SystemExit("no rows loaded for the chosen households")
    return pd.concat(parts, ignore_index=True)


def ci95(values: list[float]) -> tuple[float, float, float]:
    """Mean and 95% t-interval half-width (df = n-1). Returns (mean, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, mean, mean
    sd = float(arr.std(ddof=1))
    half = float(student_t.ppf(0.975, df=arr.size - 1)) * sd / np.sqrt(arr.size)
    return mean, mean - half, mean + half


def run(args: argparse.Namespace) -> Path:
    # Bind typed locals so argparse's Any attributes do not leak downstream.
    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    k: int = args.k
    n: int = args.n
    min_half_hours: int = args.min_half_hours
    min_span_days: int = args.min_span_days
    seed: int = args.seed

    t0 = time.perf_counter()
    files = _raw_files(raw_dir)
    eligible, total_rows = enumerate_eligible(
        files, min_half_hours=min_half_hours, min_span_days=min_span_days
    )
    if len(eligible) < n:
        raise SystemExit(f"only {len(eligible)} eligible households, need >= --n ({n})")
    pool = np.array(eligible)
    spawned = np.random.SeedSequence(seed).spawn(k)

    rows: list[dict[str, float]] = []
    for j in range(k):
        rng = np.random.default_rng(spawned[j])
        chosen = set(rng.choice(pool, size=n, replace=False).tolist())
        panel = load_subsample(files, chosen)
        features, y, pred, evalu, groups = build_features(panel, h=H)
        wape = four_scheme_wape(features, y, pred, evalu, groups, seed=seed, n_splits=N_SPLITS, h=H)
        naive = wape["naive shuffled k-fold"]
        blocked = wape["blocked k-fold"]
        walk = wape["WalkForwardSplit"]
        grp = wape["GroupKFold (household)"]
        rows.append(
            {
                "subsample": float(j),
                "n_households": float(len(chosen)),
                "n_rows": float(len(y)),
                "naive_shuffled_kfold": naive,
                "blocked_kfold": blocked,
                "walkforward": walk,
                "groupkfold_household": grp,
                "temporal_gap_pts": walk - naive,
                "temporal_gap_rel_pct": (walk - naive) / walk * 100.0,
                "household_gap_pts": grp - walk,
                "household_gap_rel_pct": (grp - walk) / walk * 100.0,
            }
        )
        print(
            f"  subsample {j + 1}/{k}: naive={naive:.2f} blocked={blocked:.2f} "
            f"walk={walk:.2f} group={grp:.2f}  ({len(y):,} rows)",
            flush=True,
        )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lcl_full_benchmark.csv"
    df.to_csv(csv_path, index=False)

    elapsed = time.perf_counter() - t0
    metrics = [
        ("naive shuffled k-fold", "naive_shuffled_kfold"),
        ("blocked k-fold", "blocked_kfold"),
        ("WalkForwardSplit", "walkforward"),
        ("GroupKFold (household)", "groupkfold_household"),
        ("temporal gap (walk - naive, WAPE pts)", "temporal_gap_pts"),
        ("temporal gap (relative %)", "temporal_gap_rel_pct"),
        ("household gap (group - walk, WAPE pts)", "household_gap_pts"),
        ("household gap (relative %)", "household_gap_rel_pct"),
    ]
    lines = [
        "# LCL full-population benchmark",
        "",
        f"- generated: {datetime.now(timezone.utc).isoformat()}",
        f"- raw dir: `{raw_dir}`",
        f"- eligible Std households (>= {min_half_hours} half-hours, "
        f">= {min_span_days} days): **{len(eligible)}**",
        f"- subsamples K = {k}, households per subsample N = {n}, seed = {seed}",
        f"- raw rows scanned: {total_rows:,}",
        f"- elapsed: {elapsed / 60:.1f} min",
        f"- versions: purgedcv {version('purgedcv')}, numpy {np.__version__}, "
        f"pandas {pd.__version__}, scikit-learn {version('scikit-learn')}, "
        f"scipy {version('scipy')}",
        "",
        "| metric | mean | 95% CI low | 95% CI high |",
        "|---|---|---|---|",
    ]
    for label, col in metrics:
        mean, lo, hi = ci95(df[col].tolist())
        lines.append(f"| {label} | {mean:.2f} | {lo:.2f} | {hi:.2f} |")
    lines += [
        "",
        "WAPE = sum|err| / sum|actual|, in percent. Temporal gap > 0 means naive",
        "shuffled CV looks better than honest walk-forward. Household gap > 0 means",
        "scoring on unseen households is worse than the pooled temporal estimate.",
        f"Reproduce: `python tools/lcl_full_benchmark.py --k {k} --n {n} --seed {seed}`.",
        "",
    ]
    summary_path = out_dir / "lcl_full_benchmark_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {csv_path}\nwrote {summary_path}\n({elapsed / 60:.1f} min)")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
