"""Rebuild the cached 60-household LCL extract from the public dataset.

The selection-regret experiments
(``selection_regret_lcl_seeds.py``, ``selection_regret_lcl_targetenc.py``) read a
cached half-hourly extract at ``examples/data/lcl_halfhourly.csv`` with columns
``LCLid, tstp, energy_kwh``. That file is ~68 MB and is git-ignored, so it is not
redistributed. This script rebuilds *that exact extract* from the public Low
Carbon London dataset, using the committed household manifest
``examples/lcl_household_manifest.csv`` (the 60 ``LCLid`` values), so every
reported number is reproducible from public data alone.

Public data
-----------
Download "SmartMeter Energy Consumption Data in London Households (Low Carbon
London)" from the London Datastore:

    https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households

You may use either layout:
  * the single combined file ``CC_LCL-FullData.csv``; or
  * the directory of block files ``Power-Networks-LCL-June2015(withAcornGps)v2_*.csv``.

The public files carry columns ``LCLid``, ``stdorToU``, ``DateTime``, and
``KWH/hh (per half hour)`` (plus Acorn fields we ignore). The half-hourly reading
column contains occasional ``Null`` strings; those become NaN and are skipped by
the daily aggregation in ``load_daily``, so they do not affect any reported total.

Usage
-----
    python examples/build_lcl_cache.py --source /path/to/CC_LCL-FullData.csv
    python examples/build_lcl_cache.py --source /path/to/lcl_block_dir/
    python examples/build_lcl_cache.py --source ... --verify   # check against an
                                                               # existing cache

The output is written to ``examples/data/lcl_halfhourly.csv`` (created if needed).
``--verify`` additionally compares the rebuilt daily-per-household totals over all
dates (so both the pre-2013 prior data and the in-window feature data are checked)
against an existing cache, and exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
MANIFEST = REPO / "lcl_household_manifest.csv"
OUT = REPO / "data" / "lcl_halfhourly.csv"
CHUNK = 2_000_000


def _detect(cols: list[str]) -> tuple[str, str, str]:
    """Return (id_col, time_col, kwh_col) for raw-public or already-renamed files."""
    low = {c.lower().strip(): c for c in cols}
    id_col = low.get("lclid")
    time_col = low.get("datetime") or low.get("tstp")
    kwh_col = next((c for c in cols if "kwh" in c.lower() or c.lower() == "energy_kwh"), None)
    if not (id_col and time_col and kwh_col):
        raise ValueError(f"could not locate LCLid/DateTime/KWH columns in {cols}")
    return id_col, time_col, kwh_col


def _iter_sources(source: Path):
    if source.is_dir():
        files = sorted(source.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no CSV files under {source}")
        yield from files
    else:
        yield source


def build(source: Path, manifest: Path, out: Path) -> pd.DataFrame:
    ids = set(pd.read_csv(manifest)["LCLid"])
    if len(ids) != 60:
        raise ValueError(f"manifest must list 60 households, found {len(ids)}")
    parts: list[pd.DataFrame] = []
    for path in _iter_sources(source):
        for chunk in pd.read_csv(path, chunksize=CHUNK, dtype=str):
            id_col, time_col, kwh_col = _detect(list(chunk.columns))
            sub = chunk[chunk[id_col].isin(ids)]
            if sub.empty:
                continue
            parts.append(
                pd.DataFrame(
                    {
                        "LCLid": sub[id_col].to_numpy(),
                        "tstp": pd.to_datetime(sub[time_col], errors="coerce"),
                        "energy_kwh": pd.to_numeric(sub[kwh_col], errors="coerce"),
                    }
                )
            )
    if not parts:
        raise RuntimeError("no rows matched the manifest; check the --source path/schema")
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["tstp"]).sort_values(["LCLid", "tstp"]).reset_index(drop=True)
    found = df["LCLid"].nunique()
    if found != 60:
        raise RuntimeError(f"matched {found}/60 manifest households; the source is "
                           "incomplete, so the rebuilt extract would not reproduce the experiment")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df):,} rows, {found} households)")
    return df


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = d["tstp"].dt.normalize()
    return (
        d.groupby(["LCLid", "date"])["energy_kwh"].sum().reset_index()
        .sort_values(["LCLid", "date"]).reset_index(drop=True)
    )


def verify(rebuilt: pd.DataFrame, reference: Path) -> bool:
    """Compare rebuilt daily totals against the reference over *all* dates, so both
    the pre-2013 prior data and the in-window feature data are checked."""
    ref = pd.read_csv(reference, parse_dates=["tstp"])
    a, b = _daily(rebuilt), _daily(ref)
    ok = (
        a.shape == b.shape
        and a["LCLid"].equals(b["LCLid"]) and a["date"].equals(b["date"])
        and np.allclose(a["energy_kwh"].to_numpy(), b["energy_kwh"].to_numpy(), atol=1e-9)
    )
    print("VERIFY OK: rebuilt daily totals (pre-2013 and in-window) match the reference cache exactly"
          if ok else "VERIFY FAIL: rebuilt daily totals differ from the reference cache")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path,
                    help="public LCL FullData CSV or directory of block CSVs")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--verify", action="store_true",
                    help="compare rebuilt daily totals (pre-2013 and in-window) against --out "
                         "if it already exists; exit non-zero on mismatch")
    args = ap.parse_args()
    existing = args.out if args.out.exists() else None
    df = build(args.source, args.manifest, args.out if not (args.verify and existing) else
               args.out.with_suffix(".rebuilt.csv"))
    if args.verify and existing and not verify(df, existing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
