"""E2E: the shared LCL harness and the full-population benchmark CLI.

Mirrors ``examples/uk_smart_meter_lcl.ipynb`` at the library level (notebooks
are docs here, not run under pytest). Two guarantees:

1. ``examples/_lcl_harness.py`` -- the verbatim extract the offline benchmark
   runs -- gives a finite WAPE under all four CV schemes and is deterministic
   for a fixed seed. No magnitude is asserted: the real gap estimates are
   data-specific and produced only by the offline run over the raw corpus,
   never baked into a test.
2. ``tools/lcl_full_benchmark.py --quick`` runs end to end against a tiny
   synthetic fixture in the *real* raw LCL CSV layout, writes both outputs,
   and the per-subsample CSV is byte-identical across two runs (determinism).
   Skipped cleanly when ``tools/`` is absent (a clean checkout has no
   ``tools/lcl_full_benchmark.py`` is missing (safety guard).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from examples._lcl_harness import build_features, four_scheme_wape

SEED = 0
H = 12
N_SPLITS = 5
HALF_HOURS = 48
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "tools" / "lcl_full_benchmark.py"
RAW_HEADER = "LCLid,stdorToU,DateTime,KWH/hh (per half hour) "


def synth_panel(n_households: int, days: int, seed: int) -> pd.DataFrame:
    """LCL-structured half-hourly panel: daily + weekly + annual + AR noise.

    Identical generator to the notebook's synthetic fallback, so the harness
    sees the same shape it documents. Real seasonality, real noise floor --
    enough that WAPE is finite and positive without being engineered.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2013-01-01", periods=days * HALF_HOURS, freq="30min")
    slot = np.arange(HALF_HOURS)
    daily = 0.35 + 0.9 * np.exp(-((slot - 15) ** 2) / 8) + 1.3 * np.exp(-((slot - 38) ** 2) / 10)
    dow = idx.dayofweek.to_numpy()
    weekly = np.where(dow >= 5, 1.25, 1.0)
    doy = idx.dayofyear.to_numpy()
    annual = 1.0 + 0.15 * np.cos(2 * np.pi * doy / 365.0)
    hh_slot = idx.hour.to_numpy() * 2 + (idx.minute.to_numpy() // 30)
    frames: list[pd.DataFrame] = []
    for h in range(n_households):
        base = rng.lognormal(mean=0.0, sigma=0.4)
        signal = base * daily[hh_slot] * weekly * annual
        ar = np.zeros(len(idx))
        e = rng.normal(0, 0.08, len(idx))
        for t in range(1, len(idx)):
            ar[t] = 0.6 * ar[t - 1] + e[t]
        load = np.maximum(0.05, signal * (1.0 + ar))
        frames.append(pd.DataFrame({"LCLid": f"MAC{h:05d}", "tstp": idx, "load": load}))
    return pd.concat(frames, ignore_index=True)


def test_build_features_target_is_forward_horizon_mean() -> None:
    idx = pd.date_range("2013-01-01", periods=400, freq="30min")
    panel = pd.DataFrame(
        {
            "LCLid": "MAC00000",
            "tstp": idx,
            "load": np.arange(len(idx), dtype=float),
        }
    )

    _, y, pred, _, _ = build_features(panel, h=4)
    first_t = int((pred.iloc[0] - idx[0]) / pd.Timedelta(minutes=30))

    assert first_t == 336
    assert y[0] == np.mean([337.0, 338.0, 339.0, 340.0])


@pytest.mark.e2e
def test_harness_four_schemes_finite_and_deterministic() -> None:
    panel = synth_panel(n_households=8, days=16, seed=SEED)
    features, y, pred, evalu, groups = build_features(panel, h=H)

    first = four_scheme_wape(features, y, pred, evalu, groups, seed=SEED, n_splits=N_SPLITS, h=H)
    assert set(first) == {
        "naive shuffled k-fold",
        "blocked k-fold",
        "WalkForwardSplit",
        "GroupKFold (household)",
    }
    for name, wape in first.items():
        assert np.isfinite(wape), name
        assert 0.0 < wape < 100.0, (name, wape)

    # Same seed, same data -> identical results (the benchmark relies on this).
    second = four_scheme_wape(features, y, pred, evalu, groups, seed=SEED, n_splits=N_SPLITS, h=H)
    assert first == second

    # Both leakage gaps the benchmark reports must be finite real numbers.
    temporal_gap = first["WalkForwardSplit"] - first["naive shuffled k-fold"]
    household_gap = first["GroupKFold (household)"] - first["WalkForwardSplit"]
    assert np.isfinite(temporal_gap)
    assert np.isfinite(household_gap)


def _write_raw_fixture(raw_dir: Path, n_households: int, days: int, seed: int) -> None:
    """Write the synthetic panel in the real raw LCL CSV layout.

    Exercises the tool's actual parser: the trailing-space header, the
    ``...SS.0000000`` timestamps, whitespace-padded values and a ``Null``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    panel = synth_panel(n_households=n_households, days=days, seed=seed)
    ts = panel["tstp"].dt.strftime("%Y-%m-%d %H:%M:%S.0000000")
    energy = [f" {v:.4f} " for v in panel["load"]]
    energy[1] = " Null "  # the parser must coerce-and-drop this
    body = pd.DataFrame(
        {
            "LCLid": panel["LCLid"],
            "stdorToU": "Std",
            "DateTime": ts,
            "energy": energy,
        }
    )
    # Split across two files: a household must be reassembled across files.
    half = len(body) // 2
    for i, part in enumerate((body.iloc[:half], body.iloc[half:])):
        path = raw_dir / f"LCL-fixture_{i}.csv"
        path.write_text(
            RAW_HEADER + "\n" + part.to_csv(index=False, header=False),
            encoding="utf-8",
        )


@pytest.mark.e2e
@pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark script missing -- safety guard")
def test_full_benchmark_cli_quick_runs_and_is_deterministic(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    _write_raw_fixture(raw_dir, n_households=8, days=12, seed=SEED)

    def invoke(out_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(BENCHMARK),
                "--quick",
                "--raw-dir",
                str(raw_dir),
                "--out-dir",
                str(out_dir),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    first = invoke(out_a)
    assert first.returncode == 0, f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}"
    csv_a = out_a / "lcl_full_benchmark.csv"
    summary_a = out_a / "lcl_full_benchmark_summary.md"
    assert csv_a.exists() and summary_a.exists()

    rows = pd.read_csv(csv_a)
    assert len(rows) == 2  # --quick => K = 2 subsamples
    for col in (
        "naive_shuffled_kfold",
        "blocked_kfold",
        "walkforward",
        "groupkfold_household",
        "temporal_gap_pts",
        "household_gap_pts",
    ):
        assert col in rows.columns
        assert np.isfinite(rows[col]).all(), col

    summary_text = summary_a.read_text(encoding="utf-8")
    assert "95% CI low" in summary_text
    assert "eligible Std households" in summary_text

    # Determinism: a second independent run yields a byte-identical CSV
    # (the summary carries a wall-clock timestamp, so only the CSV is checked).
    second = invoke(out_b)
    assert second.returncode == 0, f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"
    assert (out_b / "lcl_full_benchmark.csv").read_bytes() == csv_a.read_bytes()
