"""E2E: the controlled competitor-leakage benchmark holds and is deterministic.

``tools/competitor_benchmark.py`` runs the same model through several
libraries' CV splitters on a task whose honest answer is ~0. This pins the
two invariants that make the comparison meaningful, using only the
dependency-free core (sklearn + purgedcv) via ``--core-only``:

* the naive ``sklearn KFold(shuffle=True)`` fabricates a large positive R^2
  with 100% label overlap (the leak the whole field exists to prevent);
* ``purgedcv PurgedKFold`` admits exactly 0.0 overlap and returns no skill.

No third-party competitor is asserted (they may be absent on CI; the tool
isolates and reports them). Skipped when ``tools/`` is absent -- it is
git-ignored, so a clean checkout has no benchmark script.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = REPO_ROOT / "tools" / "competitor_benchmark.py"


def _run(out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BENCHMARK), "--core-only", "--out-dir", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.e2e
@pytest.mark.skipif(not BENCHMARK.exists(), reason="benchmark script missing -- safety guard")
def test_competitor_benchmark_invariants_and_determinism(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    first = _run(out_a)
    assert first.returncode == 0, f"stdout:\n{first.stdout}\nstderr:\n{first.stderr}"
    csv_a = out_a / "competitor_benchmark.csv"
    summary_a = out_a / "competitor_benchmark_summary.md"
    assert csv_a.exists() and summary_a.exists()

    rows = pd.read_csv(csv_a)
    ran = rows[rows["status"] == "ran"]

    naive = ran[ran["splitter"].str.contains("shuffle=True")]
    assert len(naive) == 1
    # Naive shuffled k-fold fabricates skill on pure noise and leaks fully.
    assert float(naive["mean_r2"].iloc[0]) > 0.5
    assert float(naive["mean_overlap"].iloc[0]) == 1.0

    purged = ran[(ran["library"] == "purgedcv") & (ran["splitter"] == "PurgedKFold")]
    assert len(purged) == 1
    # Purged k-fold removes the overlap entirely and shows no skill.
    assert float(purged["mean_overlap"].iloc[0]) == 0.0
    assert float(purged["mean_r2"].iloc[0]) <= 0.0

    # Every purgedcv splitter must reach exactly zero label overlap.
    pcv = ran[ran["library"] == "purgedcv"]
    assert len(pcv) >= 1
    assert (pcv["mean_overlap"] == 0.0).all()

    # Deterministic: a second run yields a byte-identical CSV.
    second = _run(out_b)
    assert second.returncode == 0, f"stdout:\n{second.stdout}\nstderr:\n{second.stderr}"
    assert (out_b / "competitor_benchmark.csv").read_bytes() == csv_a.read_bytes()
