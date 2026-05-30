"""End-to-end test for the Optuna + DSR cookbook example.

Skipped unless the optional ``optuna`` extra is installed. When it is, the
cookbook is run as a subprocess exactly as a user would (`python
examples/optuna_dsr_cookbook.py`) and its output is checked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("optuna")

COOKBOOK = Path(__file__).resolve().parents[2] / "examples" / "optuna_dsr_cookbook.py"


@pytest.mark.e2e
def test_optuna_dsr_cookbook_runs() -> None:
    assert COOKBOOK.is_file()
    result = subprocess.run(
        [sys.executable, str(COOKBOOK)],
        capture_output=True,
        text=True,
        check=True,
    )
    out = result.stdout
    # The cookbook reports the deflation, including the effective trial count
    # that is strictly below the raw count for a correlated TPE search.
    assert "DSR" in out
    assert "effective trials" in out
    raw = int(next(line for line in out.splitlines() if "raw trials" in line).split(":")[1])
    effective = int(
        next(line for line in out.splitlines() if "effective trials" in line).split(":")[1]
    )
    assert 1 <= effective <= raw
