"""The documented contributor prose check runs with the development extra."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_contributor_can_run_prose_gate() -> None:
    """A dev install must include everything needed by the documented command."""
    result = subprocess.run(
        [sys.executable, "tools/prose_gate.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "RESULT: all files clear of FAIL" in result.stdout
