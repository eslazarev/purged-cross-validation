"""End-to-end test: CITATION.cff stays in step with the package version.

These tests subprocess out to ``tools/sync_citation.py`` exactly the way CI
and the release workflow invoke it (``python tools/sync_citation.py ...``).
They cover the two user stories the script exists for: a CI drift guard that
fails when CITATION.cff falls behind pyproject.toml, and a release step that
rewrites the version (and date) in place.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "sync_citation.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _sandbox(tmp_path: Path, cff_version: str, cff_date: str = "2026-06-05") -> Path:
    """A throwaway repo root with a real pyproject and a chosen CITATION version."""
    shutil.copy(REPO_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    citation = REPO_ROOT / "CITATION.cff"
    text = citation.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^version:[^\n]*$", f"version: {cff_version}", text, count=1)
    text = re.sub(r"(?m)^date-released:[^\n]*$", f'date-released: "{cff_date}"', text, count=1)
    (tmp_path / "CITATION.cff").write_text(text, encoding="utf-8")
    return tmp_path


@pytest.mark.e2e
def test_repo_citation_is_in_sync() -> None:
    """The committed CITATION.cff must match pyproject.toml (the CI guard)."""
    result = _run("--check")
    assert result.returncode == 0, f"drift detected:\n{result.stdout}\n{result.stderr}"


@pytest.mark.e2e
def test_check_fails_on_drift(tmp_path: Path) -> None:
    root = _sandbox(tmp_path, cff_version="0.0.1")
    result = _run("--check", "--root", str(root))
    assert result.returncode == 1
    assert "disagrees with pyproject.toml" in result.stderr


@pytest.mark.e2e
def test_write_brings_into_sync(tmp_path: Path) -> None:
    root = _sandbox(tmp_path, cff_version="0.0.1")

    written = _run("--write", "--root", str(root))
    assert written.returncode == 0
    assert "updated to version" in written.stdout

    checked = _run("--check", "--root", str(root))
    assert checked.returncode == 0

    # Only the version line changed; other metadata is preserved.
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    assert "given-names: Evgenii" in cff
    assert 'date-released: "2026-06-05"' in cff


@pytest.mark.e2e
def test_write_with_date_updates_release_date(tmp_path: Path) -> None:
    root = _sandbox(tmp_path, cff_version="0.0.1", cff_date="2020-01-01")
    result = _run("--write", "--date", "2026-06-14", "--root", str(root))
    assert result.returncode == 0
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    assert 'date-released: "2026-06-14"' in cff


@pytest.mark.e2e
def test_bad_date_is_rejected(tmp_path: Path) -> None:
    root = _sandbox(tmp_path, cff_version="0.0.1")
    result = _run("--write", "--date", "14-06-2026", "--root", str(root))
    assert result.returncode != 0
    assert "YYYY-MM-DD" in result.stderr
