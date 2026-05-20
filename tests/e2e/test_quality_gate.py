"""End-to-end test: project lint and type-check gates pass on the whole tree.

These tests subprocess out to ``black``, ``ruff``, and ``mypy`` invoked as
``python -m <tool>``, exactly the way a CI job would (and using the same
interpreter that runs the test suite). The test suite itself catches
regressions in code style or type discipline. If a contributor breaks
``black``/``ruff`` rules or introduces a mypy error and skips pre-commit,
pytest still flags it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_black_format_clean() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "black",
            "--check",
            "src",
            "tests",
            "tools",
            "examples/_lcl_harness.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"black format check failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.e2e
def test_ruff_check_clean() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"ruff check failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.mark.e2e
def test_mypy_strict_clean() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "src", "tests"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert (
        result.returncode == 0
    ), f"mypy failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
