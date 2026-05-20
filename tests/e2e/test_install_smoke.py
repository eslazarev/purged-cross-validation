"""End-to-end smoke test: the installed package is importable from a fresh interpreter.

This verifies that ``pip install -e ".[dev]"`` actually exposes the package
on the import path, with no hidden import-time side effects, and that the
top-level ``__version__`` attribute is present and well-formed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.e2e
def test_package_importable_in_subprocess() -> None:
    import purgedcv

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import purgedcv; print(purgedcv.__version__)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # Subprocess must agree with the parent's imported version, whatever it is.
    # Hardcoding a literal here would break the next CI version bump.
    assert result.stdout.strip() == purgedcv.__version__
    assert result.stderr == ""


@pytest.mark.e2e
def test_version_is_well_formed_string() -> None:
    """User Story: a release-tools script needs to parse the version string."""
    import purgedcv

    version = purgedcv.__version__
    assert isinstance(version, str)
    assert len(version) > 0
    # PEP 440 alpha pre-release looks like "0.1.0a0".
    major_minor_patch = version.split("a")[0]
    parts = major_minor_patch.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


@pytest.mark.e2e
def test_packaging_metadata_versions_match_runtime() -> None:
    """Fresh installs, package metadata, and runtime imports must agree."""
    import purgedcv

    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'(?m)^version = "([^"]+)"$', pyproject)
    assert match is not None

    assert match.group(1) == purgedcv.__version__
    assert version("purgedcv") == purgedcv.__version__
