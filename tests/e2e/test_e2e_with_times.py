"""A user can run the public with_times example in a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.e2e
def test_with_times_documented_example(tmp_path: Path) -> None:
    """Execute the actual docstring, including changed folds and copy semantics."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            textwrap.dedent("""\
                import doctest
                from purgedcv import BaseTemporalSplitter

                example = doctest.DocTestParser().get_doctest(
                    BaseTemporalSplitter.with_times.__doc__,
                    {}, "with_times", None, 0,
                )
                assert example.examples, "with_times must include a runnable example"
                result = doctest.DocTestRunner().run(example)
                assert result.failed == 0, result
                assert result.attempted == len(example.examples)
                print("with_times example passed")
                """),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert result.stdout.strip() == "with_times example passed"
