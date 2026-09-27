"""User stories: copy each Quickstart chapter into a fresh Python process.

Execute the Markdown itself, including the smaller follow-on examples.
No copied snippets or network access: regressions must fail where users fail.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = REPO_ROOT / "docs" / "quickstart.md"
pytestmark = pytest.mark.e2e


def _chapters() -> dict[int, list[tuple[int, str]]]:
    markdown = QUICKSTART.read_text(encoding="utf-8")
    headings = list(re.finditer(r"(?m)^## (\d+)\. .+$", markdown))
    chapters: dict[int, list[tuple[int, str]]] = {}
    for match in re.finditer(r"(?ms)^```python[ \t]*\n(.*?)^```[ \t]*$", markdown):
        preceding = [heading for heading in headings if heading.start() < match.start()]
        assert preceding, "Every Python example must belong to a runnable chapter"
        chapter = int(preceding[-1].group(1))
        line = markdown[: match.start(1)].count("\n") + 1
        chapters.setdefault(chapter, []).append((line, match.group(1)))
    assert set(chapters) == {1, 2, 3}, "Keep all Quickstart chapters covered"
    return chapters


@pytest.mark.parametrize(("chapter", "snippets"), list(_chapters().items()))
def test_user_can_run_quickstart_chapter(
    chapter: int, snippets: list[tuple[int, str]], tmp_path: Path
) -> None:
    script = r"""
import json
import sys

chapter = int(sys.argv[1])
snippets = json.loads(sys.argv[2])
namespace = {"__name__": "__main__"}
for line, source in snippets:
    # Keep traceback line numbers aligned with the actual Markdown.
    exec(compile("\n" * (line - 1) + source, "docs/quickstart.md", "exec"), namespace)

import numpy as np
from purgedcv import probabilistic_sharpe_ratio

if chapter == 1:
    assert 0 < len(namespace["train_kept"]) < len(namespace["train_idx"])
    assert not np.intersect1d(namespace["train_kept"], namespace["test_idx"]).size
elif chapter == 2:
    report = namespace["report"]
    assert len(report) == 5
    assert report["train_nonempty"].all()
    assert report["temporal_leakage_free"].all()
    assert (report["final_overlap_fraction"] == 0).all()
    for name in ("scores", "scores_w"):
        assert namespace[name].shape == (5,)
        assert np.isfinite(namespace[name]).all()
elif chapter == 3:
    paths = namespace["paths"]
    assert paths.shape == (5, 800)
    assert np.isfinite(paths).all()
    trials = namespace["trial_returns"]
    assert trials.ndim == 2
    assert namespace["n_trials"] == len(trials)
    assert namespace["n_trials"] != len(paths)
    expected_sharpes = trials.mean(axis=1) / trials.std(axis=1, ddof=1)
    np.testing.assert_allclose(namespace["trial_sharpes"], expected_sharpes)
    assert namespace["best_trial"] == int(np.argmax(expected_sharpes))
    np.testing.assert_array_equal(namespace["selected_returns"], trials[namespace["best_trial"]])
    assert namespace["var_sharpe"] == float(expected_sharpes.var(ddof=1))
    assert 0 <= namespace["dsr"] <= 1
    assert namespace["dsr"] <= probabilistic_sharpe_ratio(namespace["selected_returns"], 0.0)
print("QUICKSTART_OK")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(chapter), json.dumps(snippets)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, f"Chapter {chapter} failed:\n{result.stdout}\n{result.stderr}"
    assert result.stdout.rstrip().endswith("QUICKSTART_OK")
