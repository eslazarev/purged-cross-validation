"""User stories: an AI agent or a reader copies the guidance pages and the skill.

Each page runs top to bottom in a fresh Python process, the error messages the
pages quote must be the ones the package raises, and the plugin manifests and
llms.txt must point at files that exist. No network access.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import purgedcv

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL = REPO_ROOT / "skills" / "purgedcv" / "SKILL.md"
CHOOSING = REPO_ROOT / "docs" / "choosing-a-splitter.md"
MISTAKES = REPO_ROOT / "docs" / "common-mistakes.md"
pytestmark = pytest.mark.e2e

_RUNNER = r"""
import json
import sys

path, snippets, checks = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
namespace = {"__name__": "__main__"}
for line, source in snippets:
    # Keep traceback line numbers aligned with the Markdown file.
    exec(compile("\n" * (line - 1) + source, path, "exec"), namespace)
exec(checks, namespace)
"""


def _python_blocks(path: Path) -> list[tuple[int, str]]:
    markdown = path.read_text(encoding="utf-8")
    blocks = []
    for match in re.finditer(r"(?ms)^```python[ \t]*\n(.*?)^```[ \t]*$", markdown):
        blocks.append((markdown[: match.start(1)].count("\n") + 1, match.group(1)))
    assert blocks, f"{path.name} has no Python examples"
    return blocks


def _run_page(path: Path, checks: str, tmp_path: Path) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(path), json.dumps(_python_blocks(path)), checks],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_user_can_run_choosing_a_splitter(tmp_path: Path) -> None:
    _run_page(
        CHOOSING,
        """
assert scores.shape == (5,)
assert report["train_nonempty"].all()
assert (report["final_overlap_fraction"] == 0).all()
# PBO measures selection, not edge: both sets sit near 0.5 and only DSR separates them.
pbo_noise, dsr_noise = pbo_and_dsr(noise)
pbo_edge, dsr_edge = pbo_and_dsr(edge)
assert 0.3 <= pbo_noise <= 0.7 and 0.3 <= pbo_edge <= 0.7
assert dsr_noise < 0.95 < dsr_edge
""",
        tmp_path,
    )


def test_common_mistakes_quote_the_real_errors(tmp_path: Path) -> None:
    stdout = _run_page(
        MISTAKES,
        """
report = audit_splitter(by_asset, X_panel)
assert (report["final_train_size"] == 0).all()
assert not report["train_nonempty"].any()
assert (prices["label_end"] > prices["date"]).all()
row = prices.loc[prices["date"] == "2024-03-28"].iloc[0]
assert row["label_end"] == pd.Timestamp("2024-04-08")
assert row["calendar_end"] == pd.Timestamp("2024-04-04")
""",
        tmp_path,
    )
    quoted = re.findall(r"print\(err\)  # (.+?)(?: \.\.\.)?$", MISTAKES.read_text(), re.M)
    assert len(quoted) == 4
    for message in quoted:
        assert message in stdout, f"docs quote {message!r}, package printed:\n{stdout}"


def test_agent_can_run_every_skill_example(tmp_path: Path) -> None:
    _run_page(
        SKILL,
        """
assert scores.shape == (5,)
assert report["train_nonempty"].all()
assert 0.0 <= pbo <= 1.0
assert 0.0 <= dsr <= 1.0
assert years_needed > 0
""",
        tmp_path,
    )


def test_skill_label_recipe_survives_shuffled_rows() -> None:
    import numpy as np
    import pandas as pd

    recipe = re.search(r"(?ms)^```text\n(df = df\.sort_values.*?)^```", SKILL.read_text())
    assert recipe, "SKILL.md should show the label_end recipe"
    days = pd.bdate_range("2024-01-01", periods=30)
    ordered = pd.DataFrame(
        [(d, a) for d in days for a in ("AAA", "BBB")], columns=["date", "asset"]
    )
    shuffled = ordered.sample(frac=1, random_state=0).reset_index(drop=True)
    namespace: dict[str, object] = {"df": shuffled, "h": 5}
    exec(recipe.group(1), namespace)
    result = namespace["df"]
    assert isinstance(result, pd.DataFrame)
    assert (result["label_end"] > result["date"]).all()
    assert result["date"].is_monotonic_increasing
    position = {d: i for i, d in enumerate(days)}
    assert all(
        position[e] - position[d] == 5
        for d, e in zip(result["date"], result["label_end"], strict=True)
    )
    assert np.array_equal(result["asset"].unique(), ["AAA", "BBB"])


def test_time_series_split_gap_matches_the_docs() -> None:
    import numpy as np
    from sklearn.model_selection import TimeSeriesSplit

    # The docs say gap=h is enough for one series with a fixed h-row horizon.
    h, rows = 5, np.arange(200)
    for gap, expect_overlap in ((0, True), (h, False)):
        for train, test in TimeSeriesSplit(n_splits=5, gap=gap).split(rows):
            overlap = bool(((rows[train] + h) > rows[test].min()).any())
            assert overlap is expect_overlap
    assert "gap=h" in CHOOSING.read_text() and "TimeSeriesSplit(gap=h)" in SKILL.read_text()


def test_skill_names_only_real_api_and_real_errors() -> None:
    text = SKILL.read_text(encoding="utf-8")
    frontmatter = re.match(r"(?s)^---\n(.*?)\n---\n", text)
    assert frontmatter, "SKILL.md needs YAML frontmatter"
    fields = dict(re.findall(r"(?m)^(\w+): (.+)$", frontmatter.group(1)))
    assert fields["name"] == SKILL.parent.name == "purgedcv"
    assert re.fullmatch(r"[a-z0-9-]{1,64}", fields["name"])
    assert 0 < len(fields["description"]) <= 1024

    imported: set[str] = set()
    for names in re.findall(r"from purgedcv import ([\w, ]+)", text):
        imported.update(name.strip() for name in names.split(","))
    assert imported, "the skill should import from purgedcv"
    assert imported <= set(purgedcv.__all__)

    for message in re.findall(r"^- `([^`]+)`: ", text, re.M):
        if message.startswith(("prediction_times", "evaluation_times", "X length", "embargo,")):
            stem = re.split(r" index i| N | M\.", message)[0]
            assert stem in MISTAKES.read_text(), f"skill error {message!r} is not documented"
    for name in ("TemporalLeakageError", "EmbargoViolationError", "GroupLeakageError"):
        assert name in text
        assert hasattr(purgedcv, name) or hasattr(purgedcv.exceptions, name)


def test_plugin_manifests_install_the_skill() -> None:
    plugin = json.loads((REPO_ROOT / ".claude-plugin" / "plugin.json").read_text())
    marketplace = json.loads((REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text())
    (entry,) = marketplace["plugins"]
    assert plugin["name"] == entry["name"] == "purgedcv"
    assert entry["source"] == "./"
    assert plugin["description"] == entry["description"]
    assert plugin["license"] == "MIT"
    assert (REPO_ROOT / "skills" / plugin["name"] / "SKILL.md").is_file()
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "/plugin install purgedcv --marketplace eslazarev/purged-cross-validation" in readme


def test_context7_reads_the_docs_and_the_skill() -> None:
    config = json.loads((REPO_ROOT / "context7.json").read_text())
    assert {"docs", "skills"} <= set(config["folders"])
    assert "docs" not in config["excludeFolders"]
    for folder in config["folders"]:
        assert (REPO_ROOT / folder).is_dir(), folder


def test_docs_site_publishes_llms_txt(tmp_path: Path) -> None:
    pytest.importorskip("mkdocs", reason="llms.txt test requires the docs extra")
    pytest.importorskip("mkdocs_llmstxt", reason="llms.txt test requires the docs extra")
    site = tmp_path / "site"
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(site)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    index = (site / "llms.txt").read_text(encoding="utf-8")
    for page in ("quickstart", "choosing-a-splitter", "common-mistakes", "api/splitters"):
        assert f"/{page}/index.md" in index
        assert (site / page / "index.md").is_file()
    full = (site / "llms-full.txt").read_text(encoding="utf-8")
    assert "Grouping a date-sorted panel by asset" in full
    assert "PurgedGroupKFold" in full
