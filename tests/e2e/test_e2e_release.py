"""Exercise release planning in disposable Git repos, with no remote or uploads."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "release.py"
VERSION_FILES = ("pyproject.toml", "src/purgedcv/__init__.py", "CITATION.cff")
pytestmark = pytest.mark.e2e


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _commit(root: Path) -> str:
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return _git(root, "rev-parse", "HEAD")


def _versions(root: Path, version: str) -> None:
    (root / "src/purgedcv").mkdir(parents=True, exist_ok=True)
    (root / VERSION_FILES[0]).write_text(f'[project]\nversion = "{version}"\n')
    (root / VERSION_FILES[1]).write_text(f'__version__ = "{version}"\n')
    (root / VERSION_FILES[2]).write_text(
        f'cff-version: 1.2.0\nversion: {version}\ndate-released: "2026-01-01"\n'
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Release test")
    _git(tmp_path, "config", "user.email", "release-test@example.invalid")
    _versions(tmp_path, "0.1.7")
    _commit(tmp_path)
    _git(tmp_path, "tag", "v0.1.7")
    return tmp_path


def _plan(root: Path, before: str, after: str) -> dict[str, str]:
    result = _run(root, "plan", "--before", before, "--after", after)
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_package_change_prepares_pr_and_only_merged_bump_publishes(repo: Path) -> None:
    released = _git(repo, "rev-parse", "HEAD")
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    code_commit = _commit(repo)
    assert _plan(repo, released, code_commit) == {"mode": "prepare", "version": "0.1.8"}
    assert _git(repo, "status", "--porcelain") == ""

    _git(repo, "switch", "--no-track", "-c", "release/v0.1.8")
    result = _run(repo, "prepare", "--version", "0.1.8", "--date", "2026-09-26")
    assert result.returncode == 0, result.stderr
    assert set(_git(repo, "diff", "--name-only").splitlines()) == set(VERSION_FILES)
    assert 'date-released: "2026-09-26"' in (repo / "CITATION.cff").read_text()
    _commit(repo)
    _git(repo, "switch", "main")
    _git(repo, "merge", "--no-ff", "release/v0.1.8", "-m", "Merge release PR")
    merge = _git(repo, "rev-parse", "HEAD")
    assert _plan(repo, code_commit, merge) == {"mode": "publish", "version": "0.1.8"}

    # A retry after GitHub Release succeeds but PyPI fails must not bump again.
    _git(repo, "tag", "-a", "v0.1.8", "-m", "release", merge)
    assert _plan(repo, code_commit, merge) == {"mode": "publish", "version": "0.1.8"}
    (repo / "README.md").write_text("Documentation only.\n")
    docs = _commit(repo)
    assert _plan(repo, merge, docs) == {"mode": "none", "version": "0.1.8"}


def test_ci_fix_recovers_package_changes_left_by_failed_old_release(repo: Path) -> None:
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    before = _commit(repo)
    (repo / "workflow.yml").write_text("# CI-only fix\n")
    after = _commit(repo)
    assert _plan(repo, before, after) == {"mode": "prepare", "version": "0.1.8"}


def test_docs_only_changes_do_not_create_a_release(repo: Path) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("Documentation only.\n")
    after = _commit(repo)
    assert _plan(repo, before, after) == {"mode": "none", "version": "0.1.7"}


@pytest.mark.parametrize("version", ["0.1.8", "0.2.0", "1.0.0"])
def test_patch_minor_and_major_bumps_publish_exact_declared_version(
    repo: Path, version: str
) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    _versions(repo, version)
    after = _commit(repo)
    assert _plan(repo, before, after) == {"mode": "publish", "version": version}


def test_alpha_prepare_keeps_alpha_release_sequence(repo: Path) -> None:
    _versions(repo, "0.2.0a3")
    before = _commit(repo)
    _git(repo, "tag", "v0.2.0a3")
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    after = _commit(repo)
    assert _plan(repo, before, after) == {"mode": "prepare", "version": "0.2.0a4"}
    result = _run(repo, "prepare", "--version", "0.2.0a4")
    assert result.returncode == 0, result.stderr
    assert 'version = "0.2.0a4"' in (repo / "pyproject.toml").read_text()


@pytest.mark.parametrize("version", ["0.1.6", "0.1.7a1"])
def test_version_downgrades_are_rejected(repo: Path, version: str) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    _versions(repo, version)
    after = _commit(repo)
    result = _run(repo, "plan", "--before", before, "--after", after)
    assert result.returncode != 0
    assert "must increase" in result.stderr


def test_existing_tag_on_another_commit_blocks_publication(repo: Path) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v0.1.8", before)
    _versions(repo, "0.1.8")
    after = _commit(repo)
    result = _run(repo, "plan", "--before", before, "--after", after)
    assert result.returncode != 0
    assert "another commit" in result.stderr


def test_pending_publish_does_not_produce_another_version_bump(repo: Path) -> None:
    _versions(repo, "0.1.8")
    before = _commit(repo)
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    after = _commit(repo)
    assert _plan(repo, before, after) == {"mode": "none", "version": "0.1.8"}


@pytest.mark.parametrize("bad_before", ["0" * 40, "main", "--help", "a" * 40])
def test_unknown_or_invalid_event_sha_cannot_publish(repo: Path, bad_before: str) -> None:
    after = _git(repo, "rev-parse", "HEAD")
    result = _run(repo, "plan", "--before", bad_before, "--after", after)
    assert result.returncode != 0
    assert "mode=publish" not in result.stdout


@pytest.mark.parametrize("bad_file", VERSION_FILES[1:])
def test_version_drift_is_rejected_before_writing(repo: Path, bad_file: str) -> None:
    path = repo / bad_file
    path.write_text(path.read_text().replace("0.1.7", "0.1.6"))
    original = {name: (repo / name).read_bytes() for name in VERSION_FILES}
    result = _run(repo, "prepare", "--version", "0.1.8")
    assert result.returncode != 0
    assert "versions must agree" in result.stderr
    assert original == {name: (repo / name).read_bytes() for name in VERSION_FILES}


@pytest.mark.parametrize(
    "args", [("--version", "0.1.9"), ("--version", "0.1.8", "--date", "2026-02-30")]
)
def test_invalid_prepare_request_leaves_files_untouched(repo: Path, args: tuple[str, ...]) -> None:
    original = {name: (repo / name).read_bytes() for name in VERSION_FILES}
    result = _run(repo, "prepare", *args)
    assert result.returncode != 0
    assert original == {name: (repo / name).read_bytes() for name in VERSION_FILES}


def test_workflow_keeps_publication_out_of_pr_events_and_never_pushes_main() -> None:
    yaml = pytest.importorskip("yaml", reason="workflow checks require the docs extra")
    text = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in workflow["on"]
    assert "ready_for_review" in workflow["on"]["pull_request"]["types"]
    release = workflow["jobs"]["release"]
    assert release["needs"] == ["test", "citation"]
    assert "github.event_name == 'push'" in release["if"]
    assert "github.ref == 'refs/heads/main'" in release["if"]
    assert release["concurrency"]["cancel-in-progress"] == "false"
    steps = {step.get("name", ""): step for step in release["steps"]}
    pr = steps["Open or update draft release PR"]
    assert re.fullmatch(r"peter-evans/create-pull-request@[0-9a-f]{40}", pr["uses"])
    assert pr["with"]["draft"] == "always-true"
    assert pr["with"]["branch"].startswith("release/v")
    assert set(pr["with"]["add-paths"].splitlines()) == set(VERSION_FILES)
    assert "steps.tip.outputs.current == 'true'" in pr["if"]
    for name in ("Build", "GitHub Release", "Publish to PyPI"):
        assert steps[name]["if"] == "steps.plan.outputs.mode == 'publish'"
    assert steps["GitHub Release"]["with"]["target_commitish"] == "${{ github.sha }}"
    assert "git push" not in text
    assert "git commit" not in text
