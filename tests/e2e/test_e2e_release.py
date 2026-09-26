"""Exercise release planning in disposable Git repos, with no remote or uploads."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "release.py"
PR_SCRIPT = REPO_ROOT / "tools" / "prepare_pr_release.py"
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


def _prepare_pr(root: Path, base: str, head: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(PR_SCRIPT),
            "--root",
            str(root),
            "--base",
            base,
            "--head",
            head,
            *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_code_and_version_merge_in_one_pr_then_publish(repo: Path) -> None:
    released = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "--no-track", "-c", "fix/feature")
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    code_commit = _commit(repo)
    check = _prepare_pr(repo, released, code_commit, "--check")
    assert check.returncode != 0
    assert "Include version 0.1.8 in this PR" in check.stderr
    assert _git(repo, "status", "--porcelain") == ""

    result = _prepare_pr(repo, released, code_commit, "--date", "2026-09-26")
    assert result.returncode == 0, result.stderr
    assert "updated=true" in result.stdout
    assert set(_git(repo, "diff", "--name-only").splitlines()) == set(VERSION_FILES)
    assert 'date-released: "2026-09-26"' in (repo / "CITATION.cff").read_text()
    prepared = _commit(repo)
    checked = _prepare_pr(repo, released, prepared, "--check")
    assert checked.returncode == 0, checked.stderr
    assert "updated=false" in checked.stdout
    _git(repo, "switch", "main")
    _git(repo, "merge", "--no-ff", "fix/feature", "-m", "Merge working PR")
    merge = _git(repo, "rev-parse", "HEAD")
    assert _plan(repo, released, merge) == {"mode": "publish", "version": "0.1.8"}

    # A retry after GitHub Release succeeds but PyPI fails must not bump again.
    _git(repo, "tag", "-a", "v0.1.8", "-m", "release", merge)
    assert _plan(repo, released, merge) == {"mode": "publish", "version": "0.1.8"}
    (repo / "README.md").write_text("Documentation only.\n")
    docs = _commit(repo)
    assert _plan(repo, merge, docs) == {"mode": "none", "version": "0.1.8"}


def test_ci_fix_recovers_package_changes_left_by_failed_old_release(repo: Path) -> None:
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    before = _commit(repo)
    (repo / "workflow.yml").write_text("# CI-only fix\n")
    after = _commit(repo)
    # No separate release PR is opened after a merge. The working PR itself
    # must include the bump, even when its only change fixes the old CI.
    assert _plan(repo, before, after) == {"mode": "none", "version": "0.1.7"}
    result = _prepare_pr(repo, before, after)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "updated=true\nversion=0.1.8\n"


def test_docs_only_changes_do_not_create_a_release(repo: Path) -> None:
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "README.md").write_text("Documentation only.\n")
    after = _commit(repo)
    assert _plan(repo, before, after) == {"mode": "none", "version": "0.1.7"}
    result = _prepare_pr(repo, before, after)
    assert result.returncode == 0, result.stderr
    assert "updated=false" in result.stdout
    assert _git(repo, "status", "--porcelain") == ""


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
    assert _plan(repo, before, after) == {"mode": "none", "version": "0.2.0a3"}
    result = _prepare_pr(repo, before, after)
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


def test_ci_fix_recovers_untagged_version_and_retry_keeps_same_commit(repo: Path) -> None:
    released = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "--no-track", "-c", "fix/feature")
    _versions(repo, "0.1.8")
    _commit(repo)
    _git(repo, "switch", "main")
    _git(repo, "merge", "--no-ff", "fix/feature", "-m", "Prepare 0.1.8")
    prepared = _git(repo, "rev-parse", "HEAD")
    assert _plan(repo, released, prepared) == {"mode": "publish", "version": "0.1.8"}
    # The original workflow skipped release, so there is no v0.1.8 tag.
    _git(repo, "switch", "--no-track", "-c", "fix/ci")
    (repo / "workflow.yml").write_text("# Correct the release job condition\n")
    head = _commit(repo)
    original = {name: (repo / name).read_bytes() for name in VERSION_FILES}
    check = _prepare_pr(repo, prepared, head, "--check")
    assert check.returncode == 0, check.stderr
    assert check.stdout == "updated=false\nversion=0.1.8\n"
    _git(repo, "switch", "main")
    _git(repo, "merge", "--no-ff", "fix/ci", "-m", "Recover 0.1.8")
    recovered = _git(repo, "rev-parse", "HEAD")
    assert _plan(repo, prepared, recovered) == {"mode": "publish", "version": "0.1.8"}
    assert original == {name: (repo / name).read_bytes() for name in VERSION_FILES}
    assert _git(repo, "status", "--porcelain") == ""

    _git(repo, "tag", "-a", "v0.1.8", "-m", "release", recovered)
    assert _plan(repo, prepared, recovered) == {"mode": "publish", "version": "0.1.8"}
    # The old merge cannot take over a tag created by the recovery merge.
    stale = _run(repo, "plan", "--before", released, "--after", prepared)
    assert stale.returncode != 0
    assert "another commit" in stale.stderr
    (repo / "README.md").write_text("Documentation after recovery.\n")
    later = _commit(repo)
    assert _plan(repo, recovered, later) == {"mode": "none", "version": "0.1.8"}


@pytest.mark.parametrize(
    "changed_file", ["src/purgedcv/feature.py", "pyproject.toml", "CITATION.cff"]
)
def test_recovery_rejects_package_changes_since_version_was_prepared(
    repo: Path, changed_file: str
) -> None:
    _versions(repo, "0.1.8")
    _commit(repo)
    with (repo / changed_file).open("a") as stream:
        stream.write("# A later change\n")
    before = _commit(repo)
    # Checking just before..after would miss this earlier package change.
    (repo / "workflow.yml").write_text("# CI-only fix\n")
    after = _commit(repo)
    result = _run(repo, "plan", "--before", before, "--after", after)
    assert result.returncode != 0
    assert "Package files changed after the version was prepared" in result.stderr
    assert "mode=publish" not in result.stdout


def test_recovery_requires_a_version_increase_in_history(repo: Path) -> None:
    _git(repo, "tag", "-d", "v0.1.7")
    before = _git(repo, "rev-parse", "HEAD")
    (repo / "workflow.yml").write_text("# CI-only change\n")
    after = _commit(repo)
    result = _run(repo, "plan", "--before", before, "--after", after)
    assert result.returncode != 0
    assert "previous version increase" in result.stderr


def test_recovery_rejects_a_previously_downgraded_version(repo: Path) -> None:
    _versions(repo, "0.1.6")
    before = _commit(repo)
    (repo / "workflow.yml").write_text("# CI-only change\n")
    after = _commit(repo)
    result = _run(repo, "plan", "--before", before, "--after", after)
    assert result.returncode != 0
    assert "previously increased version" in result.stderr


def test_ci_fix_does_not_move_tag_after_partial_publication(repo: Path) -> None:
    _versions(repo, "0.1.8")
    prepared = _commit(repo)
    _git(repo, "tag", "v0.1.8", prepared)
    (repo / "workflow.yml").write_text("# CI-only fix\n")
    after = _commit(repo)
    assert _plan(repo, prepared, after) == {"mode": "none", "version": "0.1.8"}
    assert _git(repo, "rev-parse", "v0.1.8") == prepared


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


def test_bot_update_is_idempotent_and_preserves_manual_major_bump(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _versions(repo, "1.0.0")
    head = _commit(repo)
    for _ in range(2):
        result = _prepare_pr(repo, base, head)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "updated=false\nversion=1.0.0\n"
        assert _git(repo, "status", "--porcelain") == ""


def test_old_branch_bumps_from_latest_main_version(repo: Path) -> None:
    _git(repo, "switch", "--no-track", "-c", "fix/older")
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    head = _commit(repo)
    _git(repo, "switch", "main")
    _versions(repo, "0.1.8")
    base = _commit(repo)
    _git(repo, "tag", "v0.1.8")
    _git(repo, "switch", "fix/older")
    result = _prepare_pr(repo, base, head)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "updated=true\nversion=0.1.9\n"


def test_followup_code_changes_do_not_increment_twice(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    head = _commit(repo)
    assert _prepare_pr(repo, base, head).returncode == 0
    _commit(repo)
    (repo / "src/purgedcv/feature.py").write_text("value = 2\n")
    head = _commit(repo)
    result = _prepare_pr(repo, base, head)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "updated=false\nversion=0.1.8\n"
    assert _git(repo, "status", "--porcelain") == ""


def test_pending_main_publication_stops_automatic_pr_bump(repo: Path) -> None:
    _versions(repo, "0.1.8")
    base = _commit(repo)
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    head = _commit(repo)
    result = _prepare_pr(repo, base, head)
    assert result.returncode != 0
    assert "finish its publication first" in result.stderr
    assert _git(repo, "status", "--porcelain") == ""


def test_pr_bump_rejects_existing_target_tag(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "tag", "v0.1.8")
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    head = _commit(repo)
    result = _prepare_pr(repo, base, head)
    assert result.returncode != 0
    assert "already has a release tag" in result.stderr
    assert _git(repo, "status", "--porcelain") == ""


def test_pr_checkout_must_match_event_head(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "src/purgedcv/feature.py").write_text("value = 1\n")
    _commit(repo)
    result = _prepare_pr(repo, base, base)
    assert result.returncode != 0
    assert "expected PR head" in result.stderr


@pytest.mark.parametrize("symlink_parent", [False, True])
def test_pr_version_update_rejects_symlinks(repo: Path, symlink_parent: bool) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    outside = repo.parent / f"outside-version-{repo.name}"
    if symlink_parent:
        package = repo / "src/purgedcv"
        package.rename(outside)
        package.symlink_to(outside, target_is_directory=True)
        target = outside / "__init__.py"
    else:
        target = outside
        source = repo / "src/purgedcv/__init__.py"
        source.rename(target)
        source.symlink_to(target)
    original = target.read_bytes()
    head = _commit(repo)
    result = _prepare_pr(repo, base, head)
    assert result.returncode != 0
    assert "regular file inside the checkout" in result.stderr
    assert target.read_bytes() == original


def test_pr_tooling_and_package_are_never_imported(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD")
    code = 'raise RuntimeError("PR CODE MUST NOT RUN")\n'
    (repo / "src/purgedcv/__init__.py").write_text('__version__ = "0.1.7"\n' + code)
    (repo / "tools").mkdir()
    for name in ("release.py", "sync_citation.py", "prepare_pr_release.py"):
        (repo / "tools" / name).write_text(code)
    head = _commit(repo)
    result = _prepare_pr(repo, base, head)
    assert result.returncode == 0, result.stderr
    assert code in (repo / "src/purgedcv/__init__.py").read_text()


def test_workflow_updates_only_same_repo_prs_with_trusted_tooling() -> None:
    yaml = pytest.importorskip("yaml", reason="workflow checks require the docs extra")
    text = (REPO_ROOT / ".github/workflows/ci.yml").read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in workflow["on"]
    assert "ready_for_review" in workflow["on"]["pull_request"]["types"]
    updater = workflow["jobs"]["pr-version"]
    assert updater["permissions"] == {"contents": "write"}
    assert "github.event_name == 'pull_request'" in updater["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in updater["if"]
    update_steps = {step["name"]: step for step in updater["steps"]}
    trusted = update_steps["Check out trusted release tooling"]["with"]
    assert trusted["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert trusted["persist-credentials"] == "false"
    script = update_steps["Update version in this PR"]["run"]
    assert "python ../release-tools/tools/prepare_pr_release.py" in script
    assert 'git push origin "HEAD:refs/heads/$PR_BRANCH"' in script
    assert '[ "$PR_BRANCH" = main ]' in script
    assert '[ "$PR_BRANCH" = master ]' in script
    assert "--force" not in script
    for name in ("test", "citation"):
        job = workflow["jobs"][name]
        assert job["needs"] == ["pr-version"]
        assert "needs.pr-version.result == 'skipped'" in job["if"]
        assert "needs.pr-version.outputs.updated != 'true'" in job["if"]
    release = workflow["jobs"]["release"]
    assert release["permissions"] == {"contents": "write"}
    assert release["needs"] == ["test", "citation"]
    assert "github.event_name == 'push'" in release["if"]
    assert "github.ref == 'refs/heads/main'" in release["if"]
    assert release["concurrency"]["cancel-in-progress"] == "false"
    steps = {step.get("name", ""): step for step in release["steps"]}
    assert "create-pull-request" not in text
    assert "pull-requests: write" not in text
    assert "release/v" not in text
    for name in ("Build", "GitHub Release", "Publish to PyPI"):
        assert steps[name]["if"] == "steps.plan.outputs.mode == 'publish'"
    assert steps["GitHub Release"]["with"]["target_commitish"] == "${{ github.sha }}"
    assert not any("git push" in step.get("run", "") for step in release["steps"])


@pytest.mark.parametrize("test_result", ["success", "failure", "skipped", "cancelled"])
@pytest.mark.parametrize("citation_result", ["success", "failure", "skipped", "cancelled"])
def test_release_condition_handles_skipped_ancestor_without_bypassing_checks(
    test_result: str, citation_result: str
) -> None:
    """Check the actual YAML condition against the observed main-run scenario.

    This is a condition regression test, not a GitHub Actions runner. Without
    a status function GitHub adds success(), which skips release when the
    ancestor pr-version is skipped, even if its direct needs succeeded.
    """
    yaml = pytest.importorskip("yaml", reason="workflow checks require the docs extra")
    workflow = yaml.load(
        (REPO_ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader
    )
    condition = workflow["jobs"]["release"]["if"]
    clauses = [clause.strip() for clause in condition.split("&&")]
    assert "always()" in clauses, "Explicitly override skipped-ancestor propagation"
    for event, ref, cancelled in [
        ("push", "refs/heads/main", False),
        ("push", "refs/heads/main", True),
        ("pull_request", "refs/heads/main", False),
        ("push", "refs/heads/feature", False),
    ]:
        values = {
            "always()": True,
            "!cancelled()": not cancelled,
            "needs.test.result == 'success'": test_result == "success",
            "needs.citation.result == 'success'": citation_result == "success",
            "github.event_name == 'push'": event == "push",
            "github.ref == 'refs/heads/main'": ref == "refs/heads/main",
        }
        allowed = all(values[clause] for clause in clauses)
        assert allowed == (
            event == "push"
            and ref == "refs/heads/main"
            and not cancelled
            and test_result == "success"
            and citation_result == "success"
        )
