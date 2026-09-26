"""Plan releases and prepare version-only PRs without pushing or publishing.

The workflow calls ``plan`` on a tested main commit. A version change means
publish that commit; otherwise compare with the current version's tag to
find package changes waiting for a release PR. ``prepare`` updates only the
three version files in the checkout. Both commands emit GitHub step outputs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from sync_citation import cff_version, pyproject_version, sync_text

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)(?:a(\d+))?\Z")
INIT_VERSION = re.compile(r'(?m)^__version__ = "([^"]+)"$')
VERSION_FILES = ("pyproject.toml", "src/purgedcv/__init__.py", "CITATION.cff")


def version_key(value: str) -> tuple[int, int, int, bool, int]:
    """Order supported stable and alpha versions without a runtime dependency."""
    match = VERSION.fullmatch(value)
    if match is None:
        raise ValueError(f"Unsupported version: {value!r}; expected X.Y.Z or X.Y.ZaN")
    major, minor, patch, alpha = match.groups()
    return int(major), int(minor), int(patch), alpha is None, int(alpha or 0)


def next_version(value: str) -> str:
    major, minor, patch, stable, alpha = version_key(value)
    return f"{major}.{minor}.{patch + 1}" if stable else f"{major}.{minor}.{patch}a{alpha + 1}"


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def checked_version(pyproject: str, init: str, citation: str) -> str:
    version = pyproject_version(pyproject)
    version_key(version)
    match = INIT_VERSION.search(init)
    if match is None or match.group(1) != version or cff_version(citation) != version:
        raise ValueError("pyproject.toml, __init__.py, and CITATION.cff versions must agree")
    return version


def commit_version(root: Path, commit: str) -> str:
    texts = [git(root, "show", f"{commit}:{name}") for name in VERSION_FILES]
    return checked_version(*texts)


def resolve_commit(root: Path, revision: str) -> str:
    # Event SHAs only, never user-controlled ref expressions or Git options.
    if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision) is None or not revision.strip("0"):
        raise ValueError("Release planning requires a non-zero full commit SHA")
    return git(root, "rev-parse", "--verify", f"{revision}^{{commit}}")


def tagged_commit(root: Path, version: str) -> str | None:
    tag = f"refs/tags/v{version}"
    # Enumerating tags distinguishes a missing tag from a malformed existing tag.
    if tag not in git(root, "for-each-ref", "--format=%(refname)", "refs/tags").splitlines():
        return None
    return git(root, "rev-parse", "--verify", f"{tag}^{{commit}}")


def plan(root: Path, before: str, after: str) -> tuple[str, str]:
    before = resolve_commit(root, before)
    after = resolve_commit(root, after)
    git(root, "merge-base", "--is-ancestor", before, after)
    previous = commit_version(root, before)
    current = commit_version(root, after)
    tagged = tagged_commit(root, current)
    if current != previous:
        if version_key(current) <= version_key(previous):
            raise ValueError("A release version must increase")
        if tagged is not None and tagged != after:
            raise ValueError(
                f"Tag v{current} already points to another commit; refusing to publish"
            )
        # Rerunning the original merge event after a partial publish uses the
        # same before/after SHAs and takes this path again, without another bump.
        return "publish", current
    if tagged is None:
        print(
            f"No v{current} tag yet. Finish or rerun the version-bump merge workflow first.",
            file=sys.stderr,
        )
        return "none", current
    git(root, "merge-base", "--is-ancestor", tagged, after)
    changed = git(root, "diff", "--name-only", tagged, after, "--", "src", "pyproject.toml")
    # Use the release tag, not just the previous push: this also catches code
    # left unreleased by the old direct-push workflow after a CI-only fix.
    return ("prepare", next_version(current)) if changed else ("none", current)


def prepare(root: Path, expected: str, release_date: str) -> str:
    date.fromisoformat(release_date)
    paths = [root / name for name in VERSION_FILES]
    texts = [path.read_text(encoding="utf-8") for path in paths]
    current = checked_version(*texts)
    version = next_version(current)
    if version != expected:
        raise ValueError(f"Expected next version {expected}, but checkout requires {version}")
    # Validate and construct every replacement before writing any file.
    updated = [
        re.sub(r'(?m)^version = "[^"]+"$', f'version = "{version}"', texts[0], count=1),
        INIT_VERSION.sub(f'__version__ = "{version}"', texts[1], count=1),
        sync_text(texts[2], version, release_date),
    ]
    for path, text in zip(paths, updated, strict=True):
        path.write_text(text, encoding="utf-8")
    return version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    planning = commands.add_parser("plan")
    planning.add_argument("--before", required=True)
    planning.add_argument("--after", required=True)
    preparing = commands.add_parser("prepare")
    preparing.add_argument("--version", required=True)
    preparing.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = parser.parse_args()
    try:
        if args.command == "plan":
            mode, version = plan(args.root, args.before, args.after)
            print(f"mode={mode}\nversion={version}")
        else:
            print(f"version={prepare(args.root, args.version, args.date)}")
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"Release stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
