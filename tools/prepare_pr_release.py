"""Prepare version metadata in the working PR, using tooling from trusted main.

The write-enabled workflow runs this script from a separate base checkout.
Candidate files are read as data, never imported or executed. This script
does not commit, push, create PRs, or contact a package registry.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from release import (
    commit_version,
    git,
    next_version,
    resolve_commit,
    tagged_commit,
    version_files,
    version_key,
    write_version,
)


def plan_pr(root: Path, base: str, head: str) -> tuple[bool, str]:
    base = resolve_commit(root, base)
    head = resolve_commit(root, head)
    if git(root, "rev-parse", "HEAD") != head:
        raise ValueError("Candidate checkout does not match the expected PR head")
    if git(root, "status", "--porcelain"):
        raise ValueError("Candidate checkout must be clean before preparing a version")
    version_files(root)  # Reject symlinks even when no update is necessary.
    base_version = commit_version(root, base)
    head_version = commit_version(root, head)
    if version_key(head_version) > version_key(base_version):
        if tagged_commit(root, head_version) is not None:
            raise ValueError(f"Version {head_version} already has a release tag")
        # Preserve a reviewed manual minor/major bump, and never bump twice
        # when the bot's own commit triggers another synchronize event.
        return False, head_version

    ancestor = git(root, "merge-base", base, head)
    changed = git(root, "diff", "--name-only", ancestor, head, "--", "src", "pyproject.toml")
    tag = tagged_commit(root, base_version)
    pending = ""
    if tag is not None:
        git(root, "merge-base", "--is-ancestor", tag, base)
        pending = git(root, "diff", "--name-only", tag, base, "--", "src", "pyproject.toml")
    if not changed and not pending:
        return False, head_version
    if tag is None:
        raise ValueError(
            f"Version {base_version} on main has no release tag yet; finish its publication first"
        )
    version = next_version(base_version)
    if tagged_commit(root, version) is not None:
        raise ValueError(f"Version {version} already has a release tag")
    return True, version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--check", action="store_true", help="fail instead of changing metadata")
    args = parser.parse_args()
    try:
        updated, version = plan_pr(args.root, args.base, args.head)
        if updated and args.check:
            raise ValueError(f"Include version {version} in this PR before merging")
        if updated:
            write_version(args.root, version, args.date)
    except (ValueError, subprocess.CalledProcessError) as exc:
        print(f"PR version update stopped: {exc}", file=sys.stderr)
        return 1
    print(f"updated={str(updated).lower()}\nversion={version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
