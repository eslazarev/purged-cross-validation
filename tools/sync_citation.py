"""Keep ``CITATION.cff`` in step with the package version.

The single source of truth for the version is ``pyproject.toml``. The release
workflow bumps it (and ``src/purgedcv/__init__.py``) on every package-changing
merge; this script propagates that version into ``CITATION.cff`` so the
citation metadata never drifts behind PyPI.

It edits only the ``version:`` and (optionally) ``date-released:`` lines, by
regex, so comments, field order, and the rest of the YAML are preserved. No
YAML parser and no third-party dependency is needed.

Usage::

    python tools/sync_citation.py --check                  # CI drift guard
    python tools/sync_citation.py --write                  # sync version only
    python tools/sync_citation.py --write --date 2026-06-14 # version + date

``--check`` exits non-zero if ``CITATION.cff`` disagrees with
``pyproject.toml`` (used in CI). ``--write`` rewrites the file in place and is
idempotent.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_PYPROJECT_VERSION = re.compile(r'(?m)^version = "([^"]+)"$')
_CFF_VERSION_LINE = re.compile(r"(?m)^version:[^\n]*$")
_CFF_DATE_LINE = re.compile(r"(?m)^date-released:[^\n]*$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def pyproject_version(text: str) -> str:
    """The canonical version string from ``pyproject.toml``."""
    match = _PYPROJECT_VERSION.search(text)
    if match is None:
        raise SystemExit('pyproject.toml: no `version = "X.Y.Z"` line found')
    return match.group(1)


def cff_version(text: str) -> str | None:
    """The version currently declared in ``CITATION.cff`` (or ``None``)."""
    match = _CFF_VERSION_LINE.search(text)
    if match is None:
        return None
    return match.group(0).split(":", 1)[1].strip().strip("'\"")


def sync_text(cff_text: str, version: str, date: str | None) -> str:
    """Return ``cff_text`` with the version (and optional date) set."""
    if _CFF_VERSION_LINE.search(cff_text) is None:
        raise SystemExit("CITATION.cff: no `version:` line to update")
    out = _CFF_VERSION_LINE.sub(f"version: {version}", cff_text, count=1)
    if date is not None:
        if _CFF_DATE_LINE.search(out) is None:
            raise SystemExit("CITATION.cff: no `date-released:` line to update")
        out = _CFF_DATE_LINE.sub(f'date-released: "{date}"', out, count=1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if CITATION.cff version disagrees with pyproject.toml",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="rewrite CITATION.cff to match pyproject.toml",
    )
    parser.add_argument(
        "--date",
        help="ISO date (YYYY-MM-DD) to write into date-released (only with --write)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository root holding pyproject.toml and CITATION.cff",
    )
    args = parser.parse_args(argv)

    pyproject = args.root / "pyproject.toml"
    citation = args.root / "CITATION.cff"
    version = pyproject_version(pyproject.read_text(encoding="utf-8"))
    cff_text = citation.read_text(encoding="utf-8")

    if args.check:
        if args.date is not None:
            parser.error("--date is only valid with --write")
        current = cff_version(cff_text)
        if current == version:
            print(f"CITATION.cff is in sync (version {version}).")
            return 0
        print(
            f"CITATION.cff version {current!r} disagrees with pyproject.toml "
            f"{version!r}. Run: python tools/sync_citation.py --write",
            file=sys.stderr,
        )
        return 1

    if args.date is not None and _ISO_DATE.match(args.date) is None:
        parser.error(f"--date must be YYYY-MM-DD, got {args.date!r}")
    new_text = sync_text(cff_text, version, args.date)
    if new_text == cff_text:
        print(f"CITATION.cff already at version {version}; nothing to write.")
        return 0
    citation.write_text(new_text, encoding="utf-8")
    suffix = f" (date-released {args.date})" if args.date else ""
    print(f"CITATION.cff updated to version {version}{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
