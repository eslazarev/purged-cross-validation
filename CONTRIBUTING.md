# Contributing to purgedcv

Thank you for considering a contribution. This project keeps a small,
opinionated surface so that the cross-validation primitives it ships stay
correct. The conventions below are what reviewers will check for.

## Quick start

```bash
git clone https://github.com/eslazarev/purged-cross-validation.git
cd purged-cross-validation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

Python 3.10 or newer is required.

## Running the gates locally

The code CI runs four checks. Run them locally before opening a pull request:

```bash
ruff check .
black --check src tests tools examples/_lcl_harness.py
mypy src tests
pytest -q
```

All four must pass. The test suite includes property-based tests
(`hypothesis`), doctest collection, and end-to-end tests that subprocess
the installed package; the `test_quality_gate.py` e2e re-runs black, ruff,
and mypy through pytest so a regression cannot slip past `pytest` alone.

A documentation build check is also part of CI:

```bash
mkdocs build --strict
```

`--strict` treats warnings as errors, so broken links or missing autodoc
references will fail the build.

## What needs an end-to-end test

Any user-visible behaviour (a new splitter, a metric, a diagnostic, or a
new CLI tool under `tools/`) needs an entry in `tests/e2e/`. Unit and
property tests stay in `tests/` (flat). When a feature spans multiple
modules, prefer a subprocess-style e2e test that exercises the public API
the way a user would.

## Prose-quality gate for documentation

User-facing documentation listed in `tools/prose_gate.py` `TARGETS` (the
README, examples README, notebooks with substantial markdown, papers in
`docs/`, and this file) is checked by a small local heuristic gate that
flags AI-tell phrasing and uniform sentence rhythm. Run it before opening
a documentation PR:

```bash
python tools/prose_gate.py
```

Any **FAIL** result blocks a PR; **WARN** is advisory. The gate is a
heuristic and not a detector, but it catches regressions reliably. If you
disagree with a flag in your prose, raise it in the PR.

## Commit messages and pull requests

- Conventional commits style is appreciated (`feat:`, `fix:`, `chore:`,
  `docs:`, `test:`). Not enforced.
- Keep PRs focused. One feature, one fix, one refactor, not a mix.
- Update `CHANGELOG.md` under `Unreleased` if your change is user-visible.
- Code and release metadata belong in the same PR. See the maintainer steps
  below; no separate release PR is needed.

## Releasing with protected main

Open one PR with your code changes. For branches in this repository, CI adds
a commit to that same branch when a release is needed. It updates only
`pyproject.toml`, `src/purgedcv/__init__.py`, and `CITATION.cff`. The bump advances
the patch, or the alpha number for alpha versions. Later pushes to the PR
keep the prepared version rather than bumping it again. A manual minor or
major bump is preserved.

Wait for checks on the **new head commit**, not the commit before the bot's
update. GitHub may show **Approve workflows to run** after a bot push. Approve
those runs if asked. If no new run appears, converting the PR to a draft and
then marking it Ready for review triggers CI. Review the version metadata
with the code, then merge this one PR.

The write-enabled job uses tooling from a separate checkout of trusted
`main`; it never installs or executes code from the PR. Fork PRs do not get
write access. Their authors must include the version bump themselves:

```bash
python tools/release.py prepare --version <next-version>
```

That command edits the three files locally. Commit them to the same working
branch. For a minor or major release, edit the two package version strings
and run `python tools/sync_citation.py --write --date YYYY-MM-DD` instead.
CI checks the three versions agree and reports a missing bump. If another PR
releases first, update your branch from `main` and run the checks again.

Docs-only PRs leave the version unchanged unless older package changes are
still waiting for release. The bot compares against the current version's
tag so changes left by the old failed release are not forgotten. No new PR
is created after a merge. This workflow does not need permission to create
or approve PRs, a personal access token, or a branch-protection bypass.

After merge, `main` runs tests and citation checks again. A version increase
then builds, tags that exact tested commit, creates the GitHub release, and
uploads to PyPI. An unchanged version does not publish. During publication,
the version on `main` is pending, not yet released.

If publication fails, rerun the failed jobs on the **original version-changing
merge workflow**. Do not bump again. An existing tag must point to that same
commit; a conflicting tag stops publication. Already-uploaded files are
skipped on retry. Finish a pending release before preparing the next one.

The citation date is set when the version is prepared. Adjust it before
merging if the PR has been waiting for another day. When introducing this
workflow for the first time, include the version bump manually in that PR:
the bot cannot use the new trusted helper until it is on `main`.

## Reporting bugs and requesting features

Open an issue from one of the templates in `.github/ISSUE_TEMPLATE/`. For
bug reports, the most helpful thing you can include is a minimal
reproducer: a few lines of code, the actual output, and the expected
output. For feature requests, describing the cross-validation use case
matters more than describing the API you would like.

## Code of conduct

This project follows the [Contributor Covenant 2.1](CODE_OF_CONDUCT.md).
By participating you agree to abide by it. Report unacceptable behaviour
to elazarev@gmail.com.

## Citing the package

If `purgedcv` contributes to academic work, please cite it via
`CITATION.cff` (machine-readable) or the JOSS paper at `paper/paper.md`.
