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
- Releases use a separate version-bump PR. See the maintainer steps below.
  Do not bump the version in an ordinary feature or bug-fix PR.

## Releasing with protected main

No direct push to `main` is needed. Keep branch protection enabled, including
for administrators. In **Settings > Actions > General > Workflow permissions**,
enable **Allow GitHub Actions to create and approve pull requests**. GitHub
uses that setting for PR creation too; this workflow never approves or merges
its own PR. Default token permissions can stay read-only.

After CI passes on `main`, the release job compares the package files
(`src/**` and `pyproject.toml`) with the current version's tag. Pending changes
produce a draft `release/vX.Y.Z` PR updating `pyproject.toml`,
`src/purgedcv/__init__.py`, and `CITATION.cff`. Further merges can update that
draft. The bump advances the patch, or the alpha number for alpha versions.
Docs-only changes do not cause a release when no package changes are pending.
An earlier failed release may leave pending changes, so even a CI-only merge
can open the needed release PR.

Review the draft. Mark it **Ready for review** to trigger CI, and approve any
pending workflow run if GitHub asks. PRs created with `GITHUB_TOKEN` may need
that human action before checks run. Wait for green checks, then merge.
If the bot updates the PR, it becomes a draft again so the new head can be
checked. No personal access token or branch-protection bypass is required.

The version-changing merge to `main` runs tests and citation checks again.
Only after they pass does the release job build, tag that exact merge commit,
create the GitHub release, and upload to PyPI. During that interval, the
version on `main` is pending publication, not yet a published release.
For a minor or major release, prepare the three version files together in a
reviewed PR instead of using the automatic patch bump.

If publication fails, rerun the failed jobs on the **original version-bump
merge workflow**. Do not bump again. An existing tag must point to that same
commit; a conflicting tag stops publication. Already-uploaded files are
skipped on retry. A missing tag for an unchanged version pauses further
automatic bumps until the pending release is finished. Once it is finished,
rerun CI on a newer `main` push if package changes are still waiting.

The `CITATION.cff` date is the date the release PR was prepared. Adjust it
before merging if the release has been waiting for another day.

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
