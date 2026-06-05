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
- Releases are automated. The version in `pyproject.toml` is the **last
  published** version. Merging a package change (`src/**` or `pyproject.toml`)
  to `main` bumps the patch, tags it, publishes that exact version to PyPI,
  and leaves `main` holding what is on PyPI (the bump happens before the
  upload, so git and PyPI never disagree). Merges that touch only docs, tests,
  CI, tooling, or examples do not release. Do not bump the version yourself
  for a normal patch release; the workflow does it (alpha-aware). A minor or
  major release is a manual step, since the auto-bump only increments the
  patch.

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
