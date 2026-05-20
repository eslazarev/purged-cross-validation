## What changes

A short description of the change and the reason for it. Link any issues
it fixes (`Fixes #123`).

## Checklist

- [ ] `ruff check . && black --check src tests tools examples/_lcl_harness.py` is clean.
- [ ] `mypy src tests` is clean.
- [ ] `pytest -q` is green locally.
- [ ] If this changes user-visible behaviour, there is a test under
      `tests/e2e/` that exercises it the way a user would.
- [ ] If this touches user-facing documentation listed in
      `tools/prose_gate.py TARGETS`, the prose gate is FAIL-free.
- [ ] If the docs site is affected, `mkdocs build --strict` is clean.
- [ ] `CHANGELOG.md` has an entry under `Unreleased`, if the change is
      user-visible.

## Notes for the reviewer

Anything the diff alone does not make obvious: a behaviour change to
flag, a textbook reference for the algorithm, a known limitation, or a
follow-up issue.
