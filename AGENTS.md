# Notes for coding agents working on this repository

This file is for agents that change purgedcv itself. If you are writing code
that uses purgedcv, read `skills/purgedcv/SKILL.md` instead.

## Setup and checks

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,docs]"
ruff check .
black --check src tests tools examples/_lcl_harness.py
mypy src tests
pytest -q
mkdocs build --strict
python tools/prose_gate.py
```

All of them must pass before a PR. Python 3.11 is the minimum.

## Layout

- `src/purgedcv/`: the package. Files starting with `_` are internal. The
  public API is whatever `src/purgedcv/__init__.py` exports, and
  `tests/test_public_api.py` pins it, so adding or renaming a symbol means
  updating that test, `docs/api*.md` and the README API table.
- `tests/`: unit and property tests. `tests/e2e/`: user-story tests that run
  the public API or the docs in a fresh process.
- `docs/`: MkDocs site. Python blocks in `docs/quickstart.md`,
  `docs/choosing-a-splitter.md`, `docs/common-mistakes.md` and
  `skills/purgedcv/SKILL.md` are executed by e2e tests, so keep them runnable.
- `examples/`: notebooks on cached data in `examples/data/`. Notebooks are
  documentation and are not run under pytest.
- `paper/`: research artefacts. Do not edit them as part of library changes.

## Conventions

- Every user-visible change needs an e2e test in `tests/e2e/` and a
  `CHANGELOG.md` entry under `Unreleased`.
- Splitters must follow the scikit-learn splitter protocol and must never
  return training rows whose label window overlaps the test window.
- Do not invent benchmark numbers in docs. Quote what the examples measure.
- Prose in files listed in `tools/prose_gate.py` must pass the gate. Avoid
  em dashes and filler words.
- Version metadata is bumped by CI or `tools/release.py`; do not edit version
  strings by hand. See `CONTRIBUTING.md`.
