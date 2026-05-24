# Code metadata

Standard SoftwareX / Elsevier "Code metadata" table for `purgedcv`,
ready to paste into a SoftwareX submission. The values are kept in sync
with `pyproject.toml`, `CITATION.cff`, and `.zenodo.json`.

| Nr. | Code metadata description | Value |
|---|---|---|
| C1 | Current code version | v0.0.5 |
| C2 | Permanent link to code/repository used for this code version | <https://github.com/eslazarev/purged-cross-validation/releases/tag/v0.0.5> |
| C3 | Permanent link to Reproducible Capsule | n/a (deterministic; reproduce with `pytest -q` and `python tools/lcl_full_benchmark.py --k 20 --n 60 --seed 0`) |
| C4 | Legal Code License | MIT |
| C5 | Code versioning system used | git |
| C6 | Software code languages, tools, and services used | Python 3.10+; runtime dependencies: numpy ≥ 1.24, pandas ≥ 2.0, scikit-learn ≥ 1.3, scipy ≥ 1.10; development: pytest, hypothesis, ruff, mypy (strict), pandas-stubs, pre-commit; build: hatchling; docs: MkDocs Material, mkdocstrings; CI: GitHub Actions |
| C7 | Compilation requirements, operating environments & dependencies | Pure Python; no compilation step. Tested on CPython 3.10, 3.11, 3.12, 3.13, 3.14 (ubuntu-latest). Cross-platform (Linux, macOS, Windows). |
| C8 | If available, Link to developer documentation/manual | <https://eslazarev.github.io/purged-cross-validation/> |
| C9 | Support email for questions | <elazarev@gmail.com> |

## Software metadata (where applicable to a pure-Python library)

| Nr. | Software metadata description | Value |
|---|---|---|
| S1 | Current software version | v0.0.5 |
| S2 | Permanent link to executables of this version | <https://pypi.org/project/purgedcv/0.0.5/> (wheel + sdist published by the release workflow) |
| S3 | Legal Software License | MIT |
| S4 | Computing platforms / Operating Systems | Linux, macOS, Windows (any CPython 3.10+) |
| S5 | Installation requirements & dependencies | `pip install purgedcv` (base); optional extras `[dev]`, `[docs]`, `[examples]` |
| S6 | If available, link to user manual | <https://eslazarev.github.io/purged-cross-validation/> |
| S7 | Support email for questions | <elazarev@gmail.com> |

## Reproducibility one-liners

The headline numbers cited in `paper/paper.md` are produced by the
commands below. Each is deterministic from a fixed seed.

```bash
# 1. Full test suite (285 tests, ~2-4 min depending on hardware)
pip install -e ".[dev]"
pytest -q

# 2. Static gates (the same ones CI runs)
ruff check . && ruff format --check . && mypy src tests

# 3. Controlled synthetic-leakage proof (no download; deterministic)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  examples/synthetic_leakage_proof.ipynb

# 4. Competitor benchmark — sklearn + purgedcv core run (no network)
pip install -e ".[dev]"
python tools/competitor_benchmark.py --core-only --out-dir /tmp/competitor

# 5. Full UK Low Carbon London benchmark (offline; ~55 min, ~8 GB raw data)
python tools/lcl_full_benchmark.py --k 20 --n 60 --seed 0
```
