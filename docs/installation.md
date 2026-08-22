# Installation

`purgedcv` is on PyPI and conda-forge, and requires Python 3.10 or newer.

## From PyPI

The base install gets the splitters, metrics, and diagnostics:

```bash
pip install purgedcv
```

## From conda-forge

For conda or mamba users, the package is published on conda-forge:

```bash
conda install -c conda-forge purgedcv
# or
mamba install -c conda-forge purgedcv
```

## Optional extras

Each extra is independent; install only what you need.

=== "Examples"

    To run the fifteen worked notebooks in `examples/` (Jupyter,
    Matplotlib, the `pricehub` OHLC fetcher for the crypto examples):

    ```bash
    pip install "purgedcv[examples]"
    ```

=== "Development"

    To run the test suite, the linter, and the type checker the same way
    CI does:

    ```bash
    pip install "purgedcv[dev]"
    ```

    Then the gates locally are:

    ```bash
    ruff check .
    black --check src tests tools examples/_lcl_harness.py
    mypy src tests
    pytest -q
    ```

=== "Documentation site"

    To build this documentation site locally:

    ```bash
    pip install "purgedcv[docs]"
    mkdocs serve
    ```

## From source

```bash
git clone https://github.com/eslazarev/purged-cross-validation.git
cd purged-cross-validation
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

The build backend is `hatchling`; no special steps are required.

## Verifying the install

```bash
python -c "import purgedcv; print(purgedcv.__version__)"
```

should print the installed version. The full public surface is in
[`purgedcv.__all__`](api.md).

## Time and group inputs

`prediction_times` and `evaluation_times` accept pandas `Series` or
`DatetimeIndex`, numpy `datetime64` or `timedelta64` arrays, Python lists of
datetime/Timestamp/timedelta values, and polars `Series`. They are coerced to a
1-D numpy array once at the boundary and must hold a datetime64 or timedelta64
dtype; tz-aware pandas input is normalized to UTC.

`groups` accepts the same container types but holds arbitrary 1-D labels
(integers, strings, categories), not timestamps.

polars is never imported, so it is not a runtime dependency, only an optional
convenience for callers who already hold a polars `Series`. The accepted-input
type aliases `TimesLike` (for times) and `ArrayLike1D` (for labels) are exported
from `purgedcv` for annotating your own code.
