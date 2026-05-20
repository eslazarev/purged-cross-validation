# Installation

`purgedcv` is on PyPI and requires Python 3.10 or newer.

## From PyPI

The base install gets the splitters, metrics, and diagnostics:

```bash
pip install purgedcv
```

## Optional extras

Each extra is independent; install only what you need.

=== "Examples"

    To run the eleven worked notebooks in `examples/` (Jupyter,
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
