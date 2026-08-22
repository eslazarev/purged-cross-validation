"""Pin the maintained 0.1.x public API surface of purgedcv.

These tests are an explicit contract: every name listed here is part of
the 0.1.x stable surface that users may rely on. Adding or removing
exports must be a deliberate, reviewed change accompanied by an update
to this file.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from inspect import signature

import numpy as np
import pandas as pd
import pytest

import purgedcv

EXPECTED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "__version__",
        "apply_embargo",
        "ArrayLike1D",
        "BaseTemporalSplitter",
        "CombinatorialPurgedCV",
        "CombinatoriallySymmetricCV",
        "default_backtest_metrics",
        "deflated_sharpe_ratio",
        "deflated_sharpe_ratio_full",
        "diagnostics",
        "effective_n_trials",
        "EmbargoViolationError",
        "GroupLeakageError",
        "horizons_overlap",
        "min_track_record_length",
        "minimum_backtest_length",
        "parse_horizon",
        "path_metrics",
        "probabilistic_sharpe_ratio",
        "probability_of_backtest_overfitting",
        "purge",
        "PurgedGroupKFold",
        "reconstruct_paths",
        "PurgedKFold",
        "TemporalCVError",
        "TemporalLeakageError",
        "TimesLike",
        "validate_times",
        "WalkForwardSplit",
    }
)

EXPECTED_DIAGNOSTICS: frozenset[str] = frozenset(
    {
        "assert_embargo_respected",
        "assert_groups_disjoint",
        "assert_no_temporal_leakage",
        "compute_overlap_fraction",
    }
)


class TestTopLevelAPI:
    def test_version_is_string(self) -> None:
        assert isinstance(purgedcv.__version__, str)
        assert len(purgedcv.__version__) > 0

    def test_all_attribute_matches_expected(self) -> None:
        """purgedcv.__all__ must equal the expected contract."""
        assert frozenset(purgedcv.__all__) == EXPECTED_TOP_LEVEL

    def test_expected_names_are_actually_present(self) -> None:
        """Every name in __all__ must resolve as an actual attribute."""
        for name in EXPECTED_TOP_LEVEL:
            assert hasattr(purgedcv, name), f"missing public attribute: {name}"

    def test_no_unexpected_public_names(self) -> None:
        """Catch accidental leakage of internal names at the package level."""
        actual_public = {
            name for name in dir(purgedcv) if not name.startswith("_") or name == "__version__"
        }
        unexpected = actual_public - EXPECTED_TOP_LEVEL
        # Subpackages and submodules that are part of the import system
        # (e.g., 'exceptions') may appear in dir(); allow only the
        # explicitly-imported submodule "diagnostics" and tolerate
        # implementation-detail submodules that are accessible via dotted
        # paths but not listed in __all__.
        allowed_extras = {"exceptions", "optuna_integration"}
        truly_unexpected = unexpected - allowed_extras
        assert not truly_unexpected, f"unexpected public names: {sorted(truly_unexpected)}"

    @pytest.mark.parametrize(
        "callable_obj",
        [
            purgedcv.apply_embargo,
            purgedcv.WalkForwardSplit,
            purgedcv.PurgedKFold,
            purgedcv.PurgedGroupKFold,
            purgedcv.CombinatorialPurgedCV,
            purgedcv.CombinatoriallySymmetricCV,
            purgedcv.probability_of_backtest_overfitting,
            purgedcv.diagnostics.assert_embargo_respected,
        ],
    )
    def test_embargo_modes_are_available_across_public_api(
        self, callable_obj: Callable[..., object]
    ) -> None:
        parameters = signature(callable_obj).parameters
        assert "embargo" in parameters
        assert "embargo_observations" in parameters
        assert "embargo_fraction" in parameters


class TestDiagnosticsSubmodule:
    def test_diagnostics_all_matches_expected(self) -> None:
        from purgedcv import diagnostics

        assert frozenset(diagnostics.__all__) == EXPECTED_DIAGNOSTICS

    def test_diagnostics_names_present(self) -> None:
        from purgedcv import diagnostics

        for name in EXPECTED_DIAGNOSTICS:
            assert hasattr(diagnostics, name), f"missing diagnostic: {name}"


class TestExceptionHierarchyExposed:
    def test_all_four_exceptions_are_purgedcv_errors(self) -> None:
        assert issubclass(purgedcv.TemporalLeakageError, purgedcv.TemporalCVError)
        assert issubclass(purgedcv.EmbargoViolationError, purgedcv.TemporalCVError)
        assert issubclass(purgedcv.GroupLeakageError, purgedcv.TemporalCVError)
        assert issubclass(purgedcv.TemporalCVError, ValueError)


@pytest.mark.e2e
def test_subprocess_can_import_every_public_name() -> None:
    """A fresh interpreter can resolve every name in the public contract."""
    names = sorted(EXPECTED_TOP_LEVEL)
    snippet = (
        "import purgedcv\n"
        + "\n".join(f"assert hasattr(purgedcv, {name!r}), {name!r}" for name in names)
        + "\nprint('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "OK"
    assert result.stderr == ""


def test_splitters_accept_numpy_times_no_typeerror() -> None:
    import purgedcv as pcv

    n = 18
    pred = pd.date_range("2024-01-01", periods=n, freq="D").to_numpy()
    evalu = pred + np.timedelta64(1, "D")
    X = np.zeros((n, 1))  # noqa: N806
    for splitter in (
        pcv.PurgedKFold(n_splits=3, prediction_times=pred, evaluation_times=evalu),
        pcv.CombinatorialPurgedCV(
            n_splits=4, n_test_groups=2, prediction_times=pred, evaluation_times=evalu
        ),
        pcv.WalkForwardSplit(
            n_splits=3, test_size=2, prediction_times=pred, evaluation_times=evalu
        ),
    ):
        assert len(list(splitter.split(X))) >= 1
