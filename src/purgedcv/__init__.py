"""purged-cross-validation: scikit-learn-compatible cross-validation for time-series ML."""

from purgedcv import diagnostics
from purgedcv._base import BaseTemporalSplitter
from purgedcv._cpcv import CombinatoriallySymmetricCV, CombinatorialPurgedCV
from purgedcv._embargo import apply_embargo
from purgedcv._metrics import (
    DSRDiagnostics,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_full,
    effective_n_trials,
    min_track_record_length,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
)
from purgedcv._path_metrics import PathMetricFn, default_backtest_metrics, path_metrics
from purgedcv._paths import reconstruct_paths
from purgedcv._pbo import PBOResult, PerformanceMetric, probability_of_backtest_overfitting
from purgedcv._purge import purge
from purgedcv._purged_kfold import PurgedGroupKFold, PurgedKFold
from purgedcv._time import HorizonLike, horizons_overlap, parse_horizon, validate_times
from purgedcv._typing import ArrayLike1D, TimesLike
from purgedcv._walk_forward import WalkForwardSplit
from purgedcv.diagnostics import audit_splitter
from purgedcv.exceptions import (
    EmbargoViolationError,
    GroupLeakageError,
    TemporalCVError,
    TemporalLeakageError,
)

__version__ = "0.1.5"

__all__ = [
    "ArrayLike1D",
    "BaseTemporalSplitter",
    "CombinatorialPurgedCV",
    "CombinatoriallySymmetricCV",
    "DSRDiagnostics",
    "EmbargoViolationError",
    "GroupLeakageError",
    "HorizonLike",
    "PBOResult",
    "PathMetricFn",
    "PerformanceMetric",
    "PurgedGroupKFold",
    "PurgedKFold",
    "TemporalCVError",
    "TemporalLeakageError",
    "TimesLike",
    "WalkForwardSplit",
    "__version__",
    "apply_embargo",
    "audit_splitter",
    "default_backtest_metrics",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_full",
    "diagnostics",
    "effective_n_trials",
    "horizons_overlap",
    "min_track_record_length",
    "minimum_backtest_length",
    "parse_horizon",
    "path_metrics",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purge",
    "reconstruct_paths",
    "validate_times",
]
