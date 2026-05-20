"""purged-cross-validation: scikit-learn-compatible cross-validation for time-series ML."""

from purgedcv import diagnostics
from purgedcv._base import BaseTemporalSplitter
from purgedcv._cpcv import CombinatorialPurgedCV
from purgedcv._embargo import apply_embargo
from purgedcv._metrics import (
    deflated_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from purgedcv._paths import reconstruct_paths
from purgedcv._purge import purge
from purgedcv._purged_kfold import PurgedGroupKFold, PurgedKFold
from purgedcv._time import horizons_overlap, parse_horizon, validate_times
from purgedcv._walk_forward import WalkForwardSplit
from purgedcv.exceptions import (
    EmbargoViolationError,
    GroupLeakageError,
    TemporalCVError,
    TemporalLeakageError,
)

__version__ = "0.0.8"

__all__ = [
    "BaseTemporalSplitter",
    "CombinatorialPurgedCV",
    "EmbargoViolationError",
    "GroupLeakageError",
    "PurgedGroupKFold",
    "PurgedKFold",
    "TemporalCVError",
    "TemporalLeakageError",
    "WalkForwardSplit",
    "__version__",
    "apply_embargo",
    "deflated_sharpe_ratio",
    "diagnostics",
    "horizons_overlap",
    "min_track_record_length",
    "parse_horizon",
    "probabilistic_sharpe_ratio",
    "purge",
    "reconstruct_paths",
    "validate_times",
]
