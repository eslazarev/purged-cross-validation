"""End-to-end tests for PurgedGroupKFold."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import PurgedGroupKFold
from purgedcv.exceptions import GroupLeakageError  # noqa: F401


@pytest.mark.e2e
def test_user_story_clinical_patient_split() -> None:
    """A clinical researcher splits 6 patients x 5 observations each into
    3 folds. No patient appears in both train and test of any fold."""
    pred = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evalu = pred + pd.Timedelta(days=1)
    patient_ids = pd.Series(np.repeat([0, 1, 2, 3, 4, 5], 5))
    cv = PurgedGroupKFold(
        n_splits=3,
        purge_horizon="2D",
        embargo="1D",
        prediction_times=pred,
        evaluation_times=evalu,
        groups=patient_ids,
    )
    X = np.zeros((30, 1))  # noqa: N806
    folds = list(cv.split(X))
    assert len(folds) == 3
    seen_test_patients: set[int] = set()
    for train_idx, test_idx in folds:
        train_patients = set(patient_ids.iloc[train_idx].tolist())
        test_patients = set(patient_ids.iloc[test_idx].tolist())
        assert train_patients & test_patients == set()
        seen_test_patients.update(test_patients)
    # All 6 patients are seen in test across the 3 folds.
    assert seen_test_patients == {0, 1, 2, 3, 4, 5}


@pytest.mark.e2e
def test_subprocess_purged_group_kfold_smoke() -> None:
    snippet = textwrap.dedent(
        """\
        import numpy as np
        import pandas as pd
        from purgedcv import PurgedGroupKFold
        from purgedcv.exceptions import GroupLeakageError
        pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"))
        evalu = pred + pd.Timedelta(days=1)
        groups = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        cv = PurgedGroupKFold(
            n_splits=2,
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        folds = list(cv.split(np.zeros((20, 1))))
        assert len(folds) == 2
        # Verify the GroupLeakageError import is alive at runtime:
        assert GroupLeakageError.__name__ == "GroupLeakageError"
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "OK"
    assert result.stderr == ""
