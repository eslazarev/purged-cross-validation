"""End-to-end tests for ``assert_groups_disjoint``.

User stories:
- A clinical researcher with multiple observations per patient must verify
  that no patient appears in both training and test, to prevent
  patient-level information leakage.
- A reviewer wants the error message to name a representative leaking
  group so they can investigate the data pipeline that introduced it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import GroupLeakageError
from purgedcv.diagnostics import assert_groups_disjoint


@pytest.mark.e2e
def test_user_story_patient_split_clean() -> None:
    """6 patients, 5 observations each (30 rows). Train on patients 0-3,
    test on patients 4-5. No patient appears in both sides."""
    patient_ids = pd.Series(np.repeat([0, 1, 2, 3, 4, 5], 5))
    train_idx = np.arange(0, 20)  # patients 0-3
    test_idx = np.arange(20, 30)  # patients 4-5
    assert_groups_disjoint(train_idx, test_idx, patient_ids)


@pytest.mark.e2e
def test_user_story_patient_leakage_detected_with_specific_id() -> None:
    """Patient 3 has rows in both train and test — error names patient 3."""
    patient_ids = pd.Series(np.repeat([0, 1, 2, 3, 4, 5], 5))
    train_idx = np.arange(
        0, 18
    )  # patients 0-3 (rows 0-17 → patients 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3)
    test_idx = np.arange(15, 25)  # patients 3-4 (rows 15-24 → patient 3 appears in both)
    with pytest.raises(GroupLeakageError) as exc_info:
        assert_groups_disjoint(train_idx, test_idx, patient_ids)
    assert "group 3" in str(exc_info.value)


@pytest.mark.e2e
def test_user_story_string_identifiers_work() -> None:
    """Asset symbols are strings, not integers."""
    symbols = pd.Series(["AAPL", "AAPL", "MSFT", "MSFT", "GOOG", "GOOG"])
    train_idx = np.array([0, 1, 2])  # AAPL + MSFT
    test_idx = np.array([2, 3, 4, 5])  # MSFT + GOOG → MSFT leaks
    with pytest.raises(GroupLeakageError, match="MSFT"):
        assert_groups_disjoint(train_idx, test_idx, symbols)


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_group_check() -> None:
    snippet = textwrap.dedent("""\
        import numpy as np
        import pandas as pd
        from purgedcv import diagnostics, GroupLeakageError

        groups = pd.Series([0, 0, 1, 1, 2, 2])
        diagnostics.assert_groups_disjoint(
            np.array([0, 1]), np.array([2, 3]), groups
        )
        try:
            diagnostics.assert_groups_disjoint(
                np.array([0, 4]), np.array([2, 4]), groups
            )
            raise AssertionError("should have raised")
        except GroupLeakageError:
            pass
        print("OK")
        """)
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "OK"
    assert result.stderr == ""
