"""End-to-end tests for the public ``audit_splitter`` entry point.

User stories:
- A researcher wants to inspect how many observations each fold loses to
  purge and embargo before committing compute to a large model search.
- A walk-forward user wants window truncation reported separately from
  leakage controls so an unexpectedly small train set is explainable.
- An installed-package consumer imports the helper from a fresh interpreter
  and receives the same stable DataFrame schema as in-process callers.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from purgedcv import PurgedKFold, WalkForwardSplit, audit_splitter


@pytest.mark.e2e
def test_user_story_preflight_fold_removal_report() -> None:
    """A model-search preflight explains purge and embargo losses per fold."""
    pred = pd.date_range("2024-01-01", periods=20, freq="D")
    evalu = pred + pd.Timedelta(days=1)
    cv = PurgedKFold(
        4,
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon="1D",
        embargo_observations=2,
    )

    report = audit_splitter(cv, np.zeros((20, 3)))

    assert len(report) == 4
    assert report.loc[0, "candidate_train_size"] == 15
    assert report.loc[0, "rows_removed_by_purge"] == 1
    assert report.loc[0, "rows_removed_by_embargo"] == 1
    assert report.loc[0, "final_train_size"] == 13
    assert report.loc[0, "candidate_overlap_fraction"] == pytest.approx(1 / 15)
    assert report["final_overlap_fraction"].eq(0.0).all()


@pytest.mark.e2e
def test_user_story_sliding_window_is_not_mislabeled_as_leakage_control() -> None:
    """Window truncation has its own count in a rolling deployment setup."""
    pred = pd.date_range("2024-01-01", periods=20, freq="D")
    evalu = pred + pd.Timedelta(days=1)
    cv = WalkForwardSplit(
        3,
        2,
        train_size=5,
        window="sliding",
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon="1D",
    )

    report = audit_splitter(cv, np.zeros((20, 1)))

    assert report["final_train_size"].tolist() == [5, 5, 5]
    assert report["rows_removed_by_purge"].tolist() == [1, 1, 1]
    assert report["rows_removed_by_embargo"].tolist() == [0, 0, 0]
    assert report["rows_removed_by_window"].tolist() == [8, 10, 12]
    assert report["rows_added_by_finalization"].tolist() == [0, 0, 0]


@pytest.mark.e2e
def test_subprocess_fresh_interpreter_can_audit_splitter() -> None:
    snippet = textwrap.dedent("""\
        import numpy as np
        import pandas as pd
        from purgedcv import PurgedKFold, audit_splitter, diagnostics

        pred = pd.date_range("2024-01-01", periods=12, freq="D")
        evalu = pred + pd.Timedelta(days=1)
        cv = PurgedKFold(
            3,
            prediction_times=pred,
            evaluation_times=evalu,
            purge_horizon="1D",
        )
        report = audit_splitter(cv, np.zeros((12, 1)))
        assert diagnostics.audit_splitter is audit_splitter
        assert report.shape[0] == 3
        assert "candidate_overlap_fraction" in report.columns
        assert "final_overlap_fraction" in report.columns
        assert "rows_added_by_finalization" in report.columns
        assert report["final_overlap_fraction"].eq(0.0).all()
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
