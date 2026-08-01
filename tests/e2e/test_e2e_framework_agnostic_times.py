"""End-to-end tests for framework-agnostic time inputs.

User stories:
- A quant whose pipeline is numpy-only builds a purged splitter without ever
  constructing a pandas object.
- A quant who has migrated to polars extracts two time columns and runs CPCV
  to completion.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest


@pytest.mark.e2e
def test_subprocess_numpy_only_split_runs_clean() -> None:
    snippet = textwrap.dedent("""\
        import numpy as np
        from purgedcv import PurgedKFold
        n = 24
        pred = np.arange("2024-01-01", "2024-01-25", dtype="datetime64[D]")
        evalu = pred + np.timedelta64(1, "D")
        X = np.zeros((n, 1))
        sp = PurgedKFold(n_splits=3, prediction_times=pred, evaluation_times=evalu, purge_horizon="1D")
        folds = list(sp.split(X))
        assert len(folds) == 3
        for tr, te in folds:
            assert set(tr).isdisjoint(set(te))
        print("OK")
        """)
    result = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "OK"
    assert result.stderr == ""


@pytest.mark.e2e
def test_user_story_polars_quant_runs_cpcv() -> None:
    pl = pytest.importorskip("polars")
    from purgedcv import CombinatorialPurgedCV

    n = 60
    ts = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pl.DataFrame(
        {
            "prediction_time": ts,
            "evaluation_time": ts + pd.Timedelta(days=3),
            "feature": np.arange(n, dtype=float),
        }
    )
    splitter = CombinatorialPurgedCV(
        n_splits=6,
        n_test_groups=2,
        prediction_times=df["prediction_time"],
        evaluation_times=df["evaluation_time"],
        purge_horizon="1D",
    )
    X = df.select("feature").to_numpy()  # noqa: N806
    folds = list(splitter.split(X))
    assert len(folds) == 15  # C(6, 2)
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))


@pytest.mark.e2e
def test_user_story_numpy_and_pandas_agree() -> None:
    from purgedcv import PurgedKFold

    n = 40
    pred_pd = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu_pd = pred_pd + pd.Timedelta(days=2)
    X = np.zeros((n, 1))  # noqa: N806
    a = list(PurgedKFold(n_splits=5, prediction_times=pred_pd, evaluation_times=evalu_pd).split(X))
    b = list(
        PurgedKFold(
            n_splits=5, prediction_times=pred_pd.to_numpy(), evaluation_times=evalu_pd.to_numpy()
        ).split(X)
    )
    for (tr_a, te_a), (tr_b, te_b) in zip(a, b, strict=True):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(te_a, te_b)
