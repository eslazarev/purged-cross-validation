"""The purgedcv rows in ``tools/competitor_benchmark.py`` purge exactly the label horizon.

Each label covers the next ``H`` rows, so a training row ``H`` rows away from a
test row shares no noise draw with it and is safe to keep. The builders used
to pass ``purge_horizon`` on top of ``evaluation_times``, which pushed the
closest training row ``2H`` away and threw out ``H`` safe rows on every side.
These tests pin the gap at ``H`` so that over-purging cannot come back.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from purgedcv.diagnostics import compute_overlap_fraction

BENCHMARK = Path(__file__).resolve().parents[1] / "tools" / "competitor_benchmark.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("competitor_benchmark", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cb = _load()

Dataset = tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]
Folds = list[tuple[np.ndarray, np.ndarray]]

BUILDERS: dict[str, Callable[..., Folds]] = {
    "PurgedKFold": cb._pcv_purged,
    "WalkForwardSplit": cb._pcv_walk,
    "CombinatorialPurgedCV": cb._pcv_cpcv,
}


def _gaps(train: np.ndarray, test: np.ndarray) -> tuple[int | None, int | None]:
    """Distance from the test set to the nearest training row before and after it."""
    test = np.sort(test)
    pos = np.searchsorted(test, train)
    before = train[pos < len(test)]
    after = train[pos > 0]
    gap_before = int((test[np.searchsorted(test, before)] - before).min()) if len(before) else None
    gap_after = int((after - test[np.searchsorted(test, after) - 1]).min()) if len(after) else None
    return gap_before, gap_after


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    data: Dataset = cb.make_dataset()
    return data


@pytest.mark.parametrize("name", list(BUILDERS))
def test_purgedcv_rows_have_zero_label_overlap(name: str, dataset: Dataset) -> None:
    features, y, pred, evalu = dataset
    folds = BUILDERS[name](features, y, pred, evalu)
    assert folds
    for train, test in folds:
        assert len(train) and len(test)
        assert compute_overlap_fraction(train, test, pred, evalu) == 0.0


@pytest.mark.parametrize("name", list(BUILDERS))
def test_purgedcv_rows_keep_the_row_right_before_the_test_fold(name: str, dataset: Dataset) -> None:
    features, y, pred, evalu = dataset
    gaps = [_gaps(train, test)[0] for train, test in BUILDERS[name](features, y, pred, evalu)]
    gaps = [g for g in gaps if g is not None]
    assert gaps, "no fold has training rows before its test rows"
    assert gaps == [cb.H] * len(gaps)


def test_cpcv_row_keeps_the_row_right_after_each_test_group(dataset: Dataset) -> None:
    features, y, pred, evalu = dataset
    gaps = [_gaps(train, test)[1] for train, test in cb._pcv_cpcv(features, y, pred, evalu)]
    gaps = [g for g in gaps if g is not None]
    assert gaps
    assert gaps == [cb.H] * len(gaps)


def test_purged_kfold_row_adds_only_the_embargo_after_the_test_fold(dataset: Dataset) -> None:
    features, y, pred, evalu = dataset
    gaps = [_gaps(train, test)[1] for train, test in cb._pcv_purged(features, y, pred, evalu)]
    gaps = [g for g in gaps if g is not None]
    assert gaps
    # H rows of label overlap plus the H-day embargo, nothing more.
    assert gaps == [2 * cb.H + 1] * len(gaps)
