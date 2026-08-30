"""Unit tests for the per-fold splitter audit report (A6)."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from purgedcv import (
    CombinatorialPurgedCV,
    PurgedGroupKFold,
    PurgedKFold,
    WalkForwardSplit,
    audit_splitter,
)
from purgedcv._base import BaseTemporalSplitter
from purgedcv._typing import NDArrayAny
from purgedcv.exceptions import GroupLeakageError

EXPECTED_COLUMNS = [
    "fold",
    "candidate_train_size",
    "final_train_size",
    "test_size",
    "train_nonempty",
    "rows_removed_by_purge",
    "rows_removed_by_embargo",
    "rows_removed_by_finalization",
    "rows_added_by_finalization",
    "candidate_overlap_fraction",
    "final_overlap_fraction",
    "temporal_leakage_free",
    "train_block_count",
    "test_block_count",
    "train_time_envelope_start",
    "train_time_envelope_end",
    "test_time_envelope_start",
    "test_time_envelope_end",
    "groups_disjoint",
]


def _times(n: int = 20) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    prediction_times = pd.date_range("2024-01-01", periods=n, freq="D")
    return prediction_times, prediction_times + pd.Timedelta(days=1)


def test_audit_reports_real_purge_and_embargo_stages() -> None:
    pred, evalu = _times()
    cv = PurgedKFold(
        4,
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon="1D",
        embargo_observations=2,
    )

    report = audit_splitter(cv, np.zeros((20, 1)))

    assert list(report.columns) == EXPECTED_COLUMNS
    assert len(report) == cv.get_n_splits()
    first = report.iloc[0]
    assert first["candidate_train_size"] == 15
    assert first["rows_removed_by_purge"] == 1
    assert first["rows_removed_by_embargo"] == 1
    assert first["rows_removed_by_finalization"] == 0
    assert first["rows_added_by_finalization"] == 0
    assert first["final_train_size"] == 13
    assert bool(first["train_nonempty"])
    assert first["candidate_overlap_fraction"] == pytest.approx(1 / 15)
    assert first["final_overlap_fraction"] == 0.0
    assert bool(first["temporal_leakage_free"])
    assert first["train_block_count"] == 1
    assert first["test_block_count"] == 1
    assert first["train_time_envelope_start"] == pd.Timestamp("2024-01-08")
    assert first["train_time_envelope_end"] == pd.Timestamp("2024-01-21")
    assert first["test_time_envelope_start"] == pd.Timestamp("2024-01-01")
    assert first["test_time_envelope_end"] == pd.Timestamp("2024-01-06")
    assert first["groups_disjoint"] is None

    accounted = (
        report["final_train_size"]
        + report["rows_removed_by_purge"]
        + report["rows_removed_by_embargo"]
        + report["rows_removed_by_finalization"]
        - report["rows_added_by_finalization"]
    )
    pd.testing.assert_series_equal(
        accounted,
        report["candidate_train_size"],
        check_names=False,
    )


def test_audit_matches_final_indices_yielded_by_split() -> None:
    pred, evalu = _times(n=24)
    cv = CombinatorialPurgedCV(
        4,
        2,
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon="1D",
        embargo_fraction=0.1,
    )
    X = np.zeros((24, 2))  # noqa: N806

    report = audit_splitter(cv, X)
    folds = list(cv.split(X))

    assert len(report) == len(folds) == cv.get_n_splits()
    assert report["final_train_size"].tolist() == [len(train) for train, _ in folds]
    assert report["test_size"].tolist() == [len(test) for _, test in folds]
    assert report["final_overlap_fraction"].eq(0.0).all()


def test_sliding_window_removals_are_reported_as_finalization() -> None:
    pred, evalu = _times()
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
    assert report["rows_removed_by_finalization"].tolist() == [8, 10, 12]
    assert report["rows_added_by_finalization"].tolist() == [0, 0, 0]


def test_group_disjointness_is_reported_when_groups_are_bound() -> None:
    pred, evalu = _times(n=12)
    groups = np.repeat(["a", "b", "c", "d"], 3)
    cv = PurgedGroupKFold(
        2,
        prediction_times=pred,
        evaluation_times=evalu,
        groups=groups,
    )

    report = audit_splitter(cv, np.zeros((12, 1)))

    assert report["groups_disjoint"].eq(True).all()


class _GroupLeakingSplitter(BaseTemporalSplitter):
    def _iter_test_indices(self, n_samples: int) -> list[NDArrayAny]:
        return [np.array([0], dtype=np.int64)]

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        return 1


class _ReintroducingSplitter(_GroupLeakingSplitter):
    def _finalize_train_idx(
        self,
        train_idx: NDArrayAny,
        test_idx: NDArrayAny,
    ) -> NDArrayAny:
        return np.append(train_idx, test_idx[0])


class _ReplacingSplitter(_GroupLeakingSplitter):
    def _finalize_train_idx(
        self,
        train_idx: NDArrayAny,
        test_idx: NDArrayAny,
    ) -> NDArrayAny:
        return np.append(train_idx[1:], test_idx[0])


class _DuplicatingSplitter(_GroupLeakingSplitter):
    def _finalize_train_idx(
        self,
        train_idx: NDArrayAny,
        test_idx: NDArrayAny,
    ) -> NDArrayAny:
        return np.concatenate([train_idx, train_idx[:2]])


class _SplitOverride(_GroupLeakingSplitter):
    def split(
        self,
        X: NDArrayAny | pd.DataFrame,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> Iterator[tuple[NDArrayAny, NDArrayAny]]:
        for train_idx, test_idx in super().split(X, y=y, groups=groups):
            yield np.append(train_idx, test_idx[0]), test_idx


def test_group_leakage_is_reported_without_raising() -> None:
    pred, evalu = _times(n=6)
    cv = _GroupLeakingSplitter(
        prediction_times=pred,
        evaluation_times=evalu,
        groups=np.repeat("same-group", 6),
    )
    X = np.zeros((6, 1))  # noqa: N806

    report = audit_splitter(cv, X)

    assert not bool(report.loc[0, "groups_disjoint"])
    with pytest.raises(GroupLeakageError):
        list(cv.split(X))


def test_temporal_status_catches_custom_finalizer_reintroducing_test_row() -> None:
    pred, evalu = _times(n=6)
    cv = _ReintroducingSplitter(
        prediction_times=pred,
        evaluation_times=evalu,
    )

    report = audit_splitter(cv, np.zeros((6, 1)))

    assert report.loc[0, "final_overlap_fraction"] != 0.0
    assert not bool(report.loc[0, "temporal_leakage_free"])
    assert report.loc[0, "rows_removed_by_finalization"] == 0
    assert report.loc[0, "rows_added_by_finalization"] == 1


def test_finalizer_replacement_reports_one_removal_and_one_addition() -> None:
    pred, evalu = _times(n=6)
    cv = _ReplacingSplitter(
        prediction_times=pred,
        evaluation_times=evalu,
    )

    report = audit_splitter(cv, np.zeros((6, 1)))

    assert report.loc[0, "final_train_size"] == 5
    assert report.loc[0, "rows_removed_by_finalization"] == 1
    assert report.loc[0, "rows_added_by_finalization"] == 1
    assert not bool(report.loc[0, "temporal_leakage_free"])


def test_rejects_duplicate_indices_from_custom_finalizer() -> None:
    pred, evalu = _times(n=6)
    cv = _DuplicatingSplitter(
        prediction_times=pred,
        evaluation_times=evalu,
    )

    with pytest.raises(ValueError, match=r"final_train_idx.*duplicate"):
        audit_splitter(cv, np.zeros((6, 1)))
    with pytest.raises(ValueError, match=r"final_train_idx.*duplicate"):
        list(cv.split(np.zeros((6, 1))))


def test_rejects_subclass_that_overrides_split() -> None:
    pred, evalu = _times(n=8)
    cv = _SplitOverride(
        prediction_times=pred,
        evaluation_times=evalu,
    )
    X = np.zeros((8, 1))  # noqa: N806

    real_train, real_test = next(cv.split(X))
    assert real_test[0] in real_train
    with pytest.raises(TypeError, match=r"overrides split\(\)"):
        audit_splitter(cv, X)


def test_empty_final_train_has_missing_bounds() -> None:
    pred, evalu = _times(n=4)
    cv = PurgedKFold(
        2,
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon="100D",
    )

    report = audit_splitter(cv, np.zeros((4, 1)))

    assert report["final_train_size"].eq(0).all()
    assert report["train_nonempty"].eq(False).all()
    assert report["train_time_envelope_start"].isna().all()
    assert report["train_time_envelope_end"].isna().all()
    assert report["final_overlap_fraction"].eq(0.0).all()
    assert report["temporal_leakage_free"].eq(True).all()


def test_timedelta_empty_envelope_preserves_timedelta_dtype() -> None:
    pred = pd.timedelta_range(start="0D", periods=4, freq="D")
    evalu = pred + pd.Timedelta(days=1)
    cv = PurgedKFold(
        2,
        prediction_times=pred,
        evaluation_times=evalu,
        purge_horizon="100D",
    )

    report = audit_splitter(cv, np.zeros((4, 1)))

    assert str(report["train_time_envelope_start"].dtype).startswith("timedelta64")
    assert not (report["train_time_envelope_start"] < pd.Timedelta(days=1)).any()


def test_cpcv_envelopes_expose_disjoint_block_counts() -> None:
    pred, evalu = _times(n=24)
    cv = CombinatorialPurgedCV(
        4,
        2,
        prediction_times=pred,
        evaluation_times=evalu,
    )

    report = audit_splitter(cv, np.zeros((24, 1)))
    row = report.loc[report["test_block_count"].eq(2)].iloc[0]

    assert row["train_block_count"] == 2
    assert row["train_time_envelope_start"] < row["test_time_envelope_end"]
    assert row["test_time_envelope_start"] < row["train_time_envelope_end"]
    assert row["final_overlap_fraction"] == 0.0


def test_rejects_non_temporal_splitter_and_wrong_x_length() -> None:
    with pytest.raises(TypeError, match="BaseTemporalSplitter"):
        audit_splitter(object(), np.zeros((3, 1)))  # type: ignore[arg-type]

    pred, evalu = _times(n=6)
    cv = PurgedKFold(2, prediction_times=pred, evaluation_times=evalu)
    with pytest.raises(ValueError, match="X length"):
        audit_splitter(cv, np.zeros((5, 1)))
