"""Unit tests for BaseTemporalSplitter (Domain D4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purgedcv._base import BaseTemporalSplitter
from purgedcv._typing import NDArrayAny
from purgedcv.exceptions import GroupLeakageError


def _times(n: int = 20, horizon_days: int = 1) -> tuple[pd.Series, pd.Series]:
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return pred, evalu


class _TwoFoldStub(BaseTemporalSplitter):
    """Minimal concrete subclass for contract testing."""

    def _iter_test_indices(self, n_samples: int) -> list[NDArrayAny]:
        # First half, second half.
        mid = n_samples // 2
        return [np.arange(mid), np.arange(mid, n_samples)]

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        return 2


class _GroupRespectingStub(BaseTemporalSplitter):
    """Yields one fold per unique group_id, using self._groups."""

    def _iter_test_indices(self, n_samples: int) -> list[NDArrayAny]:
        assert self._groups is not None  # by construction in the test
        groups_array = np.asarray(self._groups)
        return [np.where(groups_array == g)[0] for g in pd.unique(self._groups)]

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        assert self._groups is not None
        return len(pd.unique(self._groups))


class _LeakyStub(BaseTemporalSplitter):
    """Emits exactly one fold whose test indices straddle group boundaries."""

    def _iter_test_indices(self, n_samples: int) -> list[NDArrayAny]:
        return [np.array([5, 6, 15, 16])]

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        return 1


class _BadTestIndexStub(BaseTemporalSplitter):
    """Emits malformed test indices to verify base-class validation."""

    def _iter_test_indices(self, n_samples: int) -> list[NDArrayAny]:
        return [np.array([-1])]

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        return 1


class TestBaseTemporalSplitterSkeleton:
    def test_constructor_stores_purge_and_embargo(self) -> None:
        pred, evalu = _times()
        cv = _TwoFoldStub(
            purge_horizon="1D",
            embargo="1D",
            prediction_times=pred,
            evaluation_times=evalu,
        )
        assert cv.purge_horizon == pd.Timedelta(days=1)
        assert cv.embargo == pd.Timedelta(days=1)

    def test_constructor_accepts_zero_horizons_by_default(self) -> None:
        pred, evalu = _times()
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        assert cv.purge_horizon == pd.Timedelta(0)
        assert cv.embargo == pd.Timedelta(0)
        assert cv.embargo_observations is None
        assert cv.embargo_fraction is None

    def test_constructor_stores_observation_embargo(self) -> None:
        pred, evalu = _times()
        cv = _TwoFoldStub(
            prediction_times=pred,
            evaluation_times=evalu,
            embargo_observations=3,
        )
        assert cv.embargo == pd.Timedelta(0)
        assert cv.embargo_observations == 3
        assert cv.embargo_fraction is None

    def test_constructor_rejects_multiple_embargo_modes(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="mutually exclusive"):
            _TwoFoldStub(
                prediction_times=pred,
                evaluation_times=evalu,
                embargo="1D",
                embargo_fraction=0.1,
            )

    def test_constructor_validates_times(self) -> None:
        """Mismatched-length times must be rejected at construction."""
        pred = pd.Series(pd.date_range("2024-01-01", periods=10, freq="D"))
        evalu = pd.Series(pd.date_range("2024-01-02", periods=9, freq="D"))
        with pytest.raises(ValueError, match="length"):
            _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)

    def test_constructor_rejects_non_monotonic_prediction_times(self) -> None:
        """Index-based temporal splitters require rows sorted by prediction time."""
        pred = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-03"]))
        evalu = pred + pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="monotonic"):
            _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)

    def test_constructor_rejects_negative_horizons(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="non-negative"):
            _TwoFoldStub(
                purge_horizon=pd.Timedelta(days=-1),
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_get_n_splits_returns_subclass_value(self) -> None:
        pred, evalu = _times()
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        assert cv.get_n_splits() == 2

    def test_constructor_rejects_groups_length_mismatch(self) -> None:
        pred, evalu = _times(n=10)
        groups_wrong = pd.Series(np.zeros(9))
        with pytest.raises(ValueError, match="groups length"):
            _TwoFoldStub(
                prediction_times=pred,
                evaluation_times=evalu,
                groups=groups_wrong,
            )

    def test_constructor_rejects_missing_group_labels(self) -> None:
        pred, evalu = _times(n=10)
        groups = pd.Series([0, 0, 1, 1, np.nan, 2, 2, 3, 3, 4])
        with pytest.raises(ValueError, match="missing"):
            _TwoFoldStub(
                prediction_times=pred,
                evaluation_times=evalu,
                groups=groups,
            )

    def test_constructor_resets_series_index(self) -> None:
        """Stored times are coerced to numpy, so any custom pandas index on
        the input is dropped rather than leaking into the stored positions."""
        idx = pd.RangeIndex(start=5, stop=25)
        pred = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D"), index=idx)
        evalu = pred + pd.Timedelta(days=1)
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        assert isinstance(cv._prediction_times, np.ndarray)
        assert isinstance(cv._evaluation_times, np.ndarray)
        np.testing.assert_array_equal(cv._prediction_times, pred.to_numpy())
        np.testing.assert_array_equal(cv._evaluation_times, evalu.to_numpy())


class TestBaseTemporalSplitterSplit:
    def test_split_yields_purged_and_embargoed_folds(self) -> None:
        """A fold's training indices must have purge + embargo applied."""
        pred, evalu = _times(n=20, horizon_days=2)
        cv = _TwoFoldStub(
            purge_horizon="2D",
            embargo="1D",
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        folds = list(cv.split(X))
        assert len(folds) == 2
        for train_idx, test_idx in folds:
            # train and test must be disjoint
            assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0
            # diagnostics must pass with the same params
            from purgedcv.diagnostics import (
                assert_embargo_respected,
                assert_no_temporal_leakage,
            )

            assert_no_temporal_leakage(train_idx, test_idx, pred, evalu, purge_horizon="2D")
            assert_embargo_respected(train_idx, test_idx, pred, evalu, embargo="1D")

    def test_split_ignores_y_and_groups_args(self) -> None:
        """sklearn passes y and groups; the splitter accepts them but produces
        identical folds whether they are passed or not."""
        pred, evalu = _times()
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        X = np.zeros((20, 1))  # noqa: N806
        y = np.zeros(20)
        groups_arr = pd.Series(np.zeros(20))
        folds_with_y = list(cv.split(X, y=y, groups=groups_arr))
        folds_without = list(cv.split(X))
        assert len(folds_with_y) == len(folds_without)
        for (tr1, te1), (tr2, te2) in zip(folds_with_y, folds_without, strict=True):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_split_validates_X_length_matches_times(self) -> None:
        pred, evalu = _times(n=10)
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        X_wrong = np.zeros((15, 1))  # noqa: N806
        with pytest.raises(ValueError, match="length"):
            list(cv.split(X_wrong))

    def test_split_rejects_malformed_subclass_test_indices(self) -> None:
        pred, evalu = _times(n=10)
        cv = _BadTestIndexStub(prediction_times=pred, evaluation_times=evalu)
        with pytest.raises(ValueError, match="negative"):
            list(cv.split(np.zeros((10, 1))))

    def test_split_works_when_no_purge_or_embargo_needed(self) -> None:
        """With zero purge_horizon and embargo, train and test are simply
        complement sets — no rows dropped."""
        pred, evalu = _times(n=20, horizon_days=1)
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        X = np.zeros((20, 1))  # noqa: N806
        folds = list(cv.split(X))
        # _TwoFoldStub: fold 0 test=[0..9], fold 1 test=[10..19]. Under
        # zero horizon, half-open boundaries do not cause leakage, so all
        # complementary rows are kept.
        train0, test0 = folds[0]
        train1, test1 = folds[1]
        assert sorted(train0.tolist() + test0.tolist()) == list(range(20))
        assert sorted(train1.tolist() + test1.tolist()) == list(range(20))

    def test_split_applies_both_purge_and_embargo_independently(self) -> None:
        """Construct a fold where one row survives purge but must be removed
        by embargo. If apply_embargo is silently skipped, this test catches it.
        """
        # 20 daily samples, 1-day evaluation horizon.
        pred, evalu = _times(n=20, horizon_days=1)
        cv = _TwoFoldStub(
            prediction_times=pred,
            evaluation_times=evalu,
            purge_horizon="0D",  # purge does nothing
            embargo="2D",  # embargo must drop the first post-test rows
        )
        X = np.zeros((20, 1))  # noqa: N806
        folds = list(cv.split(X))
        # Fold 0: test = [0..10). test_eval_max = evalu[9] = Jan 11.
        # Candidate train = [10..20). With purge_horizon=0, none dropped by purge.
        # With embargo=2D, cutoff = Jan 13. Rows whose pred <= Jan 13 and pred >= Jan 11
        # are in embargo: that's rows 10 (pred=Jan 11), 11 (Jan 12), 12 (Jan 13).
        # So embargoed train = [13..20). Surviving 7 rows.
        train0, test0 = folds[0]
        assert set(test0.tolist()) == set(range(10))
        assert set(train0.tolist()) == set(range(13, 20))

    def test_split_applies_observation_embargo(self) -> None:
        pred, evalu = _times(n=20, horizon_days=1)
        cv = _TwoFoldStub(
            prediction_times=pred,
            evaluation_times=evalu,
            embargo_observations=3,
        )

        train0, test0 = next(cv.split(np.zeros((20, 1))))

        np.testing.assert_array_equal(test0, np.arange(10))
        np.testing.assert_array_equal(train0, np.arange(13, 20))


class TestBaseTemporalSplitterWithTimes:
    def test_returns_new_instance_with_rebound_times(self) -> None:
        pred, evalu = _times(n=20)
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        new_pred = pd.Series(pd.date_range("2025-01-01", periods=20, freq="D"))
        new_evalu = new_pred + pd.Timedelta(days=1)
        cv2 = cv.with_times(new_pred, new_evalu)
        # New instance, not the same object.
        assert cv2 is not cv
        # Times rebound.
        assert cv2._prediction_times[0] == pd.Timestamp("2025-01-01")
        # Original unchanged.
        assert cv._prediction_times[0] == pd.Timestamp("2024-01-01")
        # Other attributes preserved.
        assert cv2.purge_horizon == cv.purge_horizon
        assert cv2.embargo == cv.embargo

    def test_preserves_positional_embargo_when_rebinding_times(self) -> None:
        pred, evalu = _times(n=20)
        cv = _TwoFoldStub(
            prediction_times=pred,
            evaluation_times=evalu,
            embargo_fraction=0.1,
        )
        new_pred = pd.Series(pd.date_range("2025-01-01", periods=20, freq="D"))
        cv2 = cv.with_times(new_pred, new_pred + pd.Timedelta(days=1))

        assert cv2.embargo_observations is None
        assert cv2.embargo_fraction == 0.1

    def test_rejects_length_mismatch(self) -> None:
        pred, evalu = _times(n=20)
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        new_pred = pd.Series(pd.date_range("2025-01-01", periods=15, freq="D"))
        new_evalu = new_pred + pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="length"):
            cv.with_times(new_pred, new_evalu)

    def test_rejects_invalid_times(self) -> None:
        """with_times must reject malformed input (NaT, eval < pred, etc.)."""
        pred, evalu = _times(n=20)
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        bad_pred = pd.Series(pd.date_range("2025-01-02", periods=20, freq="D"))
        bad_evalu = pd.Series(pd.date_range("2025-01-01", periods=20, freq="D"))
        with pytest.raises(ValueError):
            cv.with_times(bad_pred, bad_evalu)

    def test_with_times_rejects_non_monotonic_prediction_times(self) -> None:
        pred, evalu = _times(n=20)
        cv = _TwoFoldStub(prediction_times=pred, evaluation_times=evalu)
        new_pred = pd.Series(pd.to_datetime(["2025-01-02", "2025-01-01"] * 10))
        new_evalu = new_pred + pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="monotonic"):
            cv.with_times(new_pred, new_evalu)

    def test_with_times_preserves_groups(self) -> None:
        """When groups were bound at construction, with_times must preserve
        them so group-aware splitters (PurgedGroupKFold in B7) keep their
        state through a rebind."""
        pred, evalu = _times(n=20)
        groups = pd.Series(np.repeat([0, 1, 2, 3], 5))
        cv = _TwoFoldStub(
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        new_pred = pd.Series(pd.date_range("2030-01-01", periods=20, freq="D"))
        new_evalu = new_pred + pd.Timedelta(days=1)
        cv2 = cv.with_times(new_pred, new_evalu)
        # _groups attribute is preserved by reference (shallow copy).
        assert cv2._groups is not None
        assert cv._groups is not None
        np.testing.assert_array_equal(cv2._groups, cv._groups)


class TestBaseTemporalSplitterGroups:
    def test_groups_bound_at_construction_enforced(self) -> None:
        """A subclass whose folds are already group-disjoint produces no
        GroupLeakageError when split() runs the disjointness check."""
        pred, evalu = _times(n=20)
        groups = pd.Series(np.repeat([0, 1, 2, 3], 5))

        cv = _GroupRespectingStub(
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        X = np.zeros((20, 1))  # noqa: N806
        # Should not raise. Verify each fold is non-trivial and groups are disjoint
        # at the membership level (independent of the internal check that split()
        # also runs — that is verified by test_groups_leak_raises).
        folds = list(cv.split(X))
        assert len(folds) == 4
        for train_idx, test_idx in folds:
            train_groups = set(groups.iloc[train_idx].tolist())
            test_groups = set(groups.iloc[test_idx].tolist())
            assert train_groups & test_groups == set()
            assert len(train_idx) > 0

    def test_groups_leak_raises(self) -> None:
        """When a subclass produces a fold that mixes groups across train
        and test, the splitter must raise GroupLeakageError."""
        pred, evalu = _times(n=20)
        groups = pd.Series(np.repeat([0, 1], 10))

        cv = _LeakyStub(
            prediction_times=pred,
            evaluation_times=evalu,
            groups=groups,
        )
        X = np.zeros((20, 1))  # noqa: N806
        with pytest.raises(GroupLeakageError):
            list(cv.split(X))


def test_splitter_from_numpy_inputs_matches_pandas() -> None:
    from purgedcv import PurgedKFold

    n = 30
    pred_pd = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu_pd = pred_pd + pd.Timedelta(days=2)
    X = np.zeros((n, 1))  # noqa: N806

    sp_pd = PurgedKFold(n_splits=3, prediction_times=pred_pd, evaluation_times=evalu_pd)
    sp_np = PurgedKFold(
        n_splits=3, prediction_times=pred_pd.to_numpy(), evaluation_times=evalu_pd.to_numpy()
    )
    for (tr_pd, te_pd), (tr_np, te_np) in zip(sp_pd.split(X), sp_np.split(X), strict=True):
        np.testing.assert_array_equal(tr_pd, tr_np)
        np.testing.assert_array_equal(te_pd, te_np)
