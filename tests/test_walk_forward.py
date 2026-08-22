"""Unit tests for WalkForwardSplit (D5.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purgedcv._walk_forward import WalkForwardSplit


def _times(n: int = 20, horizon_days: int = 1) -> tuple[pd.Series, pd.Series]:
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return pred, evalu


class TestWalkForwardSplit:
    def test_post_test_observation_embargo_is_noop_by_design(self) -> None:
        pred, evalu = _times(n=24)
        baseline = WalkForwardSplit(
            n_splits=3,
            test_size=4,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        with_embargo = WalkForwardSplit(
            n_splits=3,
            test_size=4,
            prediction_times=pred,
            evaluation_times=evalu,
            embargo_observations=5,
        )
        X = np.zeros((24, 1))  # noqa: N806

        for (base_train, base_test), (emb_train, emb_test) in zip(
            baseline.split(X), with_embargo.split(X), strict=True
        ):
            np.testing.assert_array_equal(emb_train, base_train)
            np.testing.assert_array_equal(emb_test, base_test)

    def test_yields_n_splits_folds(self) -> None:
        pred, evalu = _times(n=20)
        cv = WalkForwardSplit(
            n_splits=5,
            test_size=2,
            window="expanding",
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        folds = list(cv.split(X))
        assert len(folds) == 5
        assert cv.get_n_splits() == 5

    def test_test_folds_are_contiguous_and_in_order(self) -> None:
        pred, evalu = _times(n=20)
        cv = WalkForwardSplit(
            n_splits=5,
            test_size=2,
            window="expanding",
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((20, 1))  # noqa: N806
        test_starts = []
        for _, test_idx in cv.split(X):
            assert np.all(np.diff(test_idx) == 1)  # contiguous
            test_starts.append(test_idx[0])
        assert test_starts == sorted(test_starts)  # monotone forward

    def test_sliding_window_train_size_is_constant(self) -> None:
        """Sliding window means each fold's training set has the same size
        (or differs by at most one due to integer math)."""
        pred, evalu = _times(n=30)
        cv = WalkForwardSplit(
            n_splits=5,
            test_size=2,
            window="sliding",
            train_size=10,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        X = np.zeros((30, 1))  # noqa: N806
        train_sizes = [len(train) for train, _ in cv.split(X)]
        # With no purge or embargo, all sliding-window training sets are
        # exactly train_size long.
        assert len(set(train_sizes)) == 1
        assert train_sizes[0] == 10

    def test_sklearn_timeseriessplit_equivalence_when_no_purge(self) -> None:
        """With purge_horizon=embargo=0 and expanding mode, WalkForwardSplit
        must produce the same folds as sklearn's TimeSeriesSplit."""
        from sklearn.model_selection import TimeSeriesSplit

        pred, evalu = _times(n=20)
        cv_our = WalkForwardSplit(
            n_splits=5,
            test_size=2,
            window="expanding",
            prediction_times=pred,
            evaluation_times=evalu,
        )
        cv_sk = TimeSeriesSplit(n_splits=5, test_size=2)
        X = np.zeros((20, 1))  # noqa: N806
        for (our_train, our_test), (sk_train, sk_test) in zip(
            cv_our.split(X), cv_sk.split(X), strict=True
        ):
            np.testing.assert_array_equal(our_train, sk_train)
            np.testing.assert_array_equal(our_test, sk_test)

    def test_rejects_invalid_window_mode(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="window"):
            WalkForwardSplit(
                n_splits=2,
                test_size=2,
                window="ratchet",  # type: ignore[arg-type]
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_rejects_n_splits_too_large_for_data(self) -> None:
        pred, evalu = _times(n=10)
        with pytest.raises(ValueError, match="too many"):
            cv = WalkForwardSplit(
                n_splits=20,
                test_size=2,
                prediction_times=pred,
                evaluation_times=evalu,
            )
            list(cv.split(np.zeros((10, 1))))

    def test_rejects_non_integer_split_sizes(self) -> None:
        pred, evalu = _times()
        with pytest.raises(TypeError, match="n_splits"):
            WalkForwardSplit(
                n_splits=2.5,  # type: ignore[arg-type]
                test_size=2,
                prediction_times=pred,
                evaluation_times=evalu,
            )
        with pytest.raises(TypeError, match="test_size"):
            WalkForwardSplit(
                n_splits=2,
                test_size=2.5,  # type: ignore[arg-type]
                prediction_times=pred,
                evaluation_times=evalu,
            )
        with pytest.raises(TypeError, match="train_size"):
            WalkForwardSplit(
                n_splits=2,
                test_size=2,
                window="sliding",
                train_size=2.5,  # type: ignore[arg-type]
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_rejects_train_size_with_expanding_window(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="train_size"):
            WalkForwardSplit(
                n_splits=2,
                test_size=2,
                window="expanding",
                train_size=5,
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_rejects_when_total_test_equals_n_samples(self) -> None:
        """The semantically interesting boundary: n_splits * test_size == n_samples
        leaves zero pre-test rows and must be rejected."""
        pred, evalu = _times(n=10)
        cv = WalkForwardSplit(
            n_splits=5,
            test_size=2,
            prediction_times=pred,
            evaluation_times=evalu,
        )
        with pytest.raises(ValueError, match="too many"):
            list(cv.split(np.zeros((10, 1))))

    def test_rejects_sliding_window_without_train_size(self) -> None:
        pred, evalu = _times()
        with pytest.raises(ValueError, match="train_size"):
            WalkForwardSplit(
                n_splits=2,
                test_size=2,
                window="sliding",
                train_size=None,
                prediction_times=pred,
                evaluation_times=evalu,
            )

    def test_rejects_non_positive_sliding_train_size(self) -> None:
        pred, evalu = _times()
        for train_size in (0, -1):
            with pytest.raises(ValueError, match="train_size"):
                WalkForwardSplit(
                    n_splits=2,
                    test_size=2,
                    window="sliding",
                    train_size=train_size,
                    prediction_times=pred,
                    evaluation_times=evalu,
                )
