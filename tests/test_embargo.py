"""Unit tests for purgedcv._embargo (Domain D3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purgedcv._embargo import apply_embargo


def _make_horizon_dataset(horizon_days: int = 1, n: int = 20) -> tuple[pd.Series, pd.Series]:
    pred = pd.Series(pd.date_range("2024-01-01", periods=n, freq="D"))
    evalu = pred + pd.Timedelta(days=horizon_days)
    return pred, evalu


class TestApplyEmbargo:
    def test_zero_embargo_is_identity(self) -> None:
        pred, evalu = _make_horizon_dataset()
        train_idx = np.arange(20)
        test_idx = np.arange(5, 10)
        result = apply_embargo(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(0))
        np.testing.assert_array_equal(result, train_idx)

    def test_rejects_negative_embargo(self) -> None:
        pred, evalu = _make_horizon_dataset()
        with pytest.raises(ValueError, match="non-negative"):
            apply_embargo(
                np.arange(20),
                np.arange(5, 10),
                pred,
                evalu,
                embargo=pd.Timedelta(days=-1),
            )

    def test_rejects_missing_embargo(self) -> None:
        pred, evalu = _make_horizon_dataset()
        with pytest.raises(ValueError, match="non-missing"):
            apply_embargo(
                np.arange(20),
                np.arange(5, 10),
                pred,
                evalu,
                embargo=pd.NaT,  # type: ignore[arg-type]
            )

    def test_drops_first_post_test_sample(self) -> None:
        """Closed window [eval_max, eval_max+embargo]. Test ends Jan 11
        (evalu[9]=Jan 11). embargo=1D -> cutoff Jan 12. Train row 11 has
        pred=Jan 12 -> in [Jan 11, Jan 12] -> dropped. Rows 12..14 (pred
        Jan 13..Jan 15) are past cutoff -> kept."""
        pred, evalu = _make_horizon_dataset()
        train_idx = np.arange(11, 15)
        test_idx = np.arange(5, 10)
        result = apply_embargo(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=1))
        np.testing.assert_array_equal(result, np.array([12, 13, 14]))

    def test_oversized_embargo_drops_all_post_test(self) -> None:
        pred, evalu = _make_horizon_dataset()
        train_idx = np.arange(11, 20)
        test_idx = np.arange(5, 10)
        result = apply_embargo(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=1000))
        assert result.size == 0

    def test_pre_test_train_never_dropped(self) -> None:
        """Embargo is asymmetric: rows with pred < eval_max are kept."""
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([0, 1, 2])
        test_idx = np.arange(10, 15)
        result = apply_embargo(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=1000))
        np.testing.assert_array_equal(result, train_idx)

    def test_preserves_dtype_and_order(self) -> None:
        pred, evalu = _make_horizon_dataset()
        train_idx = np.array([19, 18, 0], dtype=np.int64)
        test_idx = np.arange(5, 10)
        result = apply_embargo(train_idx, test_idx, pred, evalu, embargo=pd.Timedelta(days=2))
        # All three rows survive (rows 18,19 are past cutoff Jan 13; row 0 is pre-test).
        np.testing.assert_array_equal(result, train_idx)
        assert result.dtype == np.int64

    def test_empty_train(self) -> None:
        pred, evalu = _make_horizon_dataset()
        result = apply_embargo(
            np.array([], dtype=int),
            np.arange(5, 10),
            pred,
            evalu,
            embargo=pd.Timedelta(days=1),
        )
        assert result.size == 0

    def test_empty_test_returns_input(self) -> None:
        pred, evalu = _make_horizon_dataset()
        train = np.arange(0, 10)
        result = apply_embargo(
            train, np.array([], dtype=int), pred, evalu, embargo=pd.Timedelta(days=1)
        )
        np.testing.assert_array_equal(result, train)

    def test_embargoed_train_passes_diagnostic(self) -> None:
        """Output of apply_embargo must satisfy assert_embargo_respected."""
        from purgedcv.diagnostics import assert_embargo_respected

        pred, evalu = _make_horizon_dataset()
        train_idx = np.arange(0, 20)
        test_idx = np.arange(5, 10)
        emb = pd.Timedelta(days=3)
        embargoed = apply_embargo(train_idx, test_idx, pred, evalu, embargo=emb)
        assert_embargo_respected(embargoed, test_idx, pred, evalu, embargo=emb)

    def test_disjoint_test_blocks_embargo_each_local_post_test_window(self) -> None:
        """Embargo must apply after every non-contiguous test block, not
        only after the final test row."""
        pred, evalu = _make_horizon_dataset(horizon_days=1, n=12)
        train_idx = np.array([3, 4, 5, 6, 7, 8])
        test_idx = np.array([0, 1, 2, 9, 10, 11])

        result = apply_embargo(
            train_idx,
            test_idx,
            pred,
            evalu,
            embargo=pd.Timedelta(days=1),
        )

        # Test rows 0..2 embargo prediction times through Jan 5, so rows
        # 3 and 4 are removed even though they are far before the final
        # test block.
        expected = np.array([5, 6, 7, 8])
        np.testing.assert_array_equal(result, expected)
