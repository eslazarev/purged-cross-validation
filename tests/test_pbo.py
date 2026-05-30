"""Unit tests for Probability of Backtest Overfitting (PBO / CSCV)."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from purgedcv import probability_of_backtest_overfitting
from purgedcv._pbo import sharpe
from purgedcv._typing import NDArrayAny


class TestSharpeMetric:
    def test_zero_std_scores_zero(self) -> None:
        assert sharpe(np.full(10, 0.01)) == 0.0

    def test_positive_drift_scores_positive(self) -> None:
        assert sharpe(np.array([0.01, 0.02, 0.015, 0.018])) > 0


class TestPBOStructure:
    def test_n_combos_is_central_binomial(self) -> None:
        rng = np.random.default_rng(0)
        returns = rng.standard_normal((6, 200))
        result = probability_of_backtest_overfitting(returns, n_splits=8)
        assert result["n_combos"] == comb(8, 4) == 70

    def test_result_shapes(self) -> None:
        rng = np.random.default_rng(1)
        returns = rng.standard_normal((5, 160))
        result = probability_of_backtest_overfitting(returns, n_splits=6)
        n_combos = comb(6, 3)
        assert result["logits"].shape == (n_combos,)
        assert result["is_oos_performance"].shape == (n_combos, 2)
        assert 0.0 <= result["pbo"] <= 1.0

    def test_pbo_is_fraction_of_negative_logits(self) -> None:
        rng = np.random.default_rng(2)
        returns = rng.standard_normal((7, 180))
        result = probability_of_backtest_overfitting(returns, n_splits=6)
        expected = float(np.mean(result["logits"] < 0))
        assert result["pbo"] == pytest.approx(expected)

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(3)
        returns = rng.standard_normal((6, 160))
        a = probability_of_backtest_overfitting(returns, n_splits=8)
        b = probability_of_backtest_overfitting(returns, n_splits=8)
        assert a["pbo"] == b["pbo"]
        assert np.array_equal(a["logits"], b["logits"])


class TestPBODiscriminates:
    def test_dominant_config_has_low_pbo(self) -> None:
        """One configuration with genuine positive drift is the in-sample
        best in every combination and stays best out of sample, so it never
        lands below the median: PBO collapses to 0."""
        rng = np.random.default_rng(10)
        noise = rng.standard_normal((7, 240)) * 0.01
        signal = 0.05 + rng.standard_normal((1, 240)) * 0.01
        returns = np.vstack([signal, noise])
        result = probability_of_backtest_overfitting(returns, n_splits=8)
        assert result["pbo"] == 0.0

    def test_pure_noise_pbo_exceeds_signal_pbo(self) -> None:
        """Pure-noise configurations overfit: the in-sample winner is chosen
        by luck and lands near the OOS median, giving a far higher PBO than a
        set containing a genuinely dominant configuration."""
        rng = np.random.default_rng(11)
        noise_only = rng.standard_normal((8, 240)) * 0.01
        pbo_noise = probability_of_backtest_overfitting(noise_only, n_splits=8)["pbo"]

        rng2 = np.random.default_rng(11)
        noise = rng2.standard_normal((7, 240)) * 0.01
        signal = 0.05 + rng2.standard_normal((1, 240)) * 0.01
        with_signal = np.vstack([signal, noise])
        pbo_signal = probability_of_backtest_overfitting(with_signal, n_splits=8)["pbo"]

        assert pbo_noise > pbo_signal
        assert 0.2 <= pbo_noise <= 0.8  # near the 0.5 no-skill expectation


class TestPBOPurgePath:
    def test_times_without_purge_match_no_times(self) -> None:
        """Passing instantaneous times (no purge/embargo) must give exactly
        the same estimate as omitting times: both leave IS = complement."""
        rng = np.random.default_rng(20)
        returns = rng.standard_normal((6, 192)) * 0.01
        no_times = probability_of_backtest_overfitting(returns, n_splits=8)

        pred = pd.Series(pd.date_range("2024-01-01", periods=192, freq="D"))
        with_times = probability_of_backtest_overfitting(
            returns, n_splits=8, prediction_times=pred, evaluation_times=pred
        )
        assert with_times["pbo"] == no_times["pbo"]
        assert np.allclose(with_times["logits"], no_times["logits"])

    def test_purge_runs_and_returns_valid_pbo(self) -> None:
        rng = np.random.default_rng(21)
        returns = rng.standard_normal((6, 192)) * 0.01
        pred = pd.Series(pd.date_range("2024-01-01", periods=192, freq="D"))
        evalu = pred + pd.Timedelta(days=2)
        result = probability_of_backtest_overfitting(
            returns,
            n_splits=8,
            prediction_times=pred,
            evaluation_times=evalu,
            purge_horizon="2D",
            embargo="1D",
        )
        assert 0.0 <= result["pbo"] <= 1.0
        assert result["n_combos"] == comb(8, 4)

    def test_custom_metric_is_used(self) -> None:
        rng = np.random.default_rng(22)
        returns = rng.standard_normal((5, 160)) * 0.01
        result = probability_of_backtest_overfitting(
            returns, n_splits=6, metric=lambda r: float(np.mean(r))
        )
        assert 0.0 <= result["pbo"] <= 1.0


class TestPBOValidation:
    def _returns(self) -> NDArrayAny:
        rng = np.random.default_rng(99)
        return rng.standard_normal((5, 160))

    def test_rejects_odd_n_splits(self) -> None:
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(self._returns(), n_splits=7)

    def test_rejects_n_splits_below_two(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            probability_of_backtest_overfitting(self._returns(), n_splits=1)

    def test_rejects_non_2d_returns(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            probability_of_backtest_overfitting(np.zeros(100), n_splits=4)

    def test_rejects_single_config(self) -> None:
        with pytest.raises(ValueError, match="at least 2 configurations"):
            probability_of_backtest_overfitting(np.zeros((1, 100)), n_splits=4)

    def test_rejects_too_few_observations(self) -> None:
        with pytest.raises(ValueError, match="non-empty blocks"):
            probability_of_backtest_overfitting(np.zeros((3, 6)), n_splits=8)

    def test_rejects_non_finite_returns(self) -> None:
        returns = self._returns()
        returns[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN or infinite"):
            probability_of_backtest_overfitting(returns, n_splits=4)

    def test_rejects_purge_without_times(self) -> None:
        with pytest.raises(ValueError, match="prediction_times and evaluation_times"):
            probability_of_backtest_overfitting(self._returns(), n_splits=4, purge_horizon="1D")

    def test_rejects_time_length_mismatch(self) -> None:
        returns = self._returns()  # (5, 160)
        pred = pd.Series(pd.date_range("2024-01-01", periods=159, freq="D"))
        with pytest.raises(ValueError, match="prediction_times length"):
            probability_of_backtest_overfitting(
                returns, n_splits=4, prediction_times=pred, evaluation_times=pred
            )

    def test_rejects_non_finite_metric_result(self) -> None:
        """A custom metric that returns a non-finite score must raise rather
        than silently report pbo=0 with NaN logits."""
        with pytest.raises(ValueError, match="non-finite value"):
            probability_of_backtest_overfitting(
                self._returns(), n_splits=4, metric=lambda r: float("nan")
            )
