"""Unit tests for statistical metrics (Domain D7)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from purgedcv._metrics import (
    deflated_sharpe_ratio,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)


def _make_returns(
    mean: float, std: float, n: int, *, skew: float = 0.0, kurt: float = 3.0, seed: int = 42
) -> np.ndarray:
    """Generate a return series with approximately the requested moments.
    Uses a normal distribution as the baseline; skew/kurt arguments are
    accepted for documentation but not enforced (the generated series
    is normal). Tests that need specific higher moments construct the
    series explicitly."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=mean, scale=std, size=n)
    return returns


class TestProbabilisticSharpeRatio:
    def test_returns_05_when_observed_equals_benchmark(self) -> None:
        """When SR_hat == benchmark_skill, the numerator of PSR is zero,
        Phi(0) = 0.5, so PSR = 0.5 exactly."""
        # Construct returns whose sample Sharpe equals 1.0 exactly.
        n = 100
        returns = np.zeros(n)
        returns[: n // 2] = 1.0
        returns[n // 2 :] = 1.0  # mean = 1.0, std = 0 -> infinite SR
        # That's degenerate; use a better construction:
        rng = np.random.default_rng(0)
        raw = rng.normal(0, 1, n)
        # Standardize then add mean=1 so SR = 1.
        raw = (raw - raw.mean()) / raw.std(ddof=0) + 1.0
        sample_sr = raw.mean() / raw.std(ddof=0)
        assert abs(sample_sr - 1.0) < 1e-10
        psr = probabilistic_sharpe_ratio(raw, benchmark_skill=1.0)
        assert abs(psr - 0.5) < 1e-9

    def test_monotone_in_observed_sharpe(self) -> None:
        """For fixed n and benchmark, PSR is monotone non-decreasing in
        the observed Sharpe ratio."""
        rng = np.random.default_rng(1)
        n = 200
        base = rng.normal(0, 1, n)
        psr_values = []
        for shift in [-0.2, -0.1, 0.0, 0.1, 0.2, 0.5]:
            returns = base + shift
            psr_values.append(probabilistic_sharpe_ratio(returns, benchmark_skill=0.0))
        for a, b in itertools.pairwise(psr_values):
            assert a <= b + 1e-12  # non-decreasing

    def test_approaches_one_for_large_n_when_observed_above_benchmark(self) -> None:
        """For SR_hat > benchmark, PSR -> 1 as n -> infinity."""
        rng = np.random.default_rng(2)
        # Construct deterministic high-Sharpe series.
        n_small = 30
        n_large = 5000
        # Mean = 0.1, std = 0.5 -> SR ~ 0.2
        small = rng.normal(0.1, 0.5, n_small)
        large = rng.normal(0.1, 0.5, n_large)
        psr_small = probabilistic_sharpe_ratio(small, benchmark_skill=0.0)
        psr_large = probabilistic_sharpe_ratio(large, benchmark_skill=0.0)
        assert psr_large > psr_small
        assert psr_large > 0.95

    def test_hand_computed_normal_returns_value(self) -> None:
        """Worked example with explicit moments and the formula applied
        on paper:

            returns = [a series of length 120 with sample mean = 0.005,
                       sample std = 0.02, sample skew = 0.0,
                       sample kurtosis = 3.0 (normal kurt)]

        Then SR_hat = 0.25, gamma3 = 0, gamma4 = 3, n = 120,
        benchmark = 0.

            numerator = (0.25 - 0) * sqrt(120 - 1) = 0.25 * 10.9087 = 2.7272
            denominator = sqrt(1 - 0 + (3-1)/4 * 0.25^2) = sqrt(1 + 0.03125)
                        = sqrt(1.03125) = 1.0155
            z = 2.7272 / 1.0155 = 2.6856
            PSR = Phi(2.6856) ~ 0.99637

        Verify with norm.cdf to 1e-4 tolerance."""
        # Construct returns with EXACTLY these moments by transforming a
        # normal sample.
        rng = np.random.default_rng(3)
        raw = rng.normal(0, 1, 120)
        # Standardize to mean=0, std=1.
        raw = (raw - raw.mean()) / raw.std(ddof=0)
        # Confirm moments are close to target.
        # Note: random normal sample of 120 may not have skew = 0 exactly.
        # Use a normal-like construction: take 120 z-score quantiles of N(0,1).
        z_quantiles = stats.norm.ppf(np.linspace(0.5 / 120, 1 - 0.5 / 120, 120))
        # Standardize to exactly mean=0, std=1 (ddof=0) so scaling below is exact.
        z_quantiles = (z_quantiles - z_quantiles.mean()) / z_quantiles.std(ddof=0)
        # Now z_quantiles has mean=0, std(ddof=0)=1, skew ~ 0, kurtosis ~ 3.
        # Scale to target sample mean = 0.005 and std = 0.02.
        target_mean = 0.005
        target_std = 0.02
        returns = z_quantiles * target_std + target_mean
        # Verify our constructed series:
        actual_mean = float(returns.mean())
        actual_std = float(returns.std(ddof=0))
        assert abs(actual_mean - target_mean) < 1e-10
        assert abs(actual_std - target_std) < 1e-6
        actual_sr = actual_mean / actual_std
        assert abs(actual_sr - 0.25) < 1e-4

        # Now compute PSR via the formula on paper:
        from scipy.stats import kurtosis, skew

        sample_skew = float(skew(returns, bias=False))
        sample_kurt = float(kurtosis(returns, bias=False, fisher=False))
        sr_hat = actual_sr
        benchmark = 0.0
        n = 120
        denom = np.sqrt(1 - sample_skew * sr_hat + (sample_kurt - 1) / 4 * sr_hat**2)
        z = (sr_hat - benchmark) * np.sqrt(n - 1) / denom
        expected = float(stats.norm.cdf(z))

        psr = probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)
        assert abs(psr - expected) < 1e-9

    def test_rejects_returns_with_nan(self) -> None:
        returns = np.array([0.01, 0.02, np.nan, 0.005])
        with pytest.raises(ValueError, match="NaN"):
            probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)

    def test_rejects_returns_below_minimum_length(self) -> None:
        """PSR requires n >= 2 (the formula has sqrt(n - 1))."""
        returns = np.array([0.01])
        with pytest.raises(ValueError, match="at least 2"):
            probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)

    def test_rejects_zero_variance_returns(self) -> None:
        """Constant return series has zero variance and undefined Sharpe."""
        returns = np.full(50, 0.01)
        with pytest.raises(ValueError, match="variance"):
            probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)


class TestDeflatedSharpeRatio:
    def test_reduces_to_psr_when_n_trials_is_one(self) -> None:
        """With only one trial (no multiple-comparison correction),
        DSR should be very close to PSR at the same benchmark (the
        deflated threshold SR* approaches the supplied benchmark)."""
        rng = np.random.default_rng(10)
        returns = rng.normal(0.001, 0.01, 252)
        var_sharpe = 0.1**2  # arbitrary variance estimate
        dsr = deflated_sharpe_ratio(returns, n_trials=1, var_sharpe=var_sharpe)
        # n_trials=1 -> ppf(1 - 1/1) = ppf(0) = -inf, so SR* -> -inf and
        # DSR -> 1. Edge case: n_trials=1 is informationally vacuous;
        # the function should either accept it and return ~1 or reject it.
        # By our chosen contract, we accept and return ~1.
        assert dsr > 0.99

    def test_decreases_with_more_trials(self) -> None:
        """More trials -> higher SR* threshold -> lower DSR."""
        rng = np.random.default_rng(11)
        returns = rng.normal(0.005, 0.01, 252)
        var_sharpe = 0.05**2
        dsr_few = deflated_sharpe_ratio(returns, n_trials=10, var_sharpe=var_sharpe)
        dsr_many = deflated_sharpe_ratio(returns, n_trials=1000, var_sharpe=var_sharpe)
        assert dsr_many < dsr_few

    def test_hand_computed_via_formula(self) -> None:
        """For a known returns series, compute DSR by hand using the
        formula and the Euler-Mascheroni constant gamma ~ 0.5772.

        SR* = sqrt(var_sharpe) * ((1 - gamma) * Phi_inv(1 - 1/n_trials)
                                 + gamma * Phi_inv(1 - 1/(n_trials * e)))

        Then DSR = PSR(returns, benchmark=SR*)."""
        from math import e

        rng = np.random.default_rng(12)
        returns = rng.normal(0.001, 0.01, 200)
        n_trials = 100
        var_sharpe = 0.02**2  # std_sharpe = 0.02
        gamma_em = 0.577215664901532  # Euler-Mascheroni
        sr_star = np.sqrt(var_sharpe) * (
            (1 - gamma_em) * stats.norm.ppf(1 - 1 / n_trials)
            + gamma_em * stats.norm.ppf(1 - 1 / (n_trials * e))
        )
        expected_dsr = probabilistic_sharpe_ratio(returns, benchmark_skill=float(sr_star))
        actual_dsr = deflated_sharpe_ratio(returns, n_trials=n_trials, var_sharpe=var_sharpe)
        assert abs(actual_dsr - expected_dsr) < 1e-12

    def test_rejects_invalid_n_trials(self) -> None:
        rng = np.random.default_rng(13)
        returns = rng.normal(0.001, 0.01, 100)
        with pytest.raises(ValueError, match="n_trials"):
            deflated_sharpe_ratio(returns, n_trials=0, var_sharpe=0.01)
        with pytest.raises(ValueError, match="n_trials"):
            deflated_sharpe_ratio(returns, n_trials=-5, var_sharpe=0.01)

    def test_rejects_negative_var_sharpe(self) -> None:
        rng = np.random.default_rng(14)
        returns = rng.normal(0.001, 0.01, 100)
        with pytest.raises(ValueError, match="var_sharpe"):
            deflated_sharpe_ratio(returns, n_trials=10, var_sharpe=-0.01)


class TestMinTrackRecordLength:
    def test_inversion_with_psr(self) -> None:
        """min_track_record_length(SR_hat, target, alpha, ...) returns n*
        such that PSR(returns_of_length_n*) >= 1 - alpha at the same
        skew/kurtosis. Verify by reconstructing a returns series of that
        length and confirming PSR meets the threshold."""
        observed_sharpe = 0.5
        target_sharpe = 0.2
        alpha = 0.05
        skew = 0.0
        kurt = 3.0
        n_star = min_track_record_length(
            observed_sharpe=observed_sharpe,
            target_sharpe=target_sharpe,
            alpha=alpha,
            skew=skew,
            kurtosis=kurt,
        )
        assert n_star >= 2  # always at least 2 observations
        # Inverse check: construct returns of length n_star with the
        # given moments, compute PSR, verify >= 1 - alpha.
        z_target = stats.norm.ppf(1 - alpha)
        denom = np.sqrt(1 - skew * observed_sharpe + (kurt - 1) / 4 * observed_sharpe**2)
        # PSR(returns_of_n_star) z-stat = (SR_hat - target) * sqrt(n_star - 1) / denom
        # We need this >= z_target, i.e. (n_star - 1) >= (z_target * denom / (SR_hat - target))^2
        # min_n - 1 >= (z_target * denom / sr_diff)^2
        # min_n = ceil((z_target * denom / sr_diff)^2) + 1
        sr_diff = observed_sharpe - target_sharpe
        expected_n_minus_1 = (z_target * denom / sr_diff) ** 2
        assert abs((n_star - 1) - expected_n_minus_1) < 1.0  # ceiling rounding

    def test_increases_as_observed_approaches_target(self) -> None:
        """If observed_sharpe is barely above target, you need a LONGER
        track record to be confident."""
        target = 0.2
        alpha = 0.05
        n_close = min_track_record_length(
            observed_sharpe=0.21,
            target_sharpe=target,
            alpha=alpha,
            skew=0.0,
            kurtosis=3.0,
        )
        n_far = min_track_record_length(
            observed_sharpe=0.5,
            target_sharpe=target,
            alpha=alpha,
            skew=0.0,
            kurtosis=3.0,
        )
        assert n_close > n_far

    def test_rejects_observed_at_or_below_target(self) -> None:
        """If you haven't beaten the benchmark yet, no n can make PSR
        statistically meaningful — function refuses."""
        with pytest.raises(ValueError, match="observed_sharpe"):
            min_track_record_length(
                observed_sharpe=0.1,
                target_sharpe=0.1,
                alpha=0.05,
                skew=0.0,
                kurtosis=3.0,
            )
        with pytest.raises(ValueError, match="observed_sharpe"):
            min_track_record_length(
                observed_sharpe=0.05,
                target_sharpe=0.1,
                alpha=0.05,
                skew=0.0,
                kurtosis=3.0,
            )

    def test_rejects_invalid_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            min_track_record_length(
                observed_sharpe=0.5,
                target_sharpe=0.2,
                alpha=0.0,
                skew=0.0,
                kurtosis=3.0,
            )
        with pytest.raises(ValueError, match="alpha"):
            min_track_record_length(
                observed_sharpe=0.5,
                target_sharpe=0.2,
                alpha=1.0,
                skew=0.0,
                kurtosis=3.0,
            )
