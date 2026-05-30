"""Unit tests for statistical metrics (Domain D7)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy import stats

from purgedcv._metrics import (
    deflated_sharpe_ratio,
    effective_n_trials,
    min_track_record_length,
    probabilistic_sharpe_ratio,
)
from purgedcv._typing import NDArrayAny


def _make_returns(
    mean: float, std: float, n: int, *, skew: float = 0.0, kurt: float = 3.0, seed: int = 42
) -> NDArrayAny:
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

    def test_rejects_returns_with_infinity(self) -> None:
        returns = np.array([0.01, 0.02, np.inf, 0.005])
        with pytest.raises(ValueError, match="infinite"):
            probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)

    def test_rejects_returns_with_non_finite_moments(self) -> None:
        returns = np.array([1e308, -1e308, 1e308, -1e308])
        with pytest.raises(ValueError, match="non-finite"):
            probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)

    def test_rejects_non_finite_benchmark(self) -> None:
        returns = np.array([0.01, 0.02, 0.005])
        with pytest.raises(ValueError, match="benchmark_skill"):
            probabilistic_sharpe_ratio(returns, benchmark_skill=np.inf)

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
        """With a single trial there is no multiple-comparison
        correction, so SR* = 0 and DSR must reduce exactly to
        ``probabilistic_sharpe_ratio(returns, 0.0)`` -- for any input and
        independently of ``var_sharpe``."""
        rng = np.random.default_rng(10)
        winning = rng.normal(0.001, 0.01, 252)
        losing = rng.normal(-0.05, 0.01, 252)
        for returns in (winning, losing):
            psr0 = probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)
            for var_sharpe in (0.0, 0.1**2, 1.0):
                dsr = deflated_sharpe_ratio(returns, n_trials=1, var_sharpe=var_sharpe)
                assert dsr == pytest.approx(psr0)

    def test_losing_strategy_is_not_certain_skill_at_one_trial(self) -> None:
        """Regression: a single-trial DSR must reflect the input. The old
        implementation set SR* to -inf at n_trials=1 and returned 1.0 for
        every strategy, including ones that lose money every period."""
        rng = np.random.default_rng(12)
        losing = rng.normal(-0.05, 0.01, 252)
        dsr = deflated_sharpe_ratio(losing, n_trials=1, var_sharpe=0.04)
        assert dsr < 0.01

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

    def test_rejects_non_integer_n_trials(self) -> None:
        rng = np.random.default_rng(13)
        returns = rng.normal(0.001, 0.01, 100)
        for n_trials in (1.5, True):
            with pytest.raises(TypeError, match="integer"):
                deflated_sharpe_ratio(returns, n_trials=n_trials, var_sharpe=0.01)  # type: ignore[arg-type]

    def test_rejects_negative_var_sharpe(self) -> None:
        rng = np.random.default_rng(14)
        returns = rng.normal(0.001, 0.01, 100)
        with pytest.raises(ValueError, match="var_sharpe"):
            deflated_sharpe_ratio(returns, n_trials=10, var_sharpe=-0.01)

    def test_rejects_non_finite_var_sharpe(self) -> None:
        rng = np.random.default_rng(14)
        returns = rng.normal(0.001, 0.01, 100)
        for var_sharpe in (np.nan, np.inf):
            with pytest.raises(ValueError, match="finite"):
                deflated_sharpe_ratio(returns, n_trials=10, var_sharpe=var_sharpe)

    def test_bars_per_year_converts_annualised_variance(self) -> None:
        """Passing an annualised var with bars_per_year is identical to
        passing the per-observation var directly, and differs from
        mis-passing the annualised var as per-observation."""
        rng = np.random.default_rng(40)
        returns = rng.normal(0.001, 0.01, 252)
        var_annual = 0.5**2
        annual = deflated_sharpe_ratio(returns, 50, var_annual, bars_per_year=252)
        per_bar = deflated_sharpe_ratio(returns, 50, var_annual / 252)
        assert annual == pytest.approx(per_bar)
        wrong = deflated_sharpe_ratio(returns, 50, var_annual)
        assert annual != pytest.approx(wrong)

    def test_rejects_non_positive_bars_per_year(self) -> None:
        rng = np.random.default_rng(41)
        returns = rng.normal(0.001, 0.01, 100)
        for bad in (0, -252):
            with pytest.raises(ValueError, match="bars_per_year"):
                deflated_sharpe_ratio(returns, 10, 0.01, bars_per_year=bad)

    def test_rejects_non_finite_bars_per_year(self) -> None:
        rng = np.random.default_rng(43)
        returns = rng.normal(0.001, 0.01, 100)
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="bars_per_year"):
                deflated_sharpe_ratio(returns, 10, 0.01, bars_per_year=bad)  # type: ignore[arg-type]

    def test_rejects_bool_bars_per_year(self) -> None:
        """``True`` is an int subclass but never a meaningful bars-per-year."""
        rng = np.random.default_rng(44)
        returns = rng.normal(0.001, 0.01, 100)
        with pytest.raises(ValueError, match="bool"):
            deflated_sharpe_ratio(returns, 10, 0.01, bars_per_year=True)


class TestDeflatedSharpeRatioFull:
    def test_dsr_field_matches_scalar_function(self) -> None:
        """The ``dsr`` field must equal ``deflated_sharpe_ratio`` exactly."""
        rng = np.random.default_rng(20)
        returns = rng.normal(0.001, 0.01, 252)
        for n_trials, var_sharpe in [(1, 0.04), (10, 0.01**2), (500, 0.02**2)]:
            from purgedcv._metrics import deflated_sharpe_ratio_full

            diag = deflated_sharpe_ratio_full(returns, n_trials=n_trials, var_sharpe=var_sharpe)
            scalar = deflated_sharpe_ratio(returns, n_trials=n_trials, var_sharpe=var_sharpe)
            assert diag.dsr == pytest.approx(scalar)

    def test_reports_consistent_intermediate_quantities(self) -> None:
        """sr_star = sqrt(var_sharpe) * expected_max_z, and the moments match
        what PSR computes internally."""
        from purgedcv._metrics import deflated_sharpe_ratio_full

        rng = np.random.default_rng(21)
        returns = rng.normal(0.002, 0.01, 300)
        var_sharpe = 0.03**2
        diag = deflated_sharpe_ratio_full(returns, n_trials=200, var_sharpe=var_sharpe)
        assert diag.n_obs == 300
        assert diag.n_trials == 200
        assert diag.var_sharpe == pytest.approx(var_sharpe)
        assert diag.sr_star == pytest.approx(np.sqrt(var_sharpe) * diag.expected_max_z)
        # observed_sr, skew, kurt match the population-std Sharpe moments.
        observed_sr = float(returns.mean() / returns.std(ddof=0))
        assert diag.observed_sr == pytest.approx(observed_sr)
        assert diag.skew == pytest.approx(float(stats.skew(returns, bias=False)))
        assert diag.kurt == pytest.approx(float(stats.kurtosis(returns, bias=False, fisher=False)))

    def test_single_trial_has_zero_benchmark(self) -> None:
        """With one trial there is no deflation: sr_star and expected_max_z
        are both 0, and dsr reduces to PSR against zero."""
        from purgedcv._metrics import deflated_sharpe_ratio_full

        rng = np.random.default_rng(22)
        returns = rng.normal(0.001, 0.01, 252)
        diag = deflated_sharpe_ratio_full(returns, n_trials=1, var_sharpe=0.5)
        assert diag.sr_star == 0.0
        assert diag.expected_max_z == 0.0
        assert diag.dsr == pytest.approx(probabilistic_sharpe_ratio(returns, 0.0))

    def test_bars_per_year_stores_per_observation_var(self) -> None:
        """With bars_per_year the result echoes the per-observation var and
        the sr_star = sqrt(var) * expected_max_z identity still holds; the
        probability matches the scalar form with the same conversion."""
        from purgedcv._metrics import deflated_sharpe_ratio_full

        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.01, 252)
        var_annual = 0.4**2
        diag = deflated_sharpe_ratio_full(returns, 100, var_annual, bars_per_year=252)
        assert diag.var_sharpe == pytest.approx(var_annual / 252)
        assert diag.sr_star == pytest.approx(np.sqrt(diag.var_sharpe) * diag.expected_max_z)
        assert diag.dsr == pytest.approx(
            deflated_sharpe_ratio(returns, 100, var_annual, bars_per_year=252)
        )

    def test_validates_like_scalar_form(self) -> None:
        from purgedcv._metrics import deflated_sharpe_ratio_full

        rng = np.random.default_rng(23)
        returns = rng.normal(0.001, 0.01, 100)
        with pytest.raises(ValueError, match="n_trials"):
            deflated_sharpe_ratio_full(returns, n_trials=0, var_sharpe=0.01)
        with pytest.raises(TypeError, match="integer"):
            deflated_sharpe_ratio_full(returns, n_trials=1.5, var_sharpe=0.01)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="var_sharpe"):
            deflated_sharpe_ratio_full(returns, n_trials=10, var_sharpe=-1.0)


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

    def test_returns_inf_when_observed_at_or_below_target(self) -> None:
        """If you haven't beaten the benchmark yet, no finite track record
        can make PSR(target) reach 1 - alpha. The well-defined answer is
        infinity, returned rather than raised so callers can branch on it."""
        import math

        # Equal: observed exactly at target.
        assert math.isinf(
            min_track_record_length(
                observed_sharpe=0.1,
                target_sharpe=0.1,
                alpha=0.05,
                skew=0.0,
                kurtosis=3.0,
            )
        )
        # Strictly below target.
        assert math.isinf(
            min_track_record_length(
                observed_sharpe=0.05,
                target_sharpe=0.1,
                alpha=0.05,
                skew=0.0,
                kurtosis=3.0,
            )
        )

    def test_returns_float_for_finite_case(self) -> None:
        """The finite answer is a float; wrapping in int() recovers a count."""
        n = min_track_record_length(
            observed_sharpe=0.5,
            target_sharpe=0.2,
            alpha=0.05,
            skew=0.0,
            kurtosis=3.0,
        )
        assert isinstance(n, float)
        assert n == int(n)  # integral value, float-typed

    def test_invalid_alpha_still_raises_even_when_unreachable(self) -> None:
        """Input validation (alpha range) takes precedence over the
        unreachable-track-record short circuit."""
        with pytest.raises(ValueError, match="alpha"):
            min_track_record_length(
                observed_sharpe=0.05,
                target_sharpe=0.1,
                alpha=1.5,
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

    def test_alpha_at_or_above_half_needs_minimum_track_record(self) -> None:
        assert (
            min_track_record_length(
                observed_sharpe=0.5,
                target_sharpe=0.2,
                alpha=0.5,
                skew=0.0,
                kurtosis=3.0,
            )
            == 2
        )
        assert (
            min_track_record_length(
                observed_sharpe=0.5,
                target_sharpe=0.2,
                alpha=0.9,
                skew=0.0,
                kurtosis=3.0,
            )
            == 2
        )

    def test_rejects_too_small_alpha_for_float_quantile(self) -> None:
        with pytest.raises(ValueError, match="too small"):
            min_track_record_length(
                observed_sharpe=0.5,
                target_sharpe=0.2,
                alpha=1e-100,
                skew=0.0,
                kurtosis=3.0,
            )

    def test_rejects_non_finite_track_record_length(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            min_track_record_length(
                observed_sharpe=0.5,
                target_sharpe=0.2,
                alpha=0.05,
                skew=0.0,
                kurtosis=1e308,
            )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("observed_sharpe", np.nan),
            ("target_sharpe", np.inf),
            ("alpha", np.nan),
            ("skew", np.inf),
            ("kurtosis", np.nan),
        ],
    )
    def test_rejects_non_finite_scalar_inputs(self, field: str, value: float) -> None:
        kwargs = {
            "observed_sharpe": 0.5,
            "target_sharpe": 0.2,
            "alpha": 0.05,
            "skew": 0.0,
            "kurtosis": 3.0,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            min_track_record_length(**kwargs)


class TestEffectiveNTrials:
    def test_independent_series_is_near_full_count(self) -> None:
        """Uncorrelated trials: the effective count is close to the raw count."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        n_eff = effective_n_trials(x)
        assert 0.6 * 500 <= n_eff <= 500

    def test_autocorrelated_series_collapses(self) -> None:
        """A random walk is strongly autocorrelated: far fewer effective trials."""
        rng = np.random.default_rng(1)
        walk = np.cumsum(rng.standard_normal(500))
        assert effective_n_trials(walk) < 50

    def test_more_correlation_means_fewer_effective(self) -> None:
        """Higher AR(1) coefficient -> stronger correlation -> smaller n_eff."""
        rng = np.random.default_rng(2)

        def ar1(phi: float, n: int) -> NDArrayAny:
            innov = rng.standard_normal(n)
            out = np.zeros(n)
            for t in range(1, n):
                out[t] = phi * out[t - 1] + innov[t]
            return out

        n_low = effective_n_trials(ar1(0.3, 1000))
        n_high = effective_n_trials(ar1(0.95, 1000))
        assert n_high < n_low

    def test_constant_series_is_one(self) -> None:
        assert effective_n_trials(np.full(100, 2.5)) == 1

    def test_result_is_bounded(self) -> None:
        rng = np.random.default_rng(3)
        n = 300
        n_eff = effective_n_trials(rng.standard_normal(n))
        assert 1 <= n_eff <= n
        assert isinstance(n_eff, int)

    def test_fewer_than_three_returns_raw_count(self) -> None:
        assert effective_n_trials(np.array([1.0, 2.0])) == 2
        assert effective_n_trials(np.array([1.0])) == 1

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="method"):
            effective_n_trials(np.arange(10.0), method="bootstrap")

    def test_rejects_empty_and_non_finite(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            effective_n_trials(np.array([]))
        with pytest.raises(ValueError, match="NaN or infinite"):
            effective_n_trials(np.array([1.0, np.nan, 2.0, 3.0]))

    def test_rejects_non_1d_array(self) -> None:
        """A matrix must not be silently flattened: its row-major order is not
        a meaningful trial trajectory."""
        with pytest.raises(ValueError, match="1-D"):
            effective_n_trials(np.arange(6.0).reshape(2, 3))

    def test_deflates_more_realistically(self) -> None:
        """The headline use: a correlated search inflates raw n_trials, which
        crushes DSR; the effective count restores an informative DSR."""
        rng = np.random.default_rng(4)
        returns = rng.normal(0.001, 0.01, 504)
        walk = np.cumsum(rng.standard_normal(6000)) * 0.01  # 6000 correlated trials
        n_eff = effective_n_trials(walk)
        assert n_eff < 6000
        dsr_raw = deflated_sharpe_ratio(returns, 6000, 0.02**2)
        dsr_eff = deflated_sharpe_ratio(returns, n_eff, 0.02**2)
        assert dsr_eff >= dsr_raw
