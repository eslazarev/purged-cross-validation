"""Internal: statistical metrics (Domain D7).

Four closed-form tools that correct reported Sharpe ratios:

- :func:`probabilistic_sharpe_ratio` (PSR) — Bailey & Lopez de Prado (2012).
- :func:`deflated_sharpe_ratio` (DSR) — Bailey & Lopez de Prado (2014).
- :func:`deflated_sharpe_ratio_full` — DSR plus the intermediate quantities
  that explain *why* the deflation landed where it did.
- :func:`min_track_record_length` (MinTRL) — derived from PSR by inversion.

References:
- Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. Journal of Risk 15(2).
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  Journal of Portfolio Management 40(5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from ._typing import NDArrayAny
from ._validation import _validate_bars_per_year

# Euler-Mascheroni constant, used by the DSR extreme-value approximation.
_GAMMA_EM = 0.5772156649015329


def _validate_returns(returns: NDArrayAny) -> NDArrayAny:
    """Coerce input to 1-D float ndarray, reject non-finite values and length < 2."""
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"returns must have at least 2 observations, got {arr.size}.")
    if not np.isfinite(arr).all():
        raise ValueError("returns contains NaN or infinite values; filter with np.isfinite first.")
    return arr


def _sharpe_moments(arr: NDArrayAny) -> tuple[float, float, float]:
    """Return ``(sr_hat, gamma3, gamma4)`` for a validated 1-D return array.

    ``sr_hat`` is the sample Sharpe ratio (population standard deviation,
    ``ddof=0``), ``gamma3`` the sample skew, and ``gamma4`` the sample
    kurtosis (NOT excess). Shared by PSR and the DSR diagnostics so both
    report the same moments.

    Raises:
        ValueError: on non-finite mean/std, zero variance, or non-finite
            higher moments.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("returns produce non-finite mean or standard deviation.")
    if std == 0.0:
        raise ValueError("returns has zero variance; Sharpe ratio is undefined.")
    sr_hat = mean / std
    gamma3 = float(stats.skew(arr, bias=False))
    gamma4 = float(stats.kurtosis(arr, bias=False, fisher=False))  # NOT excess
    if not np.isfinite(sr_hat) or not np.isfinite(gamma3) or not np.isfinite(gamma4):
        raise ValueError("returns produce non-finite Sharpe-ratio moments.")
    return sr_hat, gamma3, gamma4


def probabilistic_sharpe_ratio(
    returns: NDArrayAny,
    benchmark_skill: float,
) -> float:
    """Probability that the true Sharpe ratio exceeds ``benchmark_skill``.

    Formula (Bailey & Lopez de Prado 2012, Eq. 7):

    .. math::
        \\text{PSR}(\\text{SR}^\\ast) = \\Phi\\!\\left(
            \\frac{(\\widehat{\\text{SR}} - \\text{SR}^\\ast)\\sqrt{n - 1}}
            {\\sqrt{1 - \\widehat{\\gamma}_3\\,\\widehat{\\text{SR}}
                    + \\frac{\\widehat{\\gamma}_4 - 1}{4}\\,\\widehat{\\text{SR}}^{\\,2}}}
        \\right)

    where :math:`\\widehat{\\gamma}_3` is sample skew, :math:`\\widehat{\\gamma}_4`
    is sample kurtosis (NOT excess kurtosis), and :math:`\\Phi` is the
    standard normal CDF.

    Args:
        returns: 1-D array of returns, length >= 2, finite values, non-zero
            variance.
        benchmark_skill: The Sharpe-ratio threshold to test against. Use 0
            for "is this strategy better than holding cash."

    Returns:
        Scalar probability in [0, 1].

    Raises:
        ValueError: on length < 2, non-finite values, or zero variance.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import probabilistic_sharpe_ratio
        >>> rng = np.random.default_rng(0)
        >>> returns = rng.normal(0.001, 0.01, 252)
        >>> psr = probabilistic_sharpe_ratio(returns, benchmark_skill=0.0)
        >>> 0.0 <= psr <= 1.0
        True
    """
    if not np.isfinite(benchmark_skill):
        raise ValueError(f"benchmark_skill must be finite, got {benchmark_skill}.")
    arr = _validate_returns(returns)
    n = arr.size
    sr_hat, gamma3, gamma4 = _sharpe_moments(arr)
    denominator_sq = 1 - gamma3 * sr_hat + (gamma4 - 1) / 4 * sr_hat**2
    if not np.isfinite(denominator_sq) or denominator_sq <= 0:
        raise ValueError(
            f"PSR denominator is non-positive ({denominator_sq:.4g}); the "
            "input distribution's higher moments are too extreme for the "
            "Gaussian approximation."
        )
    z = (sr_hat - benchmark_skill) * np.sqrt(n - 1) / np.sqrt(denominator_sq)
    return float(stats.norm.cdf(z))


def _validate_dsr_inputs(n_trials: int, var_sharpe: float) -> None:
    """Shared validation for ``deflated_sharpe_ratio`` and its ``_full`` form."""
    if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)):
        raise TypeError(f"n_trials must be an integer, got {type(n_trials).__name__}.")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if not np.isfinite(var_sharpe):
        raise ValueError(f"var_sharpe must be finite, got {var_sharpe}.")
    if var_sharpe < 0:
        raise ValueError(f"var_sharpe must be non-negative, got {var_sharpe}.")


def _to_per_observation_var(var_sharpe: float, bars_per_year: int | None) -> float:
    """Convert an annualised Sharpe variance to per-observation, or pass through.

    DSR is intrinsically per-observation (PSR depends on the per-observation
    Sharpe non-linearly), so the only correct unit for ``var_sharpe`` is the
    variance of per-observation Sharpes. When the caller has annualised trial
    Sharpes, ``bars_per_year`` performs the exact conversion
    ``var_per_obs = var_annual / bars_per_year`` (since annualised Sharpe is
    per-observation Sharpe times ``sqrt(bars_per_year)``).
    """
    _validate_bars_per_year(bars_per_year)
    if bars_per_year is None:
        return var_sharpe
    return var_sharpe / bars_per_year


def _deflated_benchmark(n_trials: int, var_sharpe: float) -> tuple[float, float]:
    """Return ``(expected_max_z, sr_star)`` for the DSR deflation.

    ``expected_max_z`` is the standardized expected maximum of ``n_trials``
    independent Sharpe estimators under the null (the bracket term of the
    extreme-value approximation). ``sr_star`` is that multiplier scaled by
    the spread of trial Sharpes, i.e. the deflated benchmark in Sharpe units:
    ``sr_star = sqrt(var_sharpe) * expected_max_z``.

    With a single trial there is no multiple-comparison correction: the
    expected maximum Sharpe under the null across one trial is 0, so both
    quantities are 0 and DSR reduces to PSR against a zero benchmark. The
    extreme-value formula is valid only for ``n_trials >= 2``, where
    ``Phi_inv(1 - 1/n_trials)`` is finite (it diverges to -inf at n = 1).
    """
    from math import e

    if n_trials == 1:
        return 0.0, 0.0
    expected_max_z = float(
        (1 - _GAMMA_EM) * stats.norm.ppf(1 - 1 / n_trials)
        + _GAMMA_EM * stats.norm.ppf(1 - 1 / (n_trials * e))
    )
    sr_star = float(np.sqrt(var_sharpe) * expected_max_z)
    return expected_max_z, sr_star


def deflated_sharpe_ratio(
    returns: NDArrayAny,
    n_trials: int,
    var_sharpe: float,
    *,
    bars_per_year: int | None = None,
) -> float:
    """Probability that the true Sharpe ratio exceeds the deflated
    benchmark that accounts for ``n_trials`` independent hyperparameter
    searches under the null.

    Formula (Bailey & Lopez de Prado 2014):

    .. math::
        \\text{SR}^\\ast_{n} = \\sqrt{V[\\text{SR}]} \\left[
            (1 - \\gamma) \\Phi^{-1}\\!\\left(1 - \\frac{1}{n_{\\text{trials}}}\\right)
            + \\gamma \\Phi^{-1}\\!\\left(1 - \\frac{1}{n_{\\text{trials}} \\cdot e}\\right)
        \\right]

    where :math:`\\gamma \\approx 0.5772` is the Euler-Mascheroni constant.
    DSR is then :func:`probabilistic_sharpe_ratio` evaluated at the
    deflated benchmark :math:`\\text{SR}^\\ast_n`.

    Args:
        returns: 1-D array of returns (passed through to PSR).
        n_trials: Number of independent hyperparameter searches the user
            ran before reporting this strategy's Sharpe. Must be >= 1.
            With ``n_trials == 1`` there is no correction to apply and
            DSR reduces to ``probabilistic_sharpe_ratio(returns, 0.0)``.
        var_sharpe: Estimated variance of Sharpe ratios across the
            ``n_trials`` candidates. Caller supplies; we do not estimate
            it because that would require knowing the distribution of
            submitted strategies, which is private to the caller.
            :class:`purgedcv.optuna_integration.TrialSharpeRecorder`
            produces it directly from an Optuna study.

            UNITS: ``var_sharpe`` must be in the same Sharpe units as the
            per-observation Sharpe of ``returns`` (DSR is intrinsically
            per-observation). If your trial Sharpes were annualised, pass
            ``bars_per_year`` and the conversion is done for you; otherwise
            ``var_sharpe`` is taken as already per-observation. Note that
            :func:`path_metrics` annualises its Sharpe when given
            ``bars_per_year``, so a ``var`` taken from its output is
            annualised: pass the same ``bars_per_year`` here.
        bars_per_year: If given, ``var_sharpe`` is interpreted as an
            annualised Sharpe variance and converted to per-observation
            (``var_sharpe / bars_per_year``) before deflation. ``None``
            (default) treats ``var_sharpe`` as already per-observation and
            leaves prior behaviour unchanged.

    Returns:
        Scalar probability in [0, 1].

    Raises:
        TypeError: if ``n_trials`` is not an integer.
        ValueError: on invalid ``n_trials``, non-finite/negative
            ``var_sharpe``, or non-positive ``bars_per_year``.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import deflated_sharpe_ratio
        >>> rng = np.random.default_rng(0)
        >>> returns = rng.normal(0.001, 0.01, 252)
        >>> dsr = deflated_sharpe_ratio(returns, n_trials=50, var_sharpe=0.01**2)
        >>> 0.0 <= dsr <= 1.0
        True
        >>> # Annualised trial-Sharpe variance: pass bars_per_year to convert.
        >>> annual = deflated_sharpe_ratio(returns, 50, 0.5**2, bars_per_year=252)
        >>> per_bar = deflated_sharpe_ratio(returns, 50, 0.5**2 / 252)
        >>> abs(annual - per_bar) < 1e-12
        True
    """
    _validate_dsr_inputs(n_trials, var_sharpe)
    var_sharpe = _to_per_observation_var(var_sharpe, bars_per_year)
    _, sr_star = _deflated_benchmark(n_trials, var_sharpe)
    return probabilistic_sharpe_ratio(returns, benchmark_skill=sr_star)


@dataclass(frozen=True)
class DSRDiagnostics:
    """Return type of :func:`deflated_sharpe_ratio_full`.

    A frozen dataclass: read fields by attribute (``diag.dsr``), and call
    :func:`dataclasses.asdict` if you need a plain dict to serialise.

    Attributes:
        dsr: The deflated Sharpe probability (identical to
            :func:`deflated_sharpe_ratio` for the same inputs).
        observed_sr: Sample Sharpe ratio of ``returns`` (population
            standard deviation, ``ddof=0``). In the same per-period units
            as ``var_sharpe`` must be; see :func:`deflated_sharpe_ratio_full`.
        sr_star: Deflated benchmark in Sharpe units, i.e. the expected
            maximum Sharpe of ``n_trials`` candidates under the null.
        expected_max_z: Standardized expected maximum (the bracket term);
            ``sr_star = sqrt(var_sharpe) * expected_max_z``.
        var_sharpe: The per-observation Sharpe variance used in the
            deflation (after any ``bars_per_year`` conversion), so that
            ``sr_star == sqrt(var_sharpe) * expected_max_z`` holds.
        n_trials: The ``n_trials`` passed in.
        n_obs: Track record length (number of observations in ``returns``).
        skew: Sample skew of ``returns``.
        kurt: Sample kurtosis of ``returns`` (NOT excess).
    """

    dsr: float
    observed_sr: float
    sr_star: float
    expected_max_z: float
    var_sharpe: float
    n_trials: int
    n_obs: int
    skew: float
    kurt: float


def deflated_sharpe_ratio_full(
    returns: NDArrayAny,
    n_trials: int,
    var_sharpe: float,
    *,
    bars_per_year: int | None = None,
) -> DSRDiagnostics:
    """Like :func:`deflated_sharpe_ratio` but return the intermediate
    quantities alongside the probability.

    When DSR is near 0 the scalar form does not tell you *why*: was the
    deflated benchmark high because ``var_sharpe`` was large, because
    ``n_trials`` was large, or was the observed Sharpe simply low? This
    function returns all of those so the deflation can be inspected.

    Args:
        returns: 1-D array of returns.
        n_trials: Number of independent searches (>= 1).
        var_sharpe: Variance of Sharpe ratios across the candidates. Per
            observation by default; pass ``bars_per_year`` if it is
            annualised (see :func:`deflated_sharpe_ratio` for the unit
            contract).
        bars_per_year: If given, ``var_sharpe`` is annualised and converted
            to per-observation before deflation. ``None`` (default) treats
            it as already per-observation.

    Returns:
        A :class:`DSRDiagnostics` (frozen dataclass; read fields by
        attribute, e.g. ``diag.dsr``, ``diag.sr_star``). ``dsr`` equals
        :func:`deflated_sharpe_ratio` for the same arguments. The
        ``var_sharpe`` field holds the per-observation value actually used
        (after any ``bars_per_year`` conversion), so
        ``sr_star == sqrt(var_sharpe) * expected_max_z`` always holds.

    Raises:
        TypeError: if ``n_trials`` is not an integer.
        ValueError: on invalid ``n_trials``, non-finite/negative
            ``var_sharpe``, non-positive ``bars_per_year``, or a degenerate
            ``returns`` series.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import deflated_sharpe_ratio, deflated_sharpe_ratio_full
        >>> rng = np.random.default_rng(0)
        >>> returns = rng.normal(0.001, 0.01, 252)
        >>> diag = deflated_sharpe_ratio_full(returns, n_trials=50, var_sharpe=0.01**2)
        >>> abs(diag.dsr - deflated_sharpe_ratio(returns, 50, 0.01**2)) < 1e-12
        True
        >>> diag.n_obs
        252
        >>> diag.sr_star > 0
        True
    """
    _validate_dsr_inputs(n_trials, var_sharpe)
    var_sharpe = _to_per_observation_var(var_sharpe, bars_per_year)
    arr = _validate_returns(returns)
    sr_hat, gamma3, gamma4 = _sharpe_moments(arr)
    expected_max_z, sr_star = _deflated_benchmark(n_trials, var_sharpe)
    dsr = probabilistic_sharpe_ratio(arr, benchmark_skill=sr_star)
    return DSRDiagnostics(
        dsr=dsr,
        observed_sr=sr_hat,
        sr_star=sr_star,
        expected_max_z=expected_max_z,
        var_sharpe=float(var_sharpe),
        n_trials=int(n_trials),
        n_obs=int(arr.size),
        skew=gamma3,
        kurt=gamma4,
    )


def min_track_record_length(
    observed_sharpe: float,
    target_sharpe: float,
    alpha: float,
    skew: float,
    kurtosis: float,
) -> float:
    """Minimum sample size such that PSR(target_sharpe) >= 1 - alpha.

    Inverts the :func:`probabilistic_sharpe_ratio` formula for ``n``:

    .. math::
        n^\\ast = 1 + \\left\\lceil
            \\left(\\frac{\\Phi^{-1}(1 - \\alpha) \\cdot
                   \\sqrt{1 - \\gamma_3 \\widehat{\\text{SR}}
                          + \\frac{\\gamma_4 - 1}{4} \\widehat{\\text{SR}}^2}}
                  {\\widehat{\\text{SR}} - \\text{SR}^\\ast}
            \\right)^2
        \\right\\rceil

    Bailey & Lopez de Prado (2012, Eq. 11).

    Args:
        observed_sharpe: The sample Sharpe ratio you actually observed.
        target_sharpe: The benchmark you want to beat with confidence.
        alpha: Significance level in (0, 1). PSR must meet 1 - alpha.
        skew: Sample skew of the return distribution.
        kurtosis: Sample kurtosis (NOT excess) of the return distribution.

    Returns:
        The minimum number of observations, as a ``float``. When
        ``observed_sharpe <= target_sharpe`` no finite track record can
        establish the gap, so the answer is ``math.inf`` rather than an
        error: "no length is long enough" is a well-defined result. Wrap
        in ``int(...)`` for a count when the value is finite.

    Raises:
        ValueError: if any scalar input is non-finite, if
            ``alpha not in (0, 1)``, or if the higher moments are too
            extreme for the Gaussian approximation.

    Examples:
        >>> import math
        >>> from purgedcv import min_track_record_length
        >>> n = min_track_record_length(
        ...     observed_sharpe=0.5, target_sharpe=0.2,
        ...     alpha=0.05, skew=0.0, kurtosis=3.0,
        ... )
        >>> n > 0
        True
        >>> # No track record proves a Sharpe you have not actually beaten:
        >>> math.isinf(min_track_record_length(0.1, 0.2, 0.05, 0.0, 3.0))
        True
    """
    values = {
        "observed_sharpe": observed_sharpe,
        "target_sharpe": target_sharpe,
        "alpha": alpha,
        "skew": skew,
        "kurtosis": kurtosis,
    }
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}.")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    # The track record is too short no matter how long it runs: if the
    # observed Sharpe does not exceed the target, PSR(target) can never
    # reach 1 - alpha. Report that as infinity instead of raising, so
    # callers can branch on the value rather than wrap in try/except.
    if observed_sharpe <= target_sharpe:
        return math.inf

    z_target = float(stats.norm.ppf(1 - alpha))
    if not np.isfinite(z_target):
        raise ValueError(
            f"alpha={alpha} is too small to compute a finite normal quantile; "
            "use a representable significance level."
        )
    # alpha >= 0.5 puts the 1 - alpha confidence target at or below 0.5.
    # Any track record with observed_sharpe > target_sharpe already clears
    # that at the smallest workable length (PSR needs n >= 2), so return
    # that floor. Falling through would square a non-positive z and
    # wrongly inflate the requirement.
    if z_target <= 0:
        return 2.0
    denom_sq = 1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2
    if not np.isfinite(denom_sq) or denom_sq <= 0:
        raise ValueError(
            f"PSR denominator is non-positive ({denom_sq:.4g}); the input "
            "moments are too extreme for the Gaussian approximation."
        )
    sr_diff = observed_sharpe - target_sharpe
    with np.errstate(over="ignore", invalid="ignore"):
        n_minus_1 = (z_target * np.sqrt(denom_sq) / sr_diff) ** 2
    if not np.isfinite(n_minus_1):
        raise ValueError("inputs imply a non-finite minimum track record length.")
    return float(np.ceil(n_minus_1) + 1)


def effective_n_trials(trial_sharpes: NDArrayAny, method: str = "autocorr") -> int:
    """Estimate the number of *independent* trials behind a correlated search.

    :func:`deflated_sharpe_ratio` assumes the ``n_trials`` candidates were
    drawn independently. Sequential samplers (Optuna's TPE, CMA-ES) draw each
    trial conditioned on the previous ones, so the trials are autocorrelated
    and the raw count overstates how many independent bets were really placed.
    Feeding the raw count inflates the deflated benchmark and makes DSR
    needlessly conservative (often numerically zero). This returns a smaller
    effective count to pass to :func:`deflated_sharpe_ratio` instead.

    The estimate is the run length divided by the integrated autocorrelation
    time of the trial-performance series:
    ``n_eff = n / (1 + 2 * sum_k rho_k)``, summing the autocorrelations
    ``rho_k`` until the first non-positive lag (the initial-positive-sequence
    truncation, Geyer 1992). It is a **heuristic**: it depends on the
    truncation rule and assumes the trial order reflects the sampler's
    dependence, so treat the result as an order-of-magnitude correction, not
    an exact figure.

    Args:
        trial_sharpes: 1-D array of per-trial performance values (Sharpe or
            objective), in the order the trials were run. All finite.
        method: Estimator to use. Only ``"autocorr"`` is implemented.

    Returns:
        Effective trial count, an integer in ``[1, n]``. A near-independent
        series returns close to ``n``; a strongly autocorrelated one returns
        far fewer. A constant series returns 1 (every trial was the same).
        Fewer than 3 trials returns the raw count (too few to estimate).

    Raises:
        ValueError: on an unknown ``method``, an empty or non-1-D array, or
            non-finite values.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import effective_n_trials
        >>> rng = np.random.default_rng(0)
        >>> indep = rng.standard_normal(400)            # independent trials
        >>> drift = np.cumsum(rng.standard_normal(400))  # strongly autocorrelated
        >>> n_indep = effective_n_trials(indep)
        >>> n_drift = effective_n_trials(drift)
        >>> 1 <= n_drift < n_indep <= 400
        True
    """
    if method != "autocorr":
        raise ValueError(f"unknown method {method!r}; only 'autocorr' is supported.")
    arr = np.asarray(trial_sharpes, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"trial_sharpes must be a 1-D array in trial order, got {arr.ndim}-D; "
            "flatten it deliberately if that order is meaningful."
        )
    if arr.size == 0:
        raise ValueError("trial_sharpes must contain at least one value.")
    if not np.isfinite(arr).all():
        raise ValueError("trial_sharpes contains NaN or infinite values.")
    n = arr.size
    if n < 3:
        return int(n)  # too few points to estimate autocorrelation
    centered = arr - arr.mean()
    var = float(np.dot(centered, centered))
    if var == 0.0:
        return 1  # every trial identical: maximally correlated
    # Autocorrelation at lags 0..n-1 (acf[0] == 1).
    acf = np.correlate(centered, centered, mode="full")[n - 1 :] / var
    tau = 1.0
    for k in range(1, n):
        rho = float(acf[k])
        if rho <= 0.0:
            break
        tau += 2.0 * rho
    n_eff = round(n / tau)
    return int(max(1, min(n, n_eff)))
