"""Internal: statistical metrics (Domain D7).

Three closed-form metrics that correct reported Sharpe ratios:

- :func:`probabilistic_sharpe_ratio` (PSR) — Bailey & Lopez de Prado (2012).
- :func:`deflated_sharpe_ratio` (DSR) — Bailey & Lopez de Prado (2014).
- :func:`min_track_record_length` (MinTRL) — derived from PSR by inversion.

References:
- Bailey, D. H., & Lopez de Prado, M. (2012). The Sharpe Ratio Efficient
  Frontier. Journal of Risk 15(2).
- Bailey, D. H., & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
  Journal of Portfolio Management 40(5).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ._typing import NDArrayAny


def _validate_returns(returns: NDArrayAny) -> NDArrayAny:
    """Coerce input to 1-D float ndarray, reject non-finite values and length < 2."""
    arr = np.asarray(returns, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"returns must have at least 2 observations, got {arr.size}.")
    if not np.isfinite(arr).all():
        raise ValueError("returns contains NaN or infinite values; filter with np.isfinite first.")
    return arr


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
    denominator_sq = 1 - gamma3 * sr_hat + (gamma4 - 1) / 4 * sr_hat**2
    if not np.isfinite(denominator_sq) or denominator_sq <= 0:
        raise ValueError(
            f"PSR denominator is non-positive ({denominator_sq:.4g}); the "
            "input distribution's higher moments are too extreme for the "
            "Gaussian approximation."
        )
    z = (sr_hat - benchmark_skill) * np.sqrt(n - 1) / np.sqrt(denominator_sq)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    returns: NDArrayAny,
    n_trials: int,
    var_sharpe: float,
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

    Returns:
        Scalar probability in [0, 1].

    Raises:
        TypeError: if ``n_trials`` is not an integer.
        ValueError: on invalid ``n_trials`` or non-finite/negative
            ``var_sharpe``.

    Examples:
        >>> import numpy as np
        >>> from purgedcv import deflated_sharpe_ratio
        >>> rng = np.random.default_rng(0)
        >>> returns = rng.normal(0.001, 0.01, 252)
        >>> dsr = deflated_sharpe_ratio(returns, n_trials=50, var_sharpe=0.01**2)
        >>> 0.0 <= dsr <= 1.0
        True
    """
    from math import e

    if isinstance(n_trials, bool) or not isinstance(n_trials, (int, np.integer)):
        raise TypeError(f"n_trials must be an integer, got {type(n_trials).__name__}.")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if not np.isfinite(var_sharpe):
        raise ValueError(f"var_sharpe must be finite, got {var_sharpe}.")
    if var_sharpe < 0:
        raise ValueError(f"var_sharpe must be non-negative, got {var_sharpe}.")

    gamma_em = 0.5772156649015329  # Euler-Mascheroni constant
    # With a single trial there is no multiple-comparison correction: the
    # expected maximum Sharpe under the null across one trial is 0, so
    # SR* = 0 and DSR reduces to PSR against a zero benchmark. The
    # extreme-value formula below is valid only for n_trials >= 2, where
    # Phi_inv(1 - 1/n_trials) is finite (it diverges to -inf at n = 1).
    if n_trials == 1:
        sr_star: float = 0.0
    else:
        sr_star = float(
            np.sqrt(var_sharpe)
            * (
                (1 - gamma_em) * stats.norm.ppf(1 - 1 / n_trials)
                + gamma_em * stats.norm.ppf(1 - 1 / (n_trials * e))
            )
        )

    return probabilistic_sharpe_ratio(returns, benchmark_skill=sr_star)


def min_track_record_length(
    observed_sharpe: float,
    target_sharpe: float,
    alpha: float,
    skew: float,
    kurtosis: float,
) -> int:
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
            Must be strictly greater than ``target_sharpe``.
        target_sharpe: The benchmark you want to beat with confidence.
        alpha: Significance level in (0, 1). PSR must meet 1 - alpha.
        skew: Sample skew of the return distribution.
        kurtosis: Sample kurtosis (NOT excess) of the return distribution.

    Returns:
        Minimum integer sample size.

    Raises:
        ValueError: if any scalar input is non-finite, if
            ``observed_sharpe <= target_sharpe``, or if
            ``alpha not in (0, 1)``.

    Examples:
        >>> from purgedcv import min_track_record_length
        >>> n = min_track_record_length(
        ...     observed_sharpe=0.5, target_sharpe=0.2,
        ...     alpha=0.05, skew=0.0, kurtosis=3.0,
        ... )
        >>> n > 0
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
    if observed_sharpe <= target_sharpe:
        raise ValueError(
            f"observed_sharpe ({observed_sharpe}) must be strictly greater "
            f"than target_sharpe ({target_sharpe})."
        )
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

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
        return 2
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
    return int(np.ceil(n_minus_1) + 1)
