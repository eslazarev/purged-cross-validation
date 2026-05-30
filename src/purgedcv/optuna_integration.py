"""Optuna integration helpers (optional extra: ``pip install purgedcv[optuna]``).

:func:`~purgedcv.deflated_sharpe_ratio` needs ``var_sharpe``: the variance of
the Sharpe ratios across the trials you searched. Optuna only stores each
trial's objective value, so users hand-roll the bookkeeping every time:

    trial.set_user_attr("sharpe", sharpe_of(pnl))
    ...
    shs = [t.user_attrs["sharpe"] for t in study.trials if "sharpe" in t.user_attrs]
    var_sharpe = np.var(shs, ddof=1)

:class:`TrialSharpeRecorder` is a study callback that does exactly that.

Importing this module does **not** require Optuna. The recorder is a plain
callback that reads attributes off whatever ``(study, trial)`` pair Optuna
passes it, so it has no import-time dependency; the ``optuna`` extra only
installs Optuna itself for the surrounding optimisation loop.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ._typing import NDArrayAny

__all__ = ["TrialSharpeRecorder"]


class _TrialLike(Protocol):
    """The slice of an Optuna ``FrozenTrial`` the recorder reads."""

    value: float | None

    @property
    def user_attrs(self) -> dict[str, Any]: ...


class TrialSharpeRecorder:
    """Optuna study callback that collects per-trial Sharpe ratios.

    Pass an instance as a callback to ``study.optimize`` and store each
    trial's Sharpe in a user attribute (default key ``"sharpe"``); the
    recorder accumulates them and reports the variance for
    :func:`~purgedcv.deflated_sharpe_ratio`. If a trial has no such user
    attribute, the recorder falls back to that trial's objective ``value``.
    Non-finite and missing values are ignored.

    Args:
        attr: The ``trial.user_attrs`` key under which the Sharpe ratio is
            stored. Defaults to ``"sharpe"``.

    Examples:
        >>> from types import SimpleNamespace
        >>> from purgedcv.optuna_integration import TrialSharpeRecorder
        >>> recorder = TrialSharpeRecorder()
        >>> # Optuna calls the recorder with (study, frozen_trial) per trial;
        >>> # SimpleNamespace stands in for the trial here.
        >>> for s in (1.2, 0.8, 1.5):
        ...     recorder(study=None, trial=SimpleNamespace(value=s, user_attrs={"sharpe": s}))
        >>> recorder.n_trials()
        3
        >>> round(recorder.var_sharpe(), 4)
        0.1233

    Usage with a real study::

        recorder = TrialSharpeRecorder()
        study.optimize(objective, n_trials=300, callbacks=[recorder])
        dsr = deflated_sharpe_ratio(
            best_returns, n_trials=recorder.n_trials(), var_sharpe=recorder.var_sharpe()
        )
    """

    def __init__(self, attr: str = "sharpe") -> None:
        self._attr = attr
        self._sharpes: list[float] = []

    def __call__(self, study: object, trial: _TrialLike) -> None:
        """Record one trial's Sharpe ratio. Matches the Optuna callback signature.

        Missing, ``None``, non-numeric, and non-finite values are skipped
        rather than raised: a callback that throws would abort the whole
        ``study.optimize`` run over a single malformed trial.
        """
        raw = trial.user_attrs.get(self._attr, trial.value)
        if raw is None:
            return
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return  # malformed attribute (e.g. a string); treat as unusable
        if np.isfinite(value):
            self._sharpes.append(value)

    def sharpes(self) -> NDArrayAny:
        """The recorded Sharpe ratios, in trial order, as a 1-D array."""
        return np.array(self._sharpes, dtype=float)

    def n_trials(self) -> int:
        """Number of trials with a usable (finite) Sharpe ratio recorded."""
        return len(self._sharpes)

    def n_effective(self, method: str = "autocorr") -> int:
        """Effective number of *independent* trials, for correlated samplers.

        TPE and CMA-ES draw each trial conditioned on the previous ones, so
        the raw trial count overstates the independent search effort and makes
        :func:`~purgedcv.deflated_sharpe_ratio` overly conservative. This
        applies :func:`~purgedcv.effective_n_trials` to the recorded Sharpe
        series; pass the result as ``n_trials`` to deflate more realistically.
        Returns 0 when nothing has been recorded yet. See
        :func:`~purgedcv.effective_n_trials` for the heuristic's caveats.
        """
        from purgedcv._metrics import effective_n_trials

        arr = self.sharpes()
        if arr.size == 0:
            return 0
        return effective_n_trials(arr, method=method)

    def var_sharpe(self, ddof: int = 1) -> float:
        """Variance of the recorded Sharpe ratios, for ``deflated_sharpe_ratio``.

        Returns ``nan`` until more than ``ddof`` trials have been recorded.

        UNITS: the variance is in the units of whatever Sharpe you stored
        per trial. :func:`~purgedcv.deflated_sharpe_ratio` compares against
        the per-period Sharpe of its ``returns`` argument, so record a
        per-period Sharpe in each trial (the same convention as the
        ``returns`` you will later deflate). Do that and the recorder feeds
        ``deflated_sharpe_ratio`` directly, with no unit conversion.
        """
        if ddof < 0:
            raise ValueError(f"ddof must be >= 0, got {ddof}.")
        arr = self.sharpes()
        if arr.size <= ddof:
            return float("nan")
        return float(arr.var(ddof=ddof))
