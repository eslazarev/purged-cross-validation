"""Unit tests for the Optuna integration helpers.

These run without Optuna installed: ``TrialSharpeRecorder`` is a plain
callback, so a ``SimpleNamespace`` stands in for the frozen trial Optuna
would otherwise pass.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from purgedcv.optuna_integration import TrialSharpeRecorder


def _trial(value: float | None, **user_attrs: float) -> SimpleNamespace:
    return SimpleNamespace(value=value, user_attrs=dict(user_attrs))


class TestTrialSharpeRecorder:
    def test_records_from_user_attr(self) -> None:
        rec = TrialSharpeRecorder()
        for s in (1.2, 0.8, 1.5):
            rec(study=None, trial=_trial(s, sharpe=s))
        assert rec.n_trials() == 3
        assert np.allclose(rec.sharpes(), [1.2, 0.8, 1.5])
        assert rec.var_sharpe() == pytest.approx(np.var([1.2, 0.8, 1.5], ddof=1))

    def test_falls_back_to_trial_value(self) -> None:
        """When the user attribute is absent, the objective value is used."""
        rec = TrialSharpeRecorder()
        rec(study=None, trial=_trial(0.9))  # no 'sharpe' attr
        assert rec.sharpes().tolist() == [0.9]

    def test_custom_attr_key(self) -> None:
        rec = TrialSharpeRecorder(attr="ann_sharpe")
        rec(study=None, trial=_trial(0.0, ann_sharpe=2.1))
        assert rec.sharpes().tolist() == [2.1]

    def test_ignores_none_and_non_finite(self) -> None:
        rec = TrialSharpeRecorder()
        rec(study=None, trial=_trial(None))  # pruned/failed trial
        rec(study=None, trial=_trial(float("nan"), sharpe=float("nan")))
        rec(study=None, trial=_trial(float("inf"), sharpe=float("inf")))
        rec(study=None, trial=_trial(1.0, sharpe=1.0))
        assert rec.n_trials() == 1
        assert rec.sharpes().tolist() == [1.0]

    def test_non_numeric_attr_is_skipped_not_raised(self) -> None:
        """A malformed (non-numeric) user attribute must not abort the study:
        the recorder skips it, like None and non-finite values."""
        rec = TrialSharpeRecorder()
        rec(study=None, trial=_trial(1.0, sharpe="oops"))  # type: ignore[arg-type]
        rec(study=None, trial=_trial(None, sharpe=object()))  # type: ignore[arg-type]
        rec(study=None, trial=_trial(2.0, sharpe=2.0))
        assert rec.n_trials() == 1
        assert rec.sharpes().tolist() == [2.0]

    def test_n_effective_ties_to_effective_n_trials(self) -> None:
        from purgedcv import effective_n_trials

        rng = np.random.default_rng(7)
        rec = TrialSharpeRecorder()
        walk = np.cumsum(rng.standard_normal(400))  # correlated trial path
        for s in walk:
            rec(study=None, trial=_trial(float(s), sharpe=float(s)))
        assert rec.n_effective() == effective_n_trials(rec.sharpes())
        assert rec.n_effective() < rec.n_trials()

    def test_n_effective_zero_when_empty(self) -> None:
        assert TrialSharpeRecorder().n_effective() == 0

    def test_var_sharpe_rejects_negative_ddof(self) -> None:
        rec = TrialSharpeRecorder()
        with pytest.raises(ValueError, match="ddof"):
            rec.var_sharpe(ddof=-1)

    def test_var_sharpe_nan_below_two_trials(self) -> None:
        rec = TrialSharpeRecorder()
        assert np.isnan(rec.var_sharpe())
        rec(study=None, trial=_trial(1.0, sharpe=1.0))
        assert np.isnan(rec.var_sharpe())  # ddof=1 needs > 1 sample
        rec(study=None, trial=_trial(2.0, sharpe=2.0))
        assert not np.isnan(rec.var_sharpe())

    def test_feeds_deflated_sharpe_ratio(self) -> None:
        """End-to-end shape: the recorder's outputs slot straight into DSR."""
        from purgedcv import deflated_sharpe_ratio

        rng = np.random.default_rng(0)
        rec = TrialSharpeRecorder()
        for s in rng.normal(1.0, 0.3, 40):
            rec(study=None, trial=_trial(float(s), sharpe=float(s)))
        returns = rng.normal(0.001, 0.01, 252)
        dsr = deflated_sharpe_ratio(returns, n_trials=rec.n_trials(), var_sharpe=rec.var_sharpe())
        assert 0.0 <= dsr <= 1.0
