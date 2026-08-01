"""Equivalence: every accepted time-input container yields identical results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purgedcv import PurgedKFold, apply_embargo, purge
from purgedcv._typing import NDArrayAny, TimesLike
from purgedcv.diagnostics import compute_overlap_fraction

N = 40
_PRED_PD = pd.Series(pd.date_range("2024-01-01", periods=N, freq="D"))
_EVAL_PD = _PRED_PD + pd.Timedelta(days=2)
_TRAIN = np.arange(0, 20)
_TEST = np.arange(25, 30)


def _variants() -> dict[str, tuple[TimesLike, TimesLike]]:
    variants: dict[str, tuple[TimesLike, TimesLike]] = {
        "pandas_series": (_PRED_PD, _EVAL_PD),
        "datetimeindex": (pd.DatetimeIndex(_PRED_PD), pd.DatetimeIndex(_EVAL_PD)),
        "numpy_datetime64": (_PRED_PD.to_numpy(), _EVAL_PD.to_numpy()),
        "python_list": (list(_PRED_PD), list(_EVAL_PD)),
    }
    try:
        import polars as pl

        variants["polars_series"] = (pl.Series(_PRED_PD), pl.Series(_EVAL_PD))
    except ImportError:
        pass
    return variants


VARIANTS = _variants()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_purge_equivalent_across_inputs(name: str) -> None:
    pred, evalu = VARIANTS[name]
    # purge()'s purge_horizon param is pd.Timedelta | None (never widened to the
    # HorizonLike string-parsing accepted by the splitters/diagnostics), so a
    # concrete Timedelta is used here rather than "1D".
    baseline = purge(_TRAIN, _TEST, _PRED_PD, _EVAL_PD, purge_horizon=pd.Timedelta(days=1))
    result = purge(_TRAIN, _TEST, pred, evalu, purge_horizon=pd.Timedelta(days=1))
    np.testing.assert_array_equal(result, baseline)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_apply_embargo_equivalent_across_inputs(name: str) -> None:
    pred, evalu = VARIANTS[name]
    baseline = apply_embargo(_TRAIN, _TEST, _PRED_PD, _EVAL_PD, pd.Timedelta(days=2))
    result = apply_embargo(_TRAIN, _TEST, pred, evalu, pd.Timedelta(days=2))
    np.testing.assert_array_equal(result, baseline)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_overlap_fraction_equivalent_across_inputs(name: str) -> None:
    pred, evalu = VARIANTS[name]
    baseline = compute_overlap_fraction(_TRAIN, _TEST, _PRED_PD, _EVAL_PD)
    result = compute_overlap_fraction(_TRAIN, _TEST, pred, evalu)
    assert result == baseline


@pytest.mark.parametrize("name", list(VARIANTS))
def test_split_equivalent_across_inputs(name: str) -> None:
    pred, evalu = VARIANTS[name]
    X: NDArrayAny = np.zeros((N, 1))  # noqa: N806
    base = list(
        PurgedKFold(n_splits=4, prediction_times=_PRED_PD, evaluation_times=_EVAL_PD).split(X)
    )
    got = list(PurgedKFold(n_splits=4, prediction_times=pred, evaluation_times=evalu).split(X))
    assert len(got) == len(base)
    for (tr_b, te_b), (tr_g, te_g) in zip(base, got, strict=True):
        np.testing.assert_array_equal(tr_g, tr_b)
        np.testing.assert_array_equal(te_g, te_b)


def test_timedelta64_family_supported() -> None:
    # A timedelta-family dataset (offsets from an epoch) is also accepted.
    pred_td = np.arange(N, dtype="timedelta64[D]")
    evalu_td = pred_td + np.timedelta64(2, "D")
    f = compute_overlap_fraction(_TRAIN, _TEST, pred_td, evalu_td)
    assert 0.0 <= f <= 1.0
