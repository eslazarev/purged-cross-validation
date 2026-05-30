"""Unit tests for per-path backtest metrics (D6 helper)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from purgedcv import default_backtest_metrics, path_metrics


class TestDefaultBacktestMetrics:
    def test_known_values(self) -> None:
        m = default_backtest_metrics(np.array([0.01, -0.02, 0.03, 0.0]))
        assert m["total_return"] == pytest.approx(0.02)
        # equity = [0.01, -0.01, 0.02, 0.02]; trough at -0.01 from peak 0.01.
        assert m["max_drawdown"] == pytest.approx(0.02)
        assert m["calmar"] == pytest.approx(0.02 / 0.02)

    def test_sharpe_matches_mean_over_std(self) -> None:
        r = np.array([0.01, 0.02, -0.01, 0.03])
        m = default_backtest_metrics(r)
        assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1))

    def test_annualisation_scales_sharpe(self) -> None:
        r = np.array([0.01, 0.02, -0.01, 0.03, 0.005, -0.002])
        per_bar = default_backtest_metrics(r)["sharpe"]
        annual = default_backtest_metrics(r, bars_per_year=252)["sharpe"]
        assert annual == pytest.approx(per_bar * np.sqrt(252))

    def test_nan_entries_dropped(self) -> None:
        clean = default_backtest_metrics(np.array([0.01, 0.02, 0.03]))
        withnan = default_backtest_metrics(np.array([0.01, np.nan, 0.02, 0.03]))
        # Dropping the NaN leaves [0.01, 0.02, 0.03]; same as clean's tail.
        assert withnan["total_return"] == pytest.approx(0.06)
        assert clean["total_return"] == pytest.approx(0.06)

    def test_too_few_finite_values_is_all_nan(self) -> None:
        m = default_backtest_metrics(np.array([np.nan, 0.01]))
        assert all(np.isnan(v) for v in m.values())

    def test_monotone_increasing_path_has_no_drawdown(self) -> None:
        m = default_backtest_metrics(np.array([0.01, 0.01, 0.01, 0.01]))
        assert m["max_drawdown"] == 0.0
        assert np.isnan(m["calmar"])  # division by zero drawdown -> nan

    def test_drawdown_counts_opening_loss(self) -> None:
        """A path that opens with a loss has a drawdown from the pre-trade
        zero peak; without seeding equity at 0 it would report none."""
        m = default_backtest_metrics(np.array([-0.1, 0.05]))
        assert m["max_drawdown"] == pytest.approx(0.1)

    def test_all_loss_path_drawdown_is_total_loss(self) -> None:
        m = default_backtest_metrics(np.array([-0.02, -0.03]))
        assert m["max_drawdown"] == pytest.approx(0.05)
        assert m["total_return"] == pytest.approx(-0.05)

    def test_rejects_non_positive_bars_per_year(self) -> None:
        for bad in (-252, 0):
            with pytest.raises(ValueError, match="bars_per_year"):
                default_backtest_metrics(np.array([0.01, -0.02, 0.03]), bars_per_year=bad)

    def test_rejects_non_finite_bars_per_year(self) -> None:
        for bad in (float("nan"), float("inf")):
            with pytest.raises(ValueError, match="bars_per_year"):
                default_backtest_metrics(
                    np.array([0.01, -0.02, 0.03]),
                    bars_per_year=bad,  # type: ignore[arg-type]
                )

    def test_rejects_bool_bars_per_year(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            default_backtest_metrics(np.array([0.01, -0.02, 0.03]), bars_per_year=True)


class TestPathMetrics:
    def test_dataframe_shape_and_columns(self) -> None:
        paths = np.array([[0.01, -0.02, 0.03, 0.0], [0.0, 0.01, 0.01, 0.01]])
        df = path_metrics(paths)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 4)
        assert list(df.columns) == ["sharpe", "calmar", "max_drawdown", "total_return"]
        assert df.index.name == "path"

    def test_rows_match_elementwise_default(self) -> None:
        rng = np.random.default_rng(0)
        paths = rng.standard_normal((5, 50)) * 0.01
        df = path_metrics(paths, bars_per_year=252)
        for i in range(paths.shape[0]):
            expected = default_backtest_metrics(paths[i], bars_per_year=252)
            assert df.iloc[i]["sharpe"] == pytest.approx(expected["sharpe"])
            assert df.iloc[i]["total_return"] == pytest.approx(expected["total_return"])

    def test_custom_metric_fn(self) -> None:
        paths = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        df = path_metrics(paths, metric_fn=lambda p: {"mean": float(np.mean(p))})
        assert list(df.columns) == ["mean"]
        assert df["mean"].tolist() == [2.0, 5.0]

    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            path_metrics(np.zeros(10))

    def test_handles_nan_rows_from_collapsed_folds(self) -> None:
        """A path that is entirely NaN (a fold that never fit) yields NaN
        metrics rather than raising."""
        paths = np.array([[0.01, 0.02, 0.03], [np.nan, np.nan, np.nan]])
        df = path_metrics(paths)
        assert np.isnan(df.iloc[1]["sharpe"])
        assert not np.isnan(df.iloc[0]["sharpe"])
