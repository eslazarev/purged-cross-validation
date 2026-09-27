# Backtest overfitting and Optuna

Estimate selection overfitting with PBO, or record a search for the
[deflated-Sharpe calculation](metrics.md#purgedcv.deflated_sharpe_ratio).
The Optuna integration is optional; install `purgedcv[optuna]` to run an Optuna study.

## Backtest overfitting

::: purgedcv.PBOResult

::: purgedcv.PerformanceMetric

::: purgedcv.probability_of_backtest_overfitting

## Optuna integration

::: purgedcv.optuna_integration.TrialSharpeRecorder
