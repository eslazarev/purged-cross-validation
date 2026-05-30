"""Optuna + Deflated Sharpe Ratio: the end-to-end pattern, in one file.

Tune a strategy with Optuna, record each trial's Sharpe with
``TrialSharpeRecorder``, then deflate the champion's Sharpe by how much you
really searched. This is the workflow most users write, and copying it avoids
the two easy traps:

1. Units. The objective here records an *annualised* Sharpe (what dashboards
   show). ``deflated_sharpe_ratio_full`` compares against a *per-bar* Sharpe,
   so we pass ``bars_per_year`` and let it convert ``var_sharpe`` for us.
2. Correlated trials. TPE draws each trial from the last ones, so the raw
   trial count overstates the independent search effort and makes DSR
   needlessly harsh. ``recorder.n_effective()`` gives the count to deflate by.

Requires the optuna extra::

    pip install purgedcv[optuna]
"""

from __future__ import annotations

import numpy as np
import optuna

from purgedcv import deflated_sharpe_ratio_full
from purgedcv.optuna_integration import TrialSharpeRecorder

BARS_PER_YEAR = 252


def strategy_returns(threshold: float, seed: int = 0) -> np.ndarray:
    """Toy long/flat strategy gated by ``threshold``. Returns one per-bar
    return series. The point is the workflow, not the (absent) alpha."""
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal(BARS_PER_YEAR * 2)
    market = 0.0005 + 0.01 * rng.standard_normal(BARS_PER_YEAR * 2)
    position = (signal > threshold).astype(float)
    return position * market


def per_bar_sharpe(returns: np.ndarray) -> float:
    std = float(returns.std(ddof=1))
    return float(returns.mean() / std) if std > 0 else 0.0


def objective(trial: optuna.Trial) -> float:
    threshold = trial.suggest_float("threshold", -1.0, 1.0)
    sharpe = per_bar_sharpe(strategy_returns(threshold))
    # Store the ANNUALISED Sharpe; bars_per_year carries the unit through to
    # the deflation below.
    trial.set_user_attr("sharpe", sharpe * np.sqrt(BARS_PER_YEAR))
    return sharpe  # optimise the per-bar Sharpe


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    recorder = TrialSharpeRecorder()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
    )
    study.optimize(objective, n_trials=120, callbacks=[recorder])

    champion_returns = strategy_returns(study.best_params["threshold"])
    diag = deflated_sharpe_ratio_full(
        champion_returns,
        n_trials=recorder.n_effective(),  # TPE trials are correlated: effective count
        var_sharpe=recorder.var_sharpe(),  # annualised (objective recorded annualised) ...
        bars_per_year=BARS_PER_YEAR,  # ... so convert it to per-bar here
    )

    print(f"raw trials       : {recorder.n_trials()}")
    print(f"effective trials : {recorder.n_effective()}")
    print(f"observed SR/bar  : {diag.observed_sr:+.4f}")
    print(
        f"deflated bench   : sr_star={diag.sr_star:+.4f}  expected_max_z={diag.expected_max_z:.3f}"
    )
    print(f"DSR              : {diag.dsr:.4f}")


if __name__ == "__main__":
    main()
