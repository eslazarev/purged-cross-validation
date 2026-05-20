# Examples

The repository ships a controlled proof and ten worked notebooks on real
public data. The point is to show the honest range of outcomes, including
the undramatic ones, not only the alarming cases.

The notebooks live in
[`examples/`](https://github.com/eslazarev/purged-cross-validation/tree/main/examples)
and run on `pip install "purgedcv[examples]"`. Each one downloads its
dataset on first run and caches it in `examples/data/` (git-ignored), so
subsequent runs are fully offline. The
[examples README](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/README.md)
documents per-notebook data sources, licences, and download sizes.

## Controlled proof (no download)

[**synthetic_leakage_proof**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/synthetic_leakage_proof.ipynb)
— a target nothing can predict next to a monotone clock feature. Naive
shuffled k-fold scores R² between 0.83 and 0.91 on noise; `PurgedKFold`
drives the train/test label overlap from 100 % to 0 % and the score
collapses below a predict-the-mean baseline. Deterministic, fixed seed,
no network.

## Real datasets — dramatic leak

[**ohlc_trading_signal**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/ohlc_trading_signal.ipynb)
— Binance BTC/USDT daily bars, 20-bar forward return. Naive shuffled
k-fold reports R² ≈ +0.85 for a feature with no economic reason to
forecast returns; `PurgedKFold` takes it to about −1.2.

[**earthquake_magnitude_leakage**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/earthquake_magnitude_leakage.ipynb)
— USGS catalogue, M5+. Magnitude is unpredictable from past magnitudes by
the Gutenberg–Richter law (empirical autocorrelation +0.02); naive
shuffled k-fold still prints R² = +0.65. Purged (−0.75), blocked (−1.13),
and walk-forward (−1.24) all return the correct verdict of no skill.

[**air_quality_clock_leakage**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/air_quality_clock_leakage.ipynb)
— UCI air quality, 72-hour benzene horizon. Three ordinary lag features
give naive R² ≈ 0.07. Adding one innocuous cumulative-hour counter
sends naive R² to 0.99 while `PurgedKFold` (−1.52) and `WalkForwardSplit`
(−0.81) do not move. The mechanism behind every phantom edge in one
notebook.

## Real datasets — measured, undramatic

[**uk_smart_meter_lcl**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/uk_smart_meter_lcl.ipynb)
— UK Power Networks Low Carbon London demand. On the full population of
4,284 eligible Standard-tariff households measured by
`tools/lcl_full_benchmark.py` (20 seeded subsamples of 60), the
temporal-leakage gap between naive shuffled k-fold and walk-forward is
small (1.60 %, 95 % CI 1.27 – 1.94 %). The leak that actually bites is by
household: scoring on unseen customers is 6.03 % worse than the pooled
temporal estimate (95 % CI 4.93 – 7.12 %). Which split you need follows
from what you intend to deploy on.

[**model_comparison_honest_cv**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/model_comparison_honest_cv.ipynb)
— same BTC data, six candidate models. Once the Deflated Sharpe Ratio
corrects for the number of trials, no model clears DSR ≥ 0.95. Reporting
no edge is the correct outcome.

[**epl_match_prediction**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/epl_match_prediction.ipynb)
— Premier League sports modelling. The honest result is calibration drift
across seasons rather than a headline accuracy gap.

## API-coverage examples

[**clinical_mortality_physionet**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/clinical_mortality_physionet.ipynb)
— PhysioNet ICU mortality, `PurgedGroupKFold` holding whole patients out.

[**predictive_maintenance_nasa**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/predictive_maintenance_nasa.ipynb)
— NASA C-MAPSS turbofan remaining-useful-life, `WalkForwardSplit` with
an expanding training window.

[**precipitation_noaa**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/precipitation_noaa.ipynb)
— NOAA GHCN-Daily rainfall, `PurgedKFold(purge_horizon="2D", embargo="1D")`
on a one-day-ahead regression.

[**energy_demand_pjm**](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/energy_demand_pjm.ipynb)
— PJM hourly load. The full toolkit: `CombinatorialPurgedCV` + backtest
paths + `probabilistic_sharpe_ratio` + `deflated_sharpe_ratio` +
`min_track_record_length`. The numerical reproduction of §7.4.1 from
*Advances in Financial Machine Learning*.

## Reproducing the full LCL benchmark

The full-population LCL result above is produced offline by
`tools/lcl_full_benchmark.py` over the raw ~8 GB corpus. The tool, the
shared notebook harness (`examples/_lcl_harness.py`), and the per-subsample
CSV summary are documented in the
[smart-meter notebook](https://github.com/eslazarev/purged-cross-validation/blob/main/examples/uk_smart_meter_lcl.ipynb)
and in the [JOSS paper](paper.md). Reproduce with:

```bash
python tools/lcl_full_benchmark.py --k 20 --n 60 --seed 0
```
