# purged-cross-validation Example Notebooks

Domain-specific notebooks that demonstrate the `purged-cross-validation` library on real-world
time-series problems. Each notebook downloads its dataset on first run and caches it in
`examples/data/` (git-ignored). Subsequent runs use the local cache and work offline.

The `examples/data/` directory is created automatically by the notebooks and is excluded
from git via `.gitignore`, so reviewers see fresh downloads on their own machine.

## How to run locally

```bash
# Install library + notebook dependencies
pip install purgedcv[examples]

# Launch Jupyter
jupyter notebook examples/
```

---

## Controlled proof (no download, deterministic)

`synthetic_leakage_proof.ipynb` is the one example with a known right answer.
The target is built so no feature can predict it, then naive shuffled k-fold
and `PurgedKFold` run side by side. Naive scores R² ≈ 0.83–0.91 on a target
nothing can predict; `PurgedKFold` drops the train/test label overlap from 100%
to 0% and that fabricated skill collapses below a predict-the-mean baseline.
Deterministic (fixed seed) and offline — no download.

---

## Notebooks

### 1. `clinical_mortality_physionet.ipynb` — ICU Mortality Prediction

**Dataset:** PhysioNet Computing in Cardiology Challenge 2012
("Predicting Mortality of ICU Patients") — set-a, 4,000 ICU patients.

**Source URL:** https://physionet.org/content/challenge-2012/1.0.0/

**Files downloaded:**
- `set-a.tar.gz` (~7 MB) — per-patient hourly vital-sign records
- `Outcomes-a.txt` (~77 KB) — in-hospital mortality labels

**License:** Open Data Commons Attribution License (ODC-By) v1.0

**Download size:** ~7 MB on first run

`PurgedGroupKFold` on binary ICU mortality. The notebook reads 4,000 patient
files from the tarball, pivots the hourly measurements (HR, SysABP, Temp, GCS)
to wide format, joins the mortality labels, and subsamples 200 patients
stratified by outcome to keep it fast. It then runs naive `KFold` against
`PurgedGroupKFold`. The naive split mixes one patient across train and test and
inflates AUC. `PurgedGroupKFold` assigns whole patients to folds and purges the
6-hour label boundary. `diagnostics.assert_groups_disjoint` provides the audit
trail.

---

### 2. `predictive_maintenance_nasa.ipynb` — Turbofan RUL Regression

**Dataset:** NASA C-MAPSS FD001 (Commercial Modular Aero-Propulsion System Simulation)
— 100 turbofan engines, run-to-failure sensor streams.

**Source URL:** https://github.com/hankroark/Turbofan-Engine-Degradation
(stable GitHub mirror of NASA PCoE distribution)

**Files downloaded:**
- `train_FD001.txt` (~3.4 MB) — training engine sensor streams
- `test_FD001.txt` (~2.1 MB) — test engine sensor streams
- `RUL_FD001.txt` (~429 B) — held-out RUL values for test engines

**License:** NASA Open Government License

**Download size:** ~6 MB on first run

`WalkForwardSplit` with an expanding training window on RUL regression. The
data is 100 engines with 26 columns: engine_id, cycle, 3 operational settings,
21 sensors. RUL is max(cycle) - cycle per engine. Each cycle maps to a synthetic
calendar date so `WalkForwardSplit` can use its timestamp-based API.
`diagnostics.assert_no_temporal_leakage` checks that no fold has future data in
its training set.

---

### 3. `precipitation_noaa.ipynb` — Next-Day Rainfall Forecasting

**Dataset:** NOAA GHCN-Daily (Global Historical Climatology Network), station
USW00014732 (La Guardia Airport, New York City).

**Source URL:**
https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00014732.csv

**Files downloaded:**
- `USW00014732.csv` (~11 MB) — full daily record for La Guardia

**License:** U.S. Government public domain (NOAA/NCEI)

**Download size:** ~11 MB on first run

`PurgedKFold` on a 1-day-ahead precipitation regression. The notebook filters
to the last 10 years, about 3,650 daily rows. Key columns are PRCP (tenths of
mm), TMAX/TMIN (tenths of °C), and AWND (average wind); features are lag-1,
lag-7, and the temperature range. It compares naive `KFold` with
`PurgedKFold(purge_horizon="2D", embargo="1D")`, and uses
`diagnostics.compute_overlap_fraction` to measure the leakage in the naive
split and confirm it drops to zero under `PurgedKFold`.

---

### 4. `energy_demand_pjm.ipynb` — Day-Ahead Electricity Demand (Full Showcase)

**Dataset:** PJM Interconnection hourly load (1998–2002).

**Source URL:**
https://github.com/panambY/Hourly_Energy_Consumption
(public GitHub mirror of PJM historical data)

**Files downloaded:**
- `PJM_Load_hourly.csv` (~900 KB) — hourly load for the full PJM region

**License:** CC0 1.0 Universal (Public Domain Dedication)

**Download size:** ~900 KB on first run

The full showcase. It runs the whole toolkit: `CombinatorialPurgedCV`,
`backtest_paths`, and the three statistical metrics. Hourly data is resampled
to daily mean, about 1,372 rows. Features are lag-1, lag-7, a rolling 7-day
mean, and a day-of-week one-hot. `CombinatorialPurgedCV(n_splits=6,
n_test_groups=2)` gives C(6,2)=15 folds and C(5,1)=5 independent backtest
paths, fitted with `GradientBoostingRegressor`. Per-path prediction errors
become a returns-like skill score against a persistence baseline, scored with
`probabilistic_sharpe_ratio` (PSR), `deflated_sharpe_ratio` (DSR, corrected for
20 trials), and `min_track_record_length`. A bar chart shows the spread of PSR
across the 5 paths, which is the point of CPCV over single-path walk-forward.

---

### 5. `ohlc_trading_signal.ipynb` — Crypto OHLC Forward-Return Leakage

**Dataset:** Binance spot BTC/USDT daily candles, pinned window
2021-01-01 to 2023-09-28 (1001 bars).

**Source:** the [`pricehub`](https://pypi.org/project/pricehub/) package
(`get_ohlc("binance_spot", "BTCUSDT", "1d", ...)`), an examples-extra
dependency. No API key.

**Files cached:**
- `btcusdt_1d_binance_spot_2021_2023.csv` (~0.1 MB) — fetched once, then offline

**License:** Binance public market data; subject to the exchange's API terms

**Download size:** ~0.1 MB on first run

The library's home domain. The label is the log return over the next 20
bars, so neighbouring labels overlap heavily; the feature is cumulative
volume, a real quantity that doubles as an accidental clock. Naive shuffled
k-fold reports R² ≈ +0.85 for a feature with no economic reason to forecast
returns; `PurgedKFold` cuts the train/test overlap from 100% to 0% and the
score collapses to about −1.2. This is the real-data counterpart to the
controlled `synthetic_leakage_proof.ipynb`.

---

### 6. `model_comparison_honest_cv.ipynb` — Which Model Survives Honest CV?

**Dataset:** Binance spot BTC/USDT daily candles, pinned window
2021-01-01 to 2023-09-28 (1001 bars; shares the cache with notebook 5).

**Source:** the [`pricehub`](https://pypi.org/project/pricehub/) package
(`get_ohlc`), an examples-extra dependency. No API key.

**Files cached:**
- `btcusdt_1d_binance_spot_2021_2023.csv` — reused from notebook 5 if present

**License:** Binance public market data; subject to the exchange's API terms

**Download size:** ~0.1 MB on first run (none if notebook 5 ran first)

Answers "which model predicts price best?" honestly. Ridge, k-NN,
RandomForest and HistGBR (plus a mean baseline) are scored two ways: naive
shuffled k-fold vs `PurgedKFold` R² (the leak gap, per model), then
`CombinatorialPurgedCV` drives a toy long/short strategy whose per-path
Sharpe feeds a Deflated Sharpe Ratio with `n_trials` set to the number of
models and `var_sharpe` estimated across them. On real BTC daily bars with
ordinary momentum/volatility/volume features, no model clears DSR ≥ 0.95 —
no edge survives the correction for having tried several models. That is the
answer, and it is the point of the package.

---

### 7. `uk_smart_meter_lcl.ipynb` — Real UK Smart-Meter Demand (Honest CV Comparison)

**Dataset:** UK Power Networks, Low Carbon London smart-meter trial,
~5,500 households, half-hourly, Nov 2011 – Feb 2014.

**Source:** London Datastore, dataset
`smartmeter-energy-use-data-in-london-households` (the 168-file CSV split).
Download the zip; the notebook builds a small cached subset.

**Files cached:**
- `lcl_halfhourly.csv` — 60 standard-tariff households, fixed-seed sample,
  built from the raw files (git-ignored). Synthetic fallback if absent.

**License:** UK Power Networks / London Datastore open terms.

**Download size:** ~760 MB raw zip (one-time, user-side); cached subset is small.

The first reproducible side-by-side of CV schemes on the canonical LCL
dataset. Honest result, reported as measured: naive shuffled k-fold (WAPE
29.3%) barely beats walk-forward (29.8%) — the temporal-leakage gap is only
~2% here, because the predictable signal is genuine seasonality every split
learns and the half-hourly noise floor is high. The leak that does bite is
by household: `GroupKFold` on unseen customers is ~7% worse than the pooled
temporal estimate, so time-based CV overstates accuracy for new-customer
deployment. Two methodological traps surface and are documented: raw
half-hourly MAPE is undefined near zero (use WAPE), and `PurgedGroupKFold`
degenerates on a fully-overlapping panel (use `GroupKFold` for the group
leak, `WalkForwardSplit` for the temporal one). A nuanced, measured finding
rather than a dramatic one.

---

### 8. `earthquake_magnitude_leakage.ipynb` — A Real Phantom Edge

**Dataset:** USGS global earthquake catalog, M5.0+ events 2014–2023
(~17,200 events).

**Source:** USGS FDSN event API
(`earthquake.usgs.gov/fdsnws/event/1/query`, CSV, no key, fixed query
window for reproducibility).

**Files cached:**
- `usgs_quakes_m5_2014_2024.csv` (~3 MB) — fetched once, then offline

**License:** U.S. Geological Survey, public domain.

**Download size:** ~3 MB on first run

The dramatic, honest, non-financial case. Earthquake magnitude is not
predictable from past magnitudes (Gutenberg-Richter; the empirical
correlation here is +0.02), so the honest R² is ~0 by established science.
The label is the mean magnitude over the next 20 events (overlapping); one
feature is the cumulative event count, a real quantity that doubles as a
monotone clock. Naive shuffled k-fold reports **R² = +0.65** — pure
leakage. `PurgedKFold` (−0.75), blocked k-fold (−1.13) and
`WalkForwardSplit` (−1.24) all collapse to the correct "no skill", and the
train/test label overlap drops from 100% to 0%. The real-data counterpart
to `synthetic_leakage_proof.ipynb`, on a dataset whose unpredictability is
a scientific fact rather than an assumption.
