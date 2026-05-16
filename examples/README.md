# purged-cross-validation Example Notebooks

Four domain-specific notebooks that demonstrate the `purged-cross-validation` library on real-world
time-series problems. Each notebook downloads its dataset on first run and caches it in
`examples/data/` (git-ignored). Subsequent runs use the local cache and work offline.

**First run downloads ~18 MB total; subsequent runs are fully offline.**

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
