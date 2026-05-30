# LCL full-population benchmark

- generated: 2026-05-19T04:13:11.836198+00:00
- raw dir: a local Low Carbon London CSV directory (passed via `--raw-dir`, not distributed)
- eligible Std households (>= 17520 half-hours, >= 365 days): **4284**
- subsamples K = 20, households per subsample N = 60, seed = 0
- raw rows scanned: 167,932,474
- elapsed: 53.8 min
- versions: purgedcv (unreleased local development build of the 0.0.x cross-validation code), numpy 2.4.5, pandas 3.0.3, scikit-learn 1.8.0, scipy 1.17.1

| metric | mean | 95% CI low | 95% CI high |
|---|---|---|---|
| naive shuffled k-fold | 41.68 | 40.37 | 42.99 |
| blocked k-fold | 42.43 | 41.07 | 43.80 |
| WalkForwardSplit | 42.36 | 41.01 | 43.71 |
| GroupKFold (household) | 44.92 | 43.38 | 46.45 |
| temporal gap (walk - naive, WAPE pts) | 0.68 | 0.53 | 0.83 |
| temporal gap (relative %) | 1.60 | 1.27 | 1.94 |
| household gap (group - walk, WAPE pts) | 2.56 | 2.08 | 3.03 |
| household gap (relative %) | 6.03 | 4.93 | 7.12 |

WAPE = sum|err| / sum|actual|, in percent. Temporal gap > 0 means naive
shuffled CV looks better than honest walk-forward. Household gap > 0 means
scoring on unseen households is worse than the pooled temporal estimate.
Reproduce: `python tools/lcl_full_benchmark.py --raw-dir <local LCL dir> --k 20 --n 60 --seed 0`.
