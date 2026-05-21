# PurgedKFold microbenchmark

- n: 1,000,000
- n_splits: 5
- horizon: 20s
- estimator fitting: none
- timings, seconds: 1.924, 1.911, 1.898
- best: 1.898 s
- mean: 1.911 s
- fold sizes (train, test): [(799961, 200000), (799922, 200000), (799922, 200000), (799922, 200000), (799961, 200000)]
- environment scope: local microbenchmark environment; table-producing benchmark versions are reported in examples/data/lcl_full_benchmark_summary.md
- versions: python 3.12.7, platform macOS-26.3.1-arm64-arm-64bit, purgedcv 0.0.6, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.8.0, scipy 1.17.1

