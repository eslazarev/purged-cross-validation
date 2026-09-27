# Splitters

Choose a splitter here. Start with [`BaseTemporalSplitter`][purgedcv.BaseTemporalSplitter]
for the shared time, purge, embargo, and group-validation parameters. Each concrete
splitter documents which options it accepts and any differences in behavior.

Use [diagnostics](diagnostics.md) to inspect its folds before fitting a model.

::: purgedcv.BaseTemporalSplitter

::: purgedcv.WalkForwardSplit

::: purgedcv.PurgedKFold

::: purgedcv.PurgedGroupKFold

::: purgedcv.CombinatorialPurgedCV

::: purgedcv.CombinatoriallySymmetricCV
