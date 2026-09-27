# Why this exists

Time-series validation starts with a question: what information will be available
when the model is used? A row's label may cover a future interval, and several
rows may belong to the same household, patient, or asset. The split needs to
reflect that structure.

`purgedcv` provides label-aware cross-validation through the scikit-learn
splitter interface. It brings purging, embargoes, and backtest-path analysis
into one package, with diagnostics that let users inspect the resulting folds.

## The underlying problem

Consider a label defined as the return over the next five days. Neighbouring
rows share part of that outcome window. If those rows land in different folds,
training and test labels can share information, making validation scores
optimistic. Purging removes training rows whose label intervals overlap test
intervals.

Overlap is not the only concern. Rows just after a test window may still be
serially dependent on it. An embargo excludes a chosen buffer after the test
window; the appropriate buffer depends on the data and the deployment question.
It does not guarantee independence.

The methods follow Marcos López de Prado's *Advances in Financial Machine
Learning* (Wiley, 2018): purging and embargoing in chapter 7, and Combinatorial
Purged Cross-Validation (CPCV) in chapter 12. Bailey and López de Prado's
Probabilistic and Deflated Sharpe Ratio address uncertainty in performance
estimates and selection across multiple trials.

## Related software

Several projects provide tools for time-series validation and financial machine
learning. They have different scopes. The links below describe each project's
own interface and intended use; this is a reading list, not a ranking.

| Project | Documented scope |
|---|---|
| [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | `TimeSeriesSplit` provides ordered train/test splits with a configurable gap measured in samples. |
| [tscv](https://tscv.readthedocs.io/en/latest/) | Time-series cross-validation with gaps between training and test observations. |
| [timeseriescv](https://github.com/sam31415/timeseriescv) | Purged walk-forward and combinatorial cross-validation using prediction and evaluation times. |
| [mlfinlab](https://hudsonthames.org/mlfinlab/) | Financial machine-learning tools, including cross-validation and backtest-overfitting analysis. |
| [mlfinpy](https://mlfinpy.readthedocs.io/en/latest/) | Financial machine-learning tools with documentation on data preparation, modelling, and cross-validation. |
| [RiskLabAI](https://github.com/RiskLabAI/RiskLabAI.py) | Quantitative-finance and financial machine-learning methods, including purged and combinatorial cross-validation. |

## Where this package sits

`purgedcv` focuses on the validation step. It is MIT-licensed and exposes
splitters that can be passed to scikit-learn model-selection functions.
Prediction and evaluation times are explicit inputs, so each observation can
have its own label horizon.

The [API reference](api.md) covers:

- Purge and embargo primitives for building or inspecting a split.
- Walk-forward, purged k-fold, group-purged, and combinatorial splitters.
- CPCV path reconstruction and metrics on the reconstructed paths.
- Sharpe-ratio statistics and backtest-overfitting analysis.
- Fold reports and assertions for temporal overlap and group separation.

The algorithms come from the cited literature. The package's contribution is
an implementation with a shared interface, typed inputs, and executable
examples. Tests cover split invariants and numerical results. They are useful
checks, not a guarantee that a particular research design is valid.

## Choosing a validation design

Start with deployment. Predicting future observations for known households is
a different question from predicting for households never seen in training.
The first calls for a temporal design; the second also needs group separation.

CPCV provides several backtest paths from combinations of test folds. Those
paths share data and are not independent trials. Walk-forward validation may
better match a deployment process that always trains on the past. Neither
choice establishes that a model will generalize.

Purging cannot correct features computed with future information, preprocessing
fitted before cross-validation, or a holdout reused during model selection.
Keep those checks in the experiment too. See the [quickstart](quickstart.md)
for examples and [diagnostics](api/diagnostics.md) for fold-level checks.

## Sources

- López de Prado (2018), *Advances in Financial Machine Learning*, chapters 7 and 12.
- Bailey and López de Prado (2012), *The Sharpe Ratio Efficient Frontier*.
- Bailey and López de Prado (2014), *The Deflated Sharpe Ratio*.

Bibliographic links are listed under
[Methodology references in the README](https://github.com/eslazarev/purged-cross-validation#methodology-references).
