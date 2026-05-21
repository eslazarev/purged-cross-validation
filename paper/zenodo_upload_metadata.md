# Zenodo Upload Metadata

Recommended record type for this manuscript: **Publication -> Preprint**.

Reserved manuscript DOI: `10.5281/zenodo.20323362`. This DOI may not resolve
through DOI.org until the Zenodo record is published.

## Files to Upload

- `paper/zenodo_paper.pdf` - main manuscript.
- `paper/zenodo_paper.tex` - LaTeX source.
- `paper/paper.bib` - bibliography source.
- `paper/zenodo_paper.bbl` - resolved bibliography output, useful for archival
  reproducibility.
- `paper/microbench_summary.md` - recorded local microbenchmark output.
- Optional: `paper/zenodo_upload_metadata.md` - this metadata checklist.

Do not upload the JDSSV-specific files for this Zenodo preprint unless you also
want to archive the journal-submission draft separately.

## Basic Information

**Resource type:** Publication

**Publication subtype:** Preprint

**Reserved DOI:** 10.5281/zenodo.20323362

**Title:** purgedcv: Label-Aware Cross-Validation for Overlapping-Horizon Prediction in Python

**Publication date:** 2026-05-21 if publishing today; otherwise use the actual
date when this preprint is first made publicly available.

**Creators:**

- Evgenii Lazarev
  - Affiliation: Independent Researcher
  - ORCID: 0009-0000-1398-7842
  - Email: elazarev@gmail.com

**Description / abstract:**

Cross-validation is routinely used to estimate out-of-sample performance in
statistical learning, but standard shuffled or blocked folds can be invalid when
responses are measured over future intervals. A label such as the mean demand
over the next twelve half-hours, the next-day rainfall amount, or the return
over the next twenty bars overlaps the labels of nearby rows. If overlapping
label intervals are split between training and test sets, the validation score
partly measures information reuse rather than generalization. This article
formalizes split-level conditions for leakage-aware validation in
overlapping-label time-series and panel data, and presents `purgedcv`, a Python
implementation that exposes purging, embargoing, walk-forward validation,
group-purged folds, and combinatorial purged cross-validation through the
`scikit-learn` splitter protocol, with diagnostic assertions for auditing
train/test splits. A controlled experiment with an unpredictable target shows
that shuffled k-fold can report a mean out-of-sample R2 of 0.918 while admitting
complete train/test label overlap. A full-population benchmark on Low Carbon
London smart-meter data shows a more
nuanced case: the temporal leakage gap is small but measurable, whereas the
larger issue is household-level generalization. The software, notebooks, tests,
and benchmark scripts are open source and make the validation choice auditable
rather than implicit.

**Keywords:**

- cross-validation
- data leakage
- time series
- panel data
- model validation
- reproducible software
- Python
- scikit-learn
- purged cross-validation
- embargo
- combinatorial purged cross-validation

**License for the manuscript:** Creative Commons Attribution 4.0 International
(CC-BY-4.0)

**License note:** The associated software remains MIT licensed. The manuscript
PDF and LaTeX source should be released under CC-BY-4.0.

**Language:** English

**Version:** v1

**Access:** Open access / public files

## Related Identifiers

Add the software archive as a related identifier. This is the software concept
DOI; on 2026-05-21 it resolved through DOI.org to the current Zenodo software
record.

- Identifier: `10.5281/zenodo.20312695`
- Scheme: DOI
- Resource type: Software
- Relation: is supplemented by

Add the source repository as an additional related identifier if the Zenodo form
allows URL identifiers:

- Identifier: `https://github.com/eslazarev/purged-cross-validation`
- Scheme: URL
- Resource type: Software
- Relation: is supplemented by

After publishing the manuscript, add the manuscript DOI back to the GitHub
README, `CITATION.cff`, and the software Zenodo record metadata if appropriate.

## Recommended Citation Text

Lazarev, E. (2026). *purgedcv: Label-Aware Cross-Validation for
Overlapping-Horizon Prediction in Python*. Zenodo.
https://doi.org/10.5281/zenodo.20323362

## Notes

This is a preprint and has not been peer reviewed. The associated software is
archived separately on Zenodo under DOI `10.5281/zenodo.20312695`.
