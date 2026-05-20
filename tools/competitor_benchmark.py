"""Controlled leakage benchmark: purgedcv vs competitor CV splitters.

Cross-validation splitters are not models -- they are leakage controls. So a
fair comparison is not "whose accuracy is higher" but "how much leakage does
each splitter admit on a task whose honest answer is known". We reuse the
controlled construction from ``examples/synthetic_leakage_proof.ipynb``:

  * the target is the mean of the next H pure-noise draws -> nothing can
    predict it, the honest R^2 is ~0 (or negative);
  * the only feature is a monotone cumulative "clock" with no real link to
    the target.

Any clearly positive R^2 on this task is fabricated -- it is the train/test
label overlap leaking the answer. We run the *same model* through each
competitor's splitter and record:

  * mean R^2 it reports (<= ~0 is honest; large positive is leakage);
  * mean train/test label-overlap fraction it admits, measured with the
    tool-neutral ``purgedcv.diagnostics.compute_overlap_fraction`` (an
    interval-overlap count, independent of which library produced the folds).

Competitors that do not install/run on a modern stack, or are closed-source,
are NOT guessed at: each splitter is isolated, and a failure is recorded with
its exact reason. Repo-only, deterministic. Writes (git-ignored)
``examples/data/competitor_benchmark.csv`` and ``..._summary.md``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
if str(EXAMPLES.parent) not in sys.path:
    sys.path.insert(0, str(EXAMPLES.parent))

from purgedcv import CombinatorialPurgedCV, PurgedKFold, WalkForwardSplit  # noqa: E402
from purgedcv.diagnostics import compute_overlap_fraction  # noqa: E402

SEED = 0
N = 1500
H = 20
N_SPLITS = 5

Folds = list[tuple[np.ndarray, np.ndarray]]


def make_dataset() -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Target depends only on noise (unpredictable); feature is a clock."""
    rng = np.random.default_rng(SEED)
    e = rng.standard_normal(N + H)
    y = np.array([e[t : t + H].mean() for t in range(N)])
    x = np.cumsum(rng.gamma(shape=2.0, scale=1.0, size=N))
    features = x.reshape(-1, 1)
    pred = pd.Series(pd.date_range("2020-01-01", periods=N, freq="D"))
    evalu = pred + pd.Timedelta(days=H)
    return features, y, pred, evalu


def score_folds(features: np.ndarray, y: np.ndarray, folds: Folds) -> float:
    """Mean out-of-fold R^2 of one fixed model across the given folds."""
    scores: list[float] = []
    for tr, te in folds:
        if len(tr) == 0 or len(te) == 0:
            continue
        model = RandomForestRegressor(n_estimators=120, random_state=SEED)
        model.fit(features[tr], y[tr])
        scores.append(float(r2_score(y[te], model.predict(features[te]))))
    return float(np.mean(scores)) if scores else float("nan")


def mean_overlap(folds: Folds, pred: pd.Series, evalu: pd.Series) -> float:
    fr = [compute_overlap_fraction(tr, te, pred, evalu) for tr, te in folds if len(tr) and len(te)]
    return float(np.mean(fr)) if fr else float("nan")


def _arr(pairs: Any) -> Folds:
    return [(np.asarray(tr), np.asarray(te)) for tr, te in pairs]


# Each builder takes (features, y, pred, evalu) and returns folds, or raises.
# Builders are isolated per splitter so one bad/absent library never hides the
# others and never produces a fabricated number.


def _sk_kfold_shuffle(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    return _arr(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(f))


def _sk_kfold_blocked(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    return _arr(KFold(n_splits=N_SPLITS, shuffle=False).split(f))


def _sk_tss(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    return _arr(TimeSeriesSplit(n_splits=N_SPLITS).split(f))


def _sk_tss_gap(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    return _arr(TimeSeriesSplit(n_splits=N_SPLITS, gap=H).split(f))


def _pcv_purged(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    cv = PurgedKFold(
        n_splits=N_SPLITS,
        prediction_times=p,
        evaluation_times=e,
        purge_horizon=f"{H}D",
        embargo=f"{H}D",
    )
    return _arr(cv.split(f))


def _pcv_walk(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    cv = WalkForwardSplit(
        n_splits=N_SPLITS,
        test_size=N // (N_SPLITS + 1),
        window="expanding",
        prediction_times=p,
        evaluation_times=e,
        purge_horizon=f"{H}D",
    )
    return _arr(cv.split(f))


def _pcv_cpcv(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    cv = CombinatorialPurgedCV(
        n_splits=6,
        n_test_groups=2,
        prediction_times=p,
        evaluation_times=e,
        purge_horizon=f"{H}D",
    )
    return _arr(cv.split(f))


def _tscv_gapkfold(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    from tscv import GapKFold

    return _arr(GapKFold(n_splits=N_SPLITS, gap_before=H, gap_after=H).split(f))


def _tcv_comb(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    from timeseriescv.cross_validation import CombPurgedKFoldCV

    cv = CombPurgedKFoldCV(n_splits=6, n_test_splits=2, embargo_td=pd.Timedelta(days=H))
    return _arr(cv.split(pd.DataFrame(f), pred_times=p, eval_times=e))


def _tcv_walk(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    from timeseriescv.cross_validation import PurgedWalkForwardCV

    cv = PurgedWalkForwardCV(n_splits=6, n_test_splits=1, min_train_splits=2)
    return _arr(cv.split(pd.DataFrame(f), pred_times=p, eval_times=e))


def _mlfinpy_cpkf(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    from mlfinpy.cross_validation import CombinatorialPurgedKFold

    info = pd.Series(e.to_numpy(), index=p.to_numpy())
    cv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, samples_info_sets=info, embargo=0)
    return _arr(cv.split(pd.DataFrame(f, index=p.to_numpy())))


def _risklabai(f: np.ndarray, y: np.ndarray, p: pd.Series, e: pd.Series) -> Folds:
    from RiskLabAI.utils.cross_validation import PurgedKFold as RlPurgedKFold

    info = pd.Series(e.to_numpy(), index=p.to_numpy())
    cv = RlPurgedKFold(n_splits=N_SPLITS, times=info, embargo=0.0)
    return _arr(cv.split(pd.DataFrame(f, index=p.to_numpy())))


SPLITTERS: list[tuple[str, str, Callable[..., Folds]]] = [
    ("sklearn", "KFold(shuffle=True)  [naive baseline]", _sk_kfold_shuffle),
    ("sklearn", "KFold(shuffle=False) [blocked]", _sk_kfold_blocked),
    ("sklearn", "TimeSeriesSplit", _sk_tss),
    ("sklearn", f"TimeSeriesSplit(gap={H})", _sk_tss_gap),
    ("purgedcv", "PurgedKFold", _pcv_purged),
    ("purgedcv", "WalkForwardSplit", _pcv_walk),
    ("purgedcv", "CombinatorialPurgedCV", _pcv_cpcv),
    ("tscv", f"GapKFold(gap_before={H},gap_after={H})", _tscv_gapkfold),
    ("timeseriescv", "CombPurgedKFoldCV", _tcv_comb),
    ("timeseriescv", "PurgedWalkForwardCV", _tcv_walk),
    ("mlfinpy", "CombinatorialPurgedKFold", _mlfinpy_cpkf),
    ("RiskLabAI", "PurgedKFold", _risklabai),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "examples" / "data",
    )
    ap.add_argument(
        "--core-only",
        action="store_true",
        help="run only sklearn + purgedcv (fast, no third-party deps; for the e2e test)",
    )
    ns = ap.parse_args(argv)
    out_dir: Path = ns.out_dir
    splitters = (
        [s for s in SPLITTERS if s[0] in ("sklearn", "purgedcv")] if ns.core_only else SPLITTERS
    )

    features, y, pred, evalu = make_dataset()
    records: list[dict[str, Any]] = []
    for lib, label, builder in splitters:
        name = f"{lib} {label}"
        try:
            folds = builder(features, y, pred, evalu)
            r2 = score_folds(features, y, folds)
            ov = mean_overlap(folds, pred, evalu)
            records.append(
                {
                    "library": lib,
                    "splitter": label,
                    "status": "ran",
                    "mean_r2": round(r2, 4),
                    "mean_overlap": round(ov, 4),
                    "n_folds": len(folds),
                    "reason": "",
                }
            )
            print(
                f"ran      {name:48s} R2={r2:+.3f} overlap={ov:.3f} folds={len(folds)}", flush=True
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0][:140]}"
            records.append(
                {
                    "library": lib,
                    "splitter": label,
                    "status": "NOT RUN",
                    "mean_r2": float("nan"),
                    "mean_overlap": float("nan"),
                    "n_folds": 0,
                    "reason": reason,
                }
            )
            print(f"NOT RUN  {name:48s} {reason}", flush=True)

    df = pd.DataFrame.from_records(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "competitor_benchmark.csv", index=False)

    lines = [
        "# Competitor benchmark -- controlled leakage task (raw results)",
        "",
        "Target = mean of next H pure-noise draws (unpredictable; honest R^2 ~ 0).",
        f"Feature = monotone clock. N={N}, H={H}, seed={SEED}, model = "
        "RandomForestRegressor(n_estimators=120). Large positive R^2 = fabricated",
        "leakage. Overlap = mean train/test label-overlap fraction admitted.",
        "",
        "| library | splitter | status | mean R^2 | mean overlap | folds |",
        "|---|---|---|---|---|---|",
    ]
    for r in records:
        if r["status"] == "ran":
            lines.append(
                f"| {r['library']} | {r['splitter']} | ran | "
                f"{r['mean_r2']:+.3f} | {r['mean_overlap']:.3f} | {r['n_folds']} |"
            )
        else:
            lines.append(f"| {r['library']} | {r['splitter']} | NOT RUN | -- | -- | -- |")
    notrun = [r for r in records if r["status"] != "ran"]
    if notrun:
        lines += ["", "Not run (exact reason, never guessed):"]
        lines += [f"- **{r['library']} {r['splitter']}**: {r['reason']}" for r in notrun]
    lines.append("")
    (out_dir / "competitor_benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {out_dir / 'competitor_benchmark.csv'}")
    print(f"wrote {out_dir / 'competitor_benchmark_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
