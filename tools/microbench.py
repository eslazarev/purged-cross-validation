"""Microbenchmark for PurgedKFold split generation.

This benchmarks splitter construction and fold generation only. It does not fit
an estimator, so the result measures validation-infrastructure overhead rather
than model runtime.
"""

from __future__ import annotations

import argparse
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from purgedcv import PurgedKFold  # noqa: E402


def _version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        if pkg == "purgedcv":
            import purgedcv

            return getattr(purgedcv, "__version__", "unknown")
        return "unknown"


def _run_once(n: int, n_splits: int, horizon: str) -> tuple[float, list[tuple[int, int]]]:
    prediction_times = pd.Series(pd.date_range("2020-01-01", periods=n, freq="s"))
    evaluation_times = prediction_times + pd.Timedelta(horizon)
    x = np.empty((n, 1), dtype=np.float64)
    cv = PurgedKFold(
        n_splits=n_splits,
        prediction_times=prediction_times,
        evaluation_times=evaluation_times,
        purge_horizon=horizon,
    )
    started = perf_counter()
    fold_sizes = [(len(train_idx), len(test_idx)) for train_idx, test_idx in cv.split(x)]
    return perf_counter() - started, fold_sizes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--horizon", default="20s")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "paper" / "microbench_summary.md")
    args = parser.parse_args(argv)

    timings: list[float] = []
    fold_sizes: list[tuple[int, int]] = []
    for _ in range(args.repeat):
        elapsed, fold_sizes = _run_once(args.n, args.n_splits, args.horizon)
        timings.append(elapsed)

    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "purgedcv": _version("purgedcv"),
        "numpy": _version("numpy"),
        "pandas": _version("pandas"),
        "scikit-learn": _version("scikit-learn"),
        "scipy": _version("scipy"),
    }
    best = min(timings)
    mean = float(np.mean(timings))
    lines = [
        "# PurgedKFold microbenchmark",
        "",
        f"- n: {args.n:,}",
        f"- n_splits: {args.n_splits}",
        f"- horizon: {args.horizon}",
        "- estimator fitting: none",
        f"- timings, seconds: {', '.join(f'{t:.3f}' for t in timings)}",
        f"- best: {best:.3f} s",
        f"- mean: {mean:.3f} s",
        f"- fold sizes (train, test): {fold_sizes}",
        "- environment scope: local microbenchmark environment; table-producing "
        "benchmark versions are reported in examples/data/lcl_full_benchmark_summary.md",
        "- versions: " + ", ".join(f"{name} {value}" for name, value in versions.items()),
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
