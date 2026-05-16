"""Internal: CombinatorialPurgedCV (Domain D5.4).

See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley 2018),
chapter 12 section 12.4. The N-choose-K fold enumeration here is the
splitter half of the CPCV idea; backtest path reconstruction (domain D6)
is deferred to Plan C.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from sklearn.base import clone

from purgedcv._base import BaseTemporalSplitter
from purgedcv._paths import reconstruct_paths
from purgedcv._time import HorizonLike


class CombinatorialPurgedCV(BaseTemporalSplitter):
    """Combinatorial Purged Cross-Validation (fold enumeration).

    Partitions the time-ordered samples into ``n_splits`` contiguous group
    blocks. For each combination of ``n_test_groups`` chosen from those
    blocks, yields one fold whose test indices are the union of the
    chosen blocks. Total folds: ``C(n_splits, n_test_groups)``.

    Each group block appears as test in exactly ``C(n_splits - 1,
    n_test_groups - 1)`` folds.

    The base class applies D2 purge and D3 embargo to each fold's train
    set. Combinatorial enumeration here is the splitter half of CPCV;
    backtest path reconstruction (assembling the C(N,K) folds into
    n_paths time-ordered out-of-sample sequences) is a separate domain
    handled in Plan C.

    See *Advances in Financial Machine Learning* (Lopez de Prado, Wiley
    2018), chapter 12 section 12.4, for the original method.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from purgedcv import CombinatorialPurgedCV
        >>> pred = pd.Series(pd.date_range("2024-01-01", periods=24, freq="D"))
        >>> evalu = pred + pd.Timedelta(days=1)
        >>> cv = CombinatorialPurgedCV(
        ...     n_splits=6, n_test_groups=2,
        ...     prediction_times=pred, evaluation_times=evalu,
        ... )
        >>> cv.get_n_splits()
        15
    """

    def __init__(
        self,
        n_splits: int,
        n_test_groups: int,
        *,
        prediction_times: pd.Series,
        evaluation_times: pd.Series,
        purge_horizon: HorizonLike | None = None,
        embargo: HorizonLike | None = None,
    ) -> None:
        """Configure a Combinatorial Purged CV splitter.

        Args:
            n_splits: Number of contiguous group blocks to partition the
                samples into. Must be at least 2.
            n_test_groups: Number of group blocks chosen as the test
                set in each fold. Must be in ``[1, n_splits - 1]``.
            prediction_times: Per-sample prediction times.
            evaluation_times: Per-sample evaluation times.
            purge_horizon: Symmetric padding around the test fold's
                label window; training rows whose label horizon overlaps
                the padded test horizon are dropped. ``None`` means no
                purge.
            embargo: Post-test embargo duration; training rows whose
                prediction time falls in the closed window
                ``[test_eval_max, test_eval_max + embargo]`` are dropped.
                ``None`` means no embargo.

        Raises:
            ValueError: if ``n_splits < 2`` or
                ``n_test_groups`` is not in ``[1, n_splits - 1]``.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
        if n_test_groups < 1 or n_test_groups >= n_splits:
            raise ValueError(
                f"n_test_groups must be in [1, n_splits-1] = [1, {n_splits - 1}], "
                f"got {n_test_groups}."
            )
        super().__init__(
            prediction_times=prediction_times,
            evaluation_times=evaluation_times,
            purge_horizon=purge_horizon,
            embargo=embargo,
        )
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups

    def get_n_splits(
        self,
        X: object = None,  # noqa: N803
        y: object = None,
        groups: object = None,
    ) -> int:
        return comb(self.n_splits, self.n_test_groups)

    def _iter_test_indices(self, n_samples: int) -> list[np.ndarray]:
        group_size, remainder = divmod(n_samples, self.n_splits)
        cursor = 0
        group_indices: list[np.ndarray] = []
        for k in range(self.n_splits):
            sz = group_size + (1 if k < remainder else 0)
            group_indices.append(np.arange(cursor, cursor + sz, dtype=np.int64))
            cursor += sz
        return [
            np.concatenate([group_indices[i] for i in combo])
            for combo in combinations(range(self.n_splits), self.n_test_groups)
        ]

    def backtest_paths(
        self,
        estimator: object,
        X: np.ndarray | pd.DataFrame,  # noqa: N803
        y: np.ndarray | pd.Series,
    ) -> np.ndarray:
        """Fit ``estimator`` on each fold and reconstruct the C(N-1, K-1)
        out-of-sample backtest paths.

        For each of the C(N, K) folds:

        1. Clone the estimator (so per-fold fits do not contaminate each
           other or the original).
        2. Fit on the fold's training set (after purge + embargo).
        3. Predict on the fold's test set.
        4. If the fold has no training rows (e.g. the structural collapse
           where the test horizon spans the entire timeline and purge
           eliminates all training data), the predictions for that fold
           are NaN.

        The per-fold predictions are then handed to :func:`reconstruct_paths`,
        which assembles them into an ``(n_paths, n_samples)`` matrix where
        each row is a complete time-ordered out-of-sample prediction
        sequence.

        Args:
            estimator: A scikit-learn estimator with ``fit(X, y)`` and
                ``predict(X)`` methods.
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.

        Returns:
            ``(n_paths, n_samples)`` array of out-of-sample predictions
            with ``n_paths = C(n_splits - 1, n_test_groups - 1)``.
            Affected rows contain NaN when an upstream fold could not be
            fit.

        Raises:
            AttributeError or TypeError: if ``estimator`` lacks ``fit`` or
                ``predict``.

        Examples:
            >>> import warnings
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.dummy import DummyRegressor
            >>> from sklearn.exceptions import FitFailedWarning
            >>> from purgedcv import CombinatorialPurgedCV
            >>> pred = pd.Series(pd.date_range("2024-01-01", periods=16, freq="D"))
            >>> evalu = pred + pd.Timedelta(days=1)
            >>> cv = CombinatorialPurgedCV(
            ...     n_splits=4, n_test_groups=2,
            ...     prediction_times=pred, evaluation_times=evalu,
            ... )
            >>> X = np.arange(16).reshape(-1, 1).astype(float)
            >>> y = np.arange(16).astype(float)
            >>> with warnings.catch_warnings():
            ...     warnings.simplefilter("ignore", FitFailedWarning)
            ...     paths = cv.backtest_paths(DummyRegressor(strategy="mean"), X, y)
            >>> paths.shape
            (3, 16)
        """
        import warnings

        from sklearn.exceptions import FitFailedWarning

        n_samples = self._n_samples_or_check(X)
        # Convert to ndarray once so per-fold indexing is uniform.
        X_arr = np.asarray(X)  # noqa: N806
        y_arr = np.asarray(y)

        fold_test_indices = list(self._iter_test_indices(n_samples))
        fold_predictions: list[np.ndarray] = []
        for train_idx, test_idx in self.split(X):
            if len(train_idx) == 0:
                warnings.warn(
                    f"Fold has empty train set; predictions for "
                    f"{len(test_idx)} test rows will be NaN.",
                    FitFailedWarning,
                    stacklevel=2,
                )
                fold_predictions.append(np.full(len(test_idx), np.nan, dtype=float))
                continue
            est = clone(estimator)
            est.fit(X_arr[train_idx], y_arr[train_idx])
            preds = np.asarray(est.predict(X_arr[test_idx]), dtype=float)
            fold_predictions.append(preds)

        return reconstruct_paths(
            fold_predictions,
            fold_test_indices,
            self.n_splits,
            self.n_test_groups,
            n_samples,
        )
