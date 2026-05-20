"""Internal interval helpers for half-open temporal windows."""

from __future__ import annotations

import numpy as np

from ._typing import NDArrayAny


def _merge_half_open_intervals(
    starts: NDArrayAny, ends: NDArrayAny
) -> tuple[NDArrayAny, NDArrayAny]:
    """Merge non-empty half-open intervals ``[start, end)``."""
    starts_arr = np.asarray(starts)
    ends_arr = np.asarray(ends)
    non_empty = starts_arr < ends_arr
    if not bool(np.any(non_empty)):
        return starts_arr[:0], ends_arr[:0]

    starts_non_empty = starts_arr[non_empty]
    ends_non_empty = ends_arr[non_empty]
    order = np.argsort(starts_non_empty, kind="mergesort")
    sorted_starts = starts_non_empty[order]
    sorted_ends = ends_non_empty[order]

    merged_starts: list[object] = []
    merged_ends: list[object] = []
    for start, end in zip(sorted_starts, sorted_ends, strict=True):
        if not merged_starts or start > merged_ends[-1]:
            merged_starts.append(start)
            merged_ends.append(end)
        elif end > merged_ends[-1]:
            merged_ends[-1] = end

    return np.asarray(merged_starts, dtype=starts_arr.dtype), np.asarray(
        merged_ends, dtype=ends_arr.dtype
    )


def _merge_closed_intervals(starts: NDArrayAny, ends: NDArrayAny) -> tuple[NDArrayAny, NDArrayAny]:
    """Merge non-empty closed intervals ``[start, end]``."""
    starts_arr = np.asarray(starts)
    ends_arr = np.asarray(ends)
    non_empty = starts_arr <= ends_arr
    if not bool(np.any(non_empty)):
        return starts_arr[:0], ends_arr[:0]

    starts_non_empty = starts_arr[non_empty]
    ends_non_empty = ends_arr[non_empty]
    order = np.argsort(starts_non_empty, kind="mergesort")
    sorted_starts = starts_non_empty[order]
    sorted_ends = ends_non_empty[order]

    merged_starts: list[object] = []
    merged_ends: list[object] = []
    for start, end in zip(sorted_starts, sorted_ends, strict=True):
        if not merged_starts or start > merged_ends[-1]:
            merged_starts.append(start)
            merged_ends.append(end)
        elif end > merged_ends[-1]:
            merged_ends[-1] = end

    return np.asarray(merged_starts, dtype=starts_arr.dtype), np.asarray(
        merged_ends, dtype=ends_arr.dtype
    )


def overlaps_any_half_open_interval(
    starts: NDArrayAny,
    ends: NDArrayAny,
    interval_starts: NDArrayAny,
    interval_ends: NDArrayAny,
) -> NDArrayAny:
    """Return mask for rows whose half-open interval overlaps any test interval."""
    starts_arr = np.asarray(starts)
    ends_arr = np.asarray(ends)
    out = np.zeros(len(starts_arr), dtype=bool)
    if len(starts_arr) == 0:
        return out

    merged_starts, merged_ends = _merge_half_open_intervals(interval_starts, interval_ends)
    if len(merged_starts) == 0:
        return out

    non_empty = starts_arr < ends_arr
    if not bool(np.any(non_empty)):
        return out

    row_starts = starts_arr[non_empty]
    row_ends = ends_arr[non_empty]
    # Half-open overlap requires interval_start < row_end, so side="left"
    # excludes intervals that only touch the row's right boundary.
    candidate = np.searchsorted(merged_starts, row_ends, side="left") - 1
    valid = candidate >= 0
    overlaps = np.zeros(len(row_starts), dtype=bool)
    overlaps[valid] = merged_ends[candidate[valid]] > row_starts[valid]
    out[non_empty] = overlaps
    return out


def points_in_any_closed_interval(
    points: NDArrayAny,
    interval_starts: NDArrayAny,
    interval_ends: NDArrayAny,
) -> NDArrayAny:
    """Return mask for points contained in any closed interval."""
    points_arr = np.asarray(points)
    out = np.zeros(len(points_arr), dtype=bool)
    if len(points_arr) == 0:
        return out

    merged_starts, merged_ends = _merge_closed_intervals(interval_starts, interval_ends)
    if len(merged_starts) == 0:
        return out

    candidate = np.searchsorted(merged_starts, points_arr, side="right") - 1
    valid = candidate >= 0
    out[valid] = merged_ends[candidate[valid]] >= points_arr[valid]
    return out
