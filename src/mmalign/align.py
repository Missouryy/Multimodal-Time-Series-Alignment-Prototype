from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import correlate
from fastdtw import fastdtw

from .data_models import TimeSeries


def _median_dt(ts: TimeSeries) -> float:
    if len(ts.timestamps) < 2:
        return 0.0
    dt = np.diff(ts.timestamps)
    med = float(np.median(dt))
    return med if med > 0 else 0.0


def _infer_rate_hz(streams: Sequence[TimeSeries], default: float = 50.0) -> float:
    rates = []
    for s in streams:
        r = s.rate_hz
        if r is None:
            dt = _median_dt(s)
            if dt > 0:
                r = 1.0 / dt
        if r is not None and r > 0:
            rates.append(r)
    return float(max(rates)) if rates else float(default)


def resample_to_rate(ts: TimeSeries, target_rate_hz: float) -> TimeSeries:
    if len(ts.timestamps) == 0:
        return ts
    t0, t1 = float(ts.timestamps[0]), float(ts.timestamps[-1])
    if t1 <= t0:
        t1 = t0 + 1.0 / max(target_rate_hz, 1.0)
    step = 1.0 / float(target_rate_hz)
    grid = np.arange(t0, t1 + step, step)
    vals = ts.values if ts.values.ndim == 2 else ts.values.reshape(-1, 1)
    interp = []
    for d in range(vals.shape[1]):
        interp.append(np.interp(grid, ts.timestamps, vals[:, d]))
    new_vals = np.stack(interp, axis=1)
    meta = dict(ts.metadata)
    meta["resampled_rate_hz"] = float(target_rate_hz)
    return ts.copy_with(timestamps=grid, values=new_vals, metadata=meta)


def _overlap_range(streams: Sequence[TimeSeries]) -> Tuple[float, float]:
    start = max(float(s.timestamps[0]) for s in streams if len(s.timestamps) > 0)
    end = min(float(s.timestamps[-1]) for s in streams if len(s.timestamps) > 0)
    if end <= start:
        end = start
    return start, end


def overlay_on_common_grid(streams: Sequence[TimeSeries]) -> pd.DataFrame:
    if not streams:
        return pd.DataFrame()
    # Build a common grid over the intersection of time ranges
    start, end = _overlap_range(streams)
    rate = _infer_rate_hz(streams, default=50.0)
    if end <= start:
        # Degenerate: return empty
        return pd.DataFrame({"time": []})
    step = 1.0 / rate
    grid = np.arange(start, end + step, step)
    columns = {"time": grid}
    for s in streams:
        # Resample each to the grid
        s_r = resample_to_rate(s, target_rate_hz=rate)
        # Clip to [start, end]
        mask = (s_r.timestamps >= start) & (s_r.timestamps <= end)
        t = s_r.timestamps[mask]
        v = s_r.values[mask]
        # If lengths mismatch due to edge inclusions, re-interp exactly on grid
        if len(t) != len(grid):
            vals = []
            for d in range(v.shape[1]):
                vals.append(np.interp(grid, s_r.timestamps, s_r.values[:, d]))
            v = np.stack(vals, axis=1)
        for d in range(v.shape[1]):
            columns[f"{s.name}_v{d}"] = v[:, d]
    return pd.DataFrame(columns)


def _to_scalar(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        x = values
    else:
        x = values.mean(axis=1)
    x = x - np.mean(x)
    std = np.std(x) + 1e-8
    return x / std


def estimate_lag_by_xcorr(ref: TimeSeries, tgt: TimeSeries, max_lag_seconds: float = 5.0, rate_hz: float | None = None) -> float:
    rate = float(rate_hz) if rate_hz is not None else _infer_rate_hz([ref, tgt], default=50.0)
    # Use overlapping window and expand by max lag to be safe
    start, end = _overlap_range([ref, tgt])
    start += 0.0
    end -= 0.0
    if end <= start:
        # Fallback to union
        start = min(ref.timestamps[0], tgt.timestamps[0])
        end = max(ref.timestamps[-1], tgt.timestamps[-1])
    # Resample both
    ref_r = resample_to_rate(ref, rate)
    tgt_r = resample_to_rate(tgt, rate)
    # Clip to common range
    start = max(start, ref_r.timestamps[0], tgt_r.timestamps[0])
    end = min(end, ref_r.timestamps[-1], tgt_r.timestamps[-1])
    step = 1.0 / rate
    grid = np.arange(start, end + step, step)
    def interp_on(ts: TimeSeries) -> np.ndarray:
        vals = []
        for d in range(ts.values.shape[1]):
            vals.append(np.interp(grid, ts.timestamps, ts.values[:, d]))
        return np.stack(vals, axis=1)
    xr = _to_scalar(interp_on(ref_r))
    xt = _to_scalar(interp_on(tgt_r))
    max_lag_samples = int(max_lag_seconds * rate)
    corr = correlate(xt, xr, mode="full")
    lags = np.arange(-len(xr) + 1, len(xt))
    # Limit search window
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
    lags = lags[mask]
    corr = corr[mask]
    best_lag_samples = int(lags[np.argmax(corr)])
    # Positive lag means target is delayed relative to ref (target shifted right)
    return float(best_lag_samples / rate)


def shift_series_in_time(ts: TimeSeries, shift_seconds: float) -> TimeSeries:
    return ts.copy_with(timestamps=ts.timestamps + float(shift_seconds))


def dtw_mapping(ref: TimeSeries, tgt: TimeSeries) -> Tuple[List[Tuple[int, int]], float]:
    # Expect same sampling rate; if not, pick a common one
    rate = _infer_rate_hz([ref, tgt], default=50.0)
    ref_r = resample_to_rate(ref, rate)
    tgt_r = resample_to_rate(tgt, rate)
    x = _to_scalar(ref_r.values)
    y = _to_scalar(tgt_r.values)
    # Use mean across dims for path, but mapping will be applied to all dims
    x1 = x if x.ndim == 1 else x
    if x1.ndim > 1:
        x1 = x1.mean(axis=1)
    y1 = y if y.ndim == 1 else y
    if y1.ndim > 1:
        y1 = y1.mean(axis=1)
    dist, path = fastdtw(list(x1), list(y1))
    # path is list of (i_ref, j_tgt)
    return [(int(i), int(j)) for i, j in path], float(dist)


def apply_mapping_and_overlay(ref: TimeSeries, tgt: TimeSeries, mapping: List[Tuple[int, int]]) -> pd.DataFrame:
    # Resample both onto the same rate and use ref timestamps as baseline
    rate = _infer_rate_hz([ref, tgt], default=50.0)
    ref_r = resample_to_rate(ref, rate)
    tgt_r = resample_to_rate(tgt, rate)
    # Aggregate target values for each ref index
    tgt_aligned = np.zeros_like(ref_r.values)
    counts = np.zeros((len(ref_r.timestamps), 1))
    for i, j in mapping:
        if 0 <= i < len(ref_r.timestamps) and 0 <= j < len(tgt_r.timestamps):
            tgt_aligned[i] += tgt_r.values[j]
            counts[i] += 1.0
    counts[counts == 0] = 1.0
    tgt_aligned = tgt_aligned / counts
    df = pd.DataFrame({"time": ref_r.timestamps})
    for d in range(ref_r.values.shape[1]):
        df[f"ref_v{d}"] = ref_r.values[:, d]
    for d in range(tgt_aligned.shape[1]):
        df[f"tgt_aligned_v{d}"] = tgt_aligned[:, d]
    return df
