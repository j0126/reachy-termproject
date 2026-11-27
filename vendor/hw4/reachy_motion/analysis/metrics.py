"""Tracking metrics: RMSE, peak error and lag between a reference and a signal.

Used in Phase 1 to quantify the delay of the *modelled* command pipeline, and intended for
Phase 3 (commanded vs measured). The reference r(t) is linearly interpolated onto the
signal's timestamps, so both may be sampled on different, non-uniform grids.
Only 1-D signals; call per joint for multi-joint data.
"""

from __future__ import annotations

import numpy as np


def _overlap(t_ref: np.ndarray, t: np.ndarray, shift: float = 0.0) -> np.ndarray:
    """Mask of signal samples whose (t - shift) lies inside the reference time span."""
    ts = t - shift
    return (ts >= t_ref[0]) & (ts <= t_ref[-1])


def error_series(
    t_ref: np.ndarray, y_ref: np.ndarray, t: np.ndarray, y: np.ndarray, shift: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """(t_masked, y - r(t - shift)) on the overlap."""
    t_ref, y_ref, t, y = (np.asarray(a, dtype=float) for a in (t_ref, y_ref, t, y))
    m = _overlap(t_ref, t, shift)
    r = np.interp(t[m] - shift, t_ref, y_ref)
    return t[m], y[m] - r


def rmse(t_ref: np.ndarray, y_ref: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """Root-mean-square error of y against r on the overlap."""
    _, e = error_series(t_ref, y_ref, t, y)
    if e.size == 0:
        raise ValueError("signals do not overlap in time")
    return float(np.sqrt(np.mean(e**2)))


def peak_error(t_ref: np.ndarray, y_ref: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """Maximum absolute error of y against r on the overlap."""
    _, e = error_series(t_ref, y_ref, t, y)
    if e.size == 0:
        raise ValueError("signals do not overlap in time")
    return float(np.max(np.abs(e)))


def estimate_lag(
    t_ref: np.ndarray,
    y_ref: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    max_lag: float = 0.2,
    step: float = 0.0005,
) -> float:
    """Time shift L (s) minimising RMSE(y(t), r(t - L)); L > 0 means y lags r.

    Grid search over [-max_lag, max_lag] with resolution `step`. Use a window that contains
    motion: on a constant segment every shift fits equally well.
    """
    t_ref, y_ref, t, y = (np.asarray(a, dtype=float) for a in (t_ref, y_ref, t, y))
    best_lag, best_err = 0.0, np.inf
    for lag in np.arange(-max_lag, max_lag + step / 2, step):
        m = _overlap(t_ref, t, lag)
        if m.sum() < 3:
            continue
        r = np.interp(t[m] - lag, t_ref, y_ref)
        err = float(np.sqrt(np.mean((y[m] - r) ** 2)))
        if err < best_err - 1e-15:
            best_lag, best_err = float(lag), err
    if not np.isfinite(best_err):
        raise ValueError("signals do not overlap in time for any lag")
    return best_lag
