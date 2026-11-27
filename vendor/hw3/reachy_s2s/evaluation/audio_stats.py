"""Descriptive statistics of a recording (levels, clipping, channel relationship)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..audio.convert import as_frames, peak_dbfs, rms_dbfs


def recording_stats(frames: np.ndarray, sample_rate: int) -> dict[str, Any]:
    x = as_frames(frames)
    stats: dict[str, Any] = {
        "duration_s": round(x.shape[0] / sample_rate, 3),
        "channels": int(x.shape[1]),
        "clipped_samples": int(np.count_nonzero(np.abs(x) >= 0.999)),
    }
    for c in range(x.shape[1]):
        stats[f"ch{c}_rms_dbfs"] = round(rms_dbfs(x[:, c]), 2)
        stats[f"ch{c}_peak_dbfs"] = round(peak_dbfs(x[:, c]), 2)
        # 10th-percentile 50 ms RMS: a rough noise-floor estimate
        win = max(1, int(0.05 * sample_rate))
        n_win = x.shape[0] // win
        if n_win >= 4:
            blocks = x[: n_win * win, c].reshape(n_win, win).astype(np.float64)
            block_db = 10.0 * np.log10(np.mean(blocks**2, axis=1) + 1e-24)
            stats[f"ch{c}_floor_dbfs_p10"] = round(float(np.percentile(block_db, 10)), 2)
            stats[f"ch{c}_loud_dbfs_p95"] = round(float(np.percentile(block_db, 95)), 2)
    if x.shape[1] >= 2 and x.shape[0] > 1:
        left, right = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
        diff = left - right
        stats["lr_max_abs_diff"] = round(float(np.max(np.abs(diff))), 6)
        stats["lr_diff_rms_dbfs"] = round(rms_dbfs(diff), 2)
        if np.std(left) > 0 and np.std(right) > 0:
            stats["lr_corr"] = round(float(np.corrcoef(left, right)[0, 1]), 6)
        stats["lr_identical"] = bool(stats["lr_max_abs_diff"] < 1e-4)
    return stats
