"""Sample-format and channel conversions, level measurement."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def as_frames(x: np.ndarray) -> np.ndarray:
    """Return float32 ``(n, channels)``; 1-D input becomes one channel."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def to_mono(x: np.ndarray, channel: str | int = "left") -> np.ndarray:
    """Pick one channel (``"left"``/``"right"``/index) or average (``"mean"``)."""
    frames = as_frames(x)
    if channel == "mean":
        return frames.mean(axis=1, dtype=np.float32)
    if channel == "left":
        index = 0
    elif channel == "right":
        index = 1 if frames.shape[1] > 1 else 0
    elif isinstance(channel, int):
        index = channel
    else:
        raise ValueError(f"unknown channel selector {channel!r}")
    return np.ascontiguousarray(frames[:, index])


def match_channels(x: np.ndarray, channels: int) -> np.ndarray:
    """Duplicate mono to ``channels`` or keep the first ``channels`` columns."""
    frames = as_frames(x)
    if frames.shape[1] == channels:
        return frames
    if frames.shape[1] == 1:
        return np.repeat(frames, channels, axis=1)
    if frames.shape[1] > channels:
        return np.ascontiguousarray(frames[:, :channels])
    return np.repeat(frames[:, :1], channels, axis=1)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype(np.int16)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) / 32768.0


def rms_dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS (full-scale sine = -3 dBFS); -inf-safe floor at -240."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return -240.0
    return float(10.0 * np.log10(np.mean(arr * arr) + _EPS * _EPS))


def peak_dbfs(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return -240.0
    return float(20.0 * np.log10(np.max(np.abs(arr)) + _EPS * _EPS))


def clipping_rate(x: np.ndarray, threshold: float = 0.999) -> float:
    """Fraction of samples at (or beyond) full scale."""
    arr = np.asarray(x)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(arr) >= threshold) / arr.size)


def normalize_peak(x: np.ndarray, target_dbfs: float = -3.0, max_gain_db: float = 30.0) -> tuple[np.ndarray, float]:
    """Scale so the peak reaches ``target_dbfs`` (gain capped at ``max_gain_db``).

    Robot mic recordings of a quiet room peak around -30 dBFS and are inaudible
    when replayed as-is. Returns ``(scaled, applied_gain_db)``; never attenuates
    below unity unless the input would clip.
    """
    arr = np.asarray(x, dtype=np.float32)
    peak = peak_dbfs(arr)
    if peak <= -200.0:
        return arr.copy(), 0.0
    gain_db = min(target_dbfs - peak, max_gain_db)
    if gain_db < 0.0 and peak <= 0.0:
        gain_db = 0.0
    return (arr * np.float32(10.0 ** (gain_db / 20.0))).astype(np.float32), float(gain_db)
