"""Deterministic test signals (speaker checks, loopback latency probe, unit tests)."""

from __future__ import annotations

import numpy as np


def _fade(x: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    n = min(int(sample_rate * fade_ms / 1000.0), x.shape[0] // 2)
    if n > 0:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, n, dtype=np.float32)) ** 2
        x[:n] *= ramp
        x[-n:] *= ramp[::-1]
    return x


def silence(duration_s: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(round(duration_s * sample_rate)), dtype=np.float32)


def tone(
    freq_hz: float,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.3,
    fade_ms: float = 10.0,
) -> np.ndarray:
    t = np.arange(int(round(duration_s * sample_rate)), dtype=np.float64) / sample_rate
    x = (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    return _fade(x, sample_rate, fade_ms)


def chirp(
    f0_hz: float,
    f1_hz: float,
    duration_s: float,
    sample_rate: int,
    amplitude: float = 0.3,
    fade_ms: float = 5.0,
) -> np.ndarray:
    """Exponential sweep from ``f0_hz`` to ``f1_hz`` (sharp autocorrelation peak)."""
    n = int(round(duration_s * sample_rate))
    t = np.arange(n, dtype=np.float64) / sample_rate
    k = (f1_hz / f0_hz) ** (1.0 / duration_s)
    phase = 2.0 * np.pi * f0_hz * (k**t - 1.0) / np.log(k)
    x = (amplitude * np.sin(phase)).astype(np.float32)
    return _fade(x, sample_rate, fade_ms)
