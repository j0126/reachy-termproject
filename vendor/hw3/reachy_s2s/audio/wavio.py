"""16-bit PCM WAV read/write with the standard library (no libsndfile needed)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .convert import as_frames, float_to_int16, int16_to_float


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int) -> Path:
    """Write float32 audio (1-D mono or ``(n, channels)``) as 16-bit PCM."""
    frames = as_frames(samples)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(frames.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(float_to_int16(frames).tobytes())
    return path


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV (8/16/32-bit int) as float32 ``(n, channels)`` and its rate."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width == 2:
        data = int16_to_float(np.frombuffer(raw, dtype="<i2"))
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width {width} bytes in {path}")
    return data.reshape(-1, channels), rate
